from rag.chunking import chunk
from services.quiz_service import mastery_from_rows
from services.flashcard_service import sm2
from agents.quality_agent import check
from agents import curriculum_agent, lesson_agent, orchestrator, tutor_agent
from ingestion import youtube_loader
from models.schemas import (
    Assessment, CoursePlan, Lesson, LessonBatch, LessonBatchItem, LessonOutline, MCQ,
    ModuleOutline, Outline, SourceTopic, SourceTopicCatalog
)
from ingestion.text_cleaner import clean
from services import llm_client
from services.lesson_content import display_items, display_text
from types import SimpleNamespace
import json
import pytest
import threading

def test_chunk_covers_all_text():
    segs = [{"start": i, "end": i + 1, "text": f"sentence number {i} " * 5} for i in range(40)]
    cs = chunk(segs, size=300, overlap=50)
    assert len(cs) > 1 and "number 39" in cs[-1]["text"]

def test_topic_batches_preserve_the_entire_transcript():
    chunks = [
        {"start": i * 60, "end": (i + 1) * 60, "text": f"topic-{i} " * 30 + f"tail-{i}"}
        for i in range(8)
    ]

    batches = curriculum_agent._topic_batches(chunks, limit=200)

    assert len(batches) > 1
    assert all(f"topic-{i}" in "\n".join(batches) for i in range(8))
    assert all(f"tail-{i}" in "\n".join(batches) for i in range(8))

def test_outline_catalogs_all_batches_before_building_outline(monkeypatch):
    calls = []
    expected_outline = Outline(
        title="Course", description="", audience="", difficulty="beginner",
        prerequisites=[], objectives=[], modules=[], estimated_minutes=10
    )

    def generate(prompt, schema):
        calls.append((prompt, schema))
        if schema is SourceTopicCatalog:
            title = f"Topic {prompt.split('transcript part ')[1].split(' of')[0]}"
            return SourceTopicCatalog(topics=[
                SourceTopic(title=title, description="Covered topic", timestamp="00:00:00")
            ])
        return expected_outline

    monkeypatch.setattr(curriculum_agent.llm_client, "generate", generate)
    chunks = [
        {"start": i * 60, "end": (i + 1) * 60,
         "text": f"section-{i} " + "x" * 5000 + f"tail-{i}"}
        for i in range(13)
    ]

    result = curriculum_agent.outline(chunks, {"level": "beginner", "style": "practical", "depth": "short"})

    assert result.title == expected_outline.title
    assert sum(schema is SourceTopicCatalog for _, schema in calls) > 1
    outline_prompt, outline_schema = calls[-1]
    assert outline_schema is Outline
    assert all(f"section-{i}" in "\n".join(prompt for prompt, _ in calls) for i in range(13))
    assert all(f"tail-{i}" in "\n".join(prompt for prompt, _ in calls) for i in range(13))
    assert "Topic 1" in outline_prompt
    assert f"Topic {sum(schema is SourceTopicCatalog for _, schema in calls)}" in outline_prompt
    lessons = [lesson for module in result.modules for lesson in module.lessons]
    assert len(lessons) == sum(schema is SourceTopicCatalog for _, schema in calls)
    assert all(lesson.key_topics == [lesson.title] for lesson in lessons)

def test_outline_combines_topic_extraction_and_outline_for_fitting_transcript(monkeypatch):
    topic = SourceTopic(title="Variables", description="Storing values", timestamp="00:00:10")
    generated_outline = Outline(
        title="Python", description="", audience="", difficulty="beginner",
        prerequisites=[], objectives=[], modules=[ModuleOutline(
            title="Basics", description="", lessons=[]
        )], estimated_minutes=10
    )
    calls = []

    def generate(prompt, schema):
        calls.append((prompt, schema))
        assert schema is CoursePlan
        return CoursePlan(topics=[topic], outline=generated_outline)

    monkeypatch.setattr(curriculum_agent.llm_client, "generate", generate)
    result = curriculum_agent.outline(
        [{"start": 0, "end": 10, "text": "A short transcript about variables."}],
        {"level": "beginner", "style": "practical", "depth": "short"},
    )

    assert len(calls) == 1
    assert result.modules[0].lessons[0].title == "Variables"

def test_outline_splits_combined_lessons_and_adds_missing_topics():
    from models.schemas import LessonOutline, Outline

    generated = Outline(
        title="Course", description="", audience="", difficulty="beginner",
        prerequisites=[], objectives=[], estimated_minutes=30,
        modules=[ModuleOutline(
            title="Foundations", description="Basics",
            lessons=[LessonOutline(
                title="Variables", objectives=["Use variables"], key_topics=["Variables", "Constants"],
                estimated_minutes=12,
            )],
        )],
    )
    topics = [
        SourceTopic(title=title, description=f"About {title}", timestamp=f"00:00:{index:02d}")
        for index, title in enumerate(("Variables", "Constants", "Variable names"))
    ]

    result = curriculum_agent._one_lesson_per_topic(generated, topics)
    lessons = [lesson for module in result.modules for lesson in module.lessons]

    assert [lesson.title for lesson in lessons] == [topic.title for topic in topics]
    assert all(lesson.key_topics == [lesson.title] for lesson in lessons)
    assert len(lessons) == 3

def test_course_generates_lessons_in_concurrent_batches_and_persists_them(monkeypatch):
    barrier = threading.Barrier(6)
    inserted = []
    progress = []

    def generate_batch(_course_title, lesson_outlines, _contexts, _opts):
        barrier.wait(timeout=5)
        return [
            SimpleNamespace(
                model_dump_json=lambda title=outline.title: json.dumps({"title": title}),
                summary=f"Summary for {outline.title}",
            )
            for outline in lesson_outlines
        ]

    monkeypatch.setattr(
        orchestrator, "_contexts",
        lambda _cid, titles, _chunks, _indexed: [("context", []) for _ in titles],
    )
    monkeypatch.setattr(orchestrator.lesson_agent, "generate_batch", generate_batch)
    monkeypatch.setattr(orchestrator.db, "x", lambda *args: inserted.append(args))
    tasks = [
        ("Module", position, SimpleNamespace(title=f"Lesson {position}"))
        for position in range(1, 13)
    ]

    orchestrator._generate_lessons(
        1, tasks, "Course", [], False, {}, 12, 0,
        lambda value, message: progress.append((value, message)),
    )

    lesson_inserts = [call for call in inserted if call[0].startswith("insert into lessons")]
    assert len(lesson_inserts) == 12
    assert sorted(call[1][2] for call in lesson_inserts) == list(range(1, 13))
    assert len([call for call in inserted if call[0].startswith("insert or replace into jobs")]) == 12
    assert len(progress) == 12
    assert max(value for value, _ in progress) == pytest.approx(0.7)

def test_embed_reports_progress_for_each_batch(monkeypatch):
    progress = []

    class Models:
        def embed_content(self, **kwargs):
            return SimpleNamespace(
                embeddings=[SimpleNamespace(values=[len(text)]) for text in kwargs["contents"]]
            )

    monkeypatch.setattr(llm_client, "_get", lambda: SimpleNamespace(models=Models()))

    result = llm_client.embed(
        ["one", "two", "three"], batch=2,
        progress=lambda done, total, batch_number, batch_total:
            progress.append((done, total, batch_number, batch_total)),
    )

    assert result == [[3], [3], [5]]
    assert progress == [(2, 3, 1, 2), (3, 3, 2, 2)]

def test_retrieve_many_embeds_queries_in_one_call(monkeypatch):
    from rag import retriever

    embedded = []
    searched = []
    monkeypatch.setattr(
        retriever.llm_client,
        "embed",
        lambda questions: embedded.append(questions) or [[1], [2]],
    )

    def search(course_id, embedding, k):
        searched.append((course_id, embedding, k))
        return [(f"source-{embedding[0]}", {"start": 0, "end": 10}, 0.1)]

    monkeypatch.setattr(retriever.vector_store, "search", search)
    results = retriever.retrieve_many(7, ["Variables", "Loops"], k=3)

    assert embedded == [["Variables", "Loops"]]
    assert searched == [(7, [1], 3), (7, [2], 3)]
    assert [group[0]["text"] for group in results] == ["source-1", "source-2"]

def test_lesson_batch_generates_separate_ordered_lessons(monkeypatch):
    prompts = []

    def lesson(title):
        return Lesson(
            title=title, introduction="", explanation="", concepts=[], examples=[],
            applications=[], terminology=[], summary="", knowledge_checks=[]
        )

    monkeypatch.setattr(
        lesson_agent.llm_client,
        "generate",
        lambda prompt, schema: prompts.append((prompt, schema)) or LessonBatch(lessons=[
            LessonBatchItem(outline_title="Loops", lesson=lesson("wrong")),
            LessonBatchItem(outline_title="Variables", lesson=lesson("wrong")),
        ]),
    )
    outlines = [
        LessonOutline(title="Variables", objectives=["Explain variables"], key_topics=["Variables"]),
        LessonOutline(title="Loops", objectives=["Explain loops"], key_topics=["Loops"]),
    ]

    results = lesson_agent.generate_batch(
        "Python", outlines, ["Variable source", "Loop source"],
        {"level": "beginner", "style": "practical", "depth": "short"},
    )

    assert [item.title for item in results] == ["Variables", "Loops"]
    prompt, schema = prompts[0]
    assert schema is LessonBatch
    assert "Do not combine topics or omit any requested lesson." in prompt
    assert "Variable source" in prompt and "Loop source" in prompt

def test_lesson_batch_rejects_missing_lesson(monkeypatch):
    lesson = Lesson(
        title="Variables", introduction="", explanation="", concepts=[], examples=[],
        applications=[], terminology=[], summary="", knowledge_checks=[]
    )
    monkeypatch.setattr(
        lesson_agent.llm_client,
        "generate",
        lambda *_args, **_kwargs: LessonBatch(lessons=[
            LessonBatchItem(outline_title="Variables", lesson=lesson)
        ]),
    )
    outlines = [
        LessonOutline(title="Variables", objectives=[], key_topics=[]),
        LessonOutline(title="Loops", objectives=[], key_topics=[]),
    ]

    with pytest.raises(ValueError, match="exactly one result"):
        lesson_agent.generate_batch(
            "Python", outlines, ["a", "b"],
            {"level": "beginner", "style": "practical", "depth": "short"},
        )

def test_delete_course_removes_database_record_and_vector_index(monkeypatch):
    deleted_rows, deleted_vectors = [], []
    monkeypatch.setattr(
        orchestrator.db, "x", lambda sql, args: deleted_rows.append((sql, args))
    )
    monkeypatch.setattr(orchestrator.db, "q", lambda sql, args: [])
    monkeypatch.setattr(orchestrator.vector_store, "delete", deleted_vectors.append)

    orchestrator.delete_course(42)

    assert deleted_rows == [("delete from courses where id=?", (42,))]
    assert deleted_vectors == [42]

def test_recover_interrupted_generation_keeps_saved_lessons(monkeypatch):
    queries = []
    updates = []
    job_updates = []
    monkeypatch.setattr(
        orchestrator.db,
        "q",
        lambda sql, args: queries.append((sql, args)) or [{"status": "generating"}],
    )
    monkeypatch.setattr(
        orchestrator.db,
        "x",
        lambda sql, args: updates.append((sql, args)),
    )
    monkeypatch.setattr(
        orchestrator,
        "_job",
        lambda *args: job_updates.append(args),
    )

    assert orchestrator.recover_interrupted_generation(7) is True

    assert updates == [("update courses set status='failed' where id=?", (7,))]
    assert job_updates[0][0:3] == (7, "failed", 0.0)
    assert "Saved lessons are kept" in job_updates[0][4]
    assert "select status from jobs where course_id=?" == queries[0][0]

def test_recover_does_not_mark_active_generation_interrupted(monkeypatch):
    queries = []
    monkeypatch.setattr(
        orchestrator.db,
        "q",
        lambda *args: queries.append(args) or [{"status": "generating"}],
    )
    monkeypatch.setattr(orchestrator.db, "x", lambda *_args: pytest.fail("must not update"))
    monkeypatch.setattr(orchestrator, "_job", lambda *_args: pytest.fail("must not replace job"))

    with orchestrator._generation_state_lock:
        orchestrator._active_generations.add(7)
    try:
        assert orchestrator.recover_interrupted_generation(7) is False
    finally:
        with orchestrator._generation_state_lock:
            orchestrator._active_generations.discard(7)

    assert queries == []

def test_delete_course_reports_database_delete_failure(monkeypatch):
    monkeypatch.setattr(orchestrator.db, "x", lambda *args: None)
    monkeypatch.setattr(orchestrator.db, "q", lambda *args: [{"id": 42}])
    vector_deleted = []
    monkeypatch.setattr(orchestrator.vector_store, "delete", vector_deleted.append)

    with pytest.raises(RuntimeError, match="could not be deleted"):
        orchestrator.delete_course(42)

    assert vector_deleted == []

def test_lesson_display_filters_null_and_nested_null_values():
    assert display_items(["Keep this", None, "NULL", ["Also keep", None, "None"]]) == [
        "Keep this", "Also keep"
    ]
    assert display_items([None, "null", "   "]) == []
    assert display_text(None) == ""
    assert display_text(" NULL ") == ""
    assert display_text("A useful explanation") == "A useful explanation"

def test_lesson_prompt_requests_clear_stepwise_topic_explanation(monkeypatch):
    prompts = []
    expected = object()
    monkeypatch.setattr(
        lesson_agent.llm_client,
        "generate",
        lambda prompt, schema: prompts.append((prompt, schema)) or expected,
    )
    outline = LessonOutline(
        title="Variables", objectives=["Explain variables"], key_topics=["Variables"]
    )

    result = lesson_agent.generate(
        "Python", outline, "Variables store values.", {"level": "beginner", "style": "practical", "depth": "detailed"}
    )

    assert result is expected
    prompt, schema = prompts[0]
    assert schema is Lesson
    assert "show how it works step by step" in prompt
    assert "thorough walkthrough" in prompt
    assert "real-world use case" in prompt
    assert "meaning of the topic and why learners use it" in prompt

def test_retry_delay_reads_gemini_retry_info():
    error = RuntimeError(
        "429 RESOURCE_EXHAUSTED: retry in 50.829554014s. "
        '{"retryDelay":"50s"}'
    )

    assert llm_client._retry_delay(error) == pytest.approx(50.829554014)

def test_retry_delay_reads_hours_minutes_and_seconds():
    error = RuntimeError("retry in 13h14m58.200587839s")

    assert llm_client._retry_delay(error) == pytest.approx(47698.200587839)

def test_generate_does_not_sleep_for_long_quota_reset(monkeypatch):
    waits = []

    class Models:
        def generate_content(self, **_kwargs):
            raise RuntimeError("429 RESOURCE_EXHAUSTED: retry in 13h14m58s")

    monkeypatch.setattr(llm_client, "_get", lambda: SimpleNamespace(models=Models()))
    monkeypatch.setattr(llm_client, "_pace_request", lambda: None)
    monkeypatch.setattr(llm_client.time, "sleep", waits.append)

    with pytest.raises(RuntimeError, match="Automatic retry was skipped"):
        llm_client.generate("prompt")

    assert waits == []

def test_generate_waits_for_quota_retry_delay(monkeypatch):
    waits = []
    responses = iter([
        RuntimeError("429 RESOURCE_EXHAUSTED: retry in 50s."),
        SimpleNamespace(text="ok"),
    ])

    class Models:
        def generate_content(self, **_kwargs):
            response = next(responses)
            if isinstance(response, Exception):
                raise response
            return response

    monkeypatch.setattr(llm_client, "_get", lambda: SimpleNamespace(models=Models()))
    monkeypatch.setattr(llm_client, "_pace_request", lambda: None)
    monkeypatch.setattr(llm_client.time, "sleep", waits.append)

    assert llm_client.generate("prompt", retries=2) == "ok"
    assert waits == [50.0]

def test_mastery_needs_min_attempts():
    rows = [{"topic": "a", "difficulty": "easy", "is_correct": 1}] * 2
    assert mastery_from_rows(rows)["a"]["mastery"] is None
    rows += [{"topic": "a", "difficulty": "hard", "is_correct": 0}]
    assert mastery_from_rows(rows)["a"]["mastery"] == 50.0

def test_sm2():
    assert sm2(2.5, 6, 2, 1)[1:] == (1, 0)
    e, i, r = sm2(2.5, 1, 1, 4)
    assert i == 6 and r == 2

def test_quality_drops_bad_mcq():
    ok = MCQ(prompt="A?", options=["1", "2", "3", "4"], answer_index=0, explanation="x", topic="t", difficulty="easy")
    dup = ok.model_copy()
    bad = MCQ(prompt="B?", options=["1", "1", "3", "4"], answer_index=0, explanation="x", topic="t", difficulty="easy")
    a, rep = check(Assessment(mcqs=[ok, dup, bad], qas=[], cards=[]))
    assert len(a.mcqs) == 1 and len(rep["dropped_mcq"]) == 2

def test_empty_transcript_rejected():
    with pytest.raises(ValueError):
        clean([{"start": 0, "end": 1, "text": "  "}])

def test_structured_teaching_fields_default_for_older_content():
    lesson = Lesson(
        title="Foundations", introduction="", explanation="", concepts=[], examples=[],
        applications=[], terminology=[], summary="", knowledge_checks=[]
    )
    module = ModuleOutline(title="Foundations", description="", lessons=[])
    assert lesson.guided_practice == []
    assert lesson.why_it_matters == ""
    assert module.objectives == []
    assert module.assessment == ""

def test_tutor_saves_answer_when_vector_retrieval_fails(monkeypatch):
    saved = []
    monkeypatch.setattr(tutor_agent.db, "q", lambda *args: [])
    monkeypatch.setattr(tutor_agent.db, "x", lambda *args: saved.append(args) or len(saved))
    monkeypatch.setattr(tutor_agent.retriever, "retrieve",
                        lambda *args: (_ for _ in ()).throw(RuntimeError("index unavailable")))
    monkeypatch.setattr(tutor_agent, "_keyword_retrieve",
                        lambda *args: [{"text": "Machine learning frameworks.", "ref": "0:00-0:10", "score": 1}])
    monkeypatch.setattr(tutor_agent.llm_client, "generate", lambda *args: "Here is the answer.")

    text, citations = tutor_agent.answer(1, "What frameworks are used?")

    assert "Here is the answer." in text
    assert citations == ["0:00-0:10"]
    assert saved[-1][1][1] == "assistant"
    assert "index is unavailable" in saved[-1][1][2]

def test_tutor_persists_no_match_when_vector_index_fails(monkeypatch):
    saved = []
    monkeypatch.setattr(tutor_agent.db, "q", lambda *args: [])
    monkeypatch.setattr(tutor_agent.db, "x", lambda *args: saved.append(args) or len(saved))
    monkeypatch.setattr(tutor_agent.retriever, "retrieve",
                        lambda *args: (_ for _ in ()).throw(RuntimeError("index unavailable")))
    monkeypatch.setattr(tutor_agent, "_keyword_retrieve", lambda *args: [])
    monkeypatch.setattr(tutor_agent.llm_client, "generate",
                        lambda prompt: "Added context: Java is used to build many Android and server applications.")

    text, citations = tutor_agent.answer(1, "A question with no matches")

    assert "Added context" in text
    assert "general knowledge only" in text
    assert citations == []
    assert saved[-1][1][1] == "assistant"

def test_tutor_uses_keyword_search_when_vector_search_returns_no_hits(monkeypatch):
    saved = []
    keyword_hits = [{"text": "Variables store values in a program.", "ref": "0:10-0:20", "score": 1}]
    prompts = []
    monkeypatch.setattr(tutor_agent.db, "q", lambda *args: [])
    monkeypatch.setattr(tutor_agent.db, "x", lambda *args: saved.append(args) or len(saved))
    monkeypatch.setattr(tutor_agent.retriever, "retrieve", lambda *args: [])
    monkeypatch.setattr(tutor_agent, "_keyword_retrieve", lambda *args: keyword_hits)
    monkeypatch.setattr(tutor_agent.llm_client, "generate",
                        lambda prompt: prompts.append(prompt) or "Variables store values.")

    text, citations = tutor_agent.answer(1, "what is variables")

    assert text == "Variables store values."
    assert citations == ["0:10-0:20"]
    assert "Variables store values in a program." in prompts[0]
    assert saved[-1][1][1] == "assistant"

def test_youtube_loader_translates_available_non_english_captions(monkeypatch):
    import youtube_transcript_api
    from youtube_transcript_api._errors import NoTranscriptFound

    caption = SimpleNamespace(start=0.0, duration=2.0, text="Hola")

    class FetchedTranscript:
        language_code = "en"

        def __iter__(self):
            return iter([caption])

    class Transcript:
        is_translatable = True

        def translate(self, language):
            assert language == "en"
            return self

        def fetch(self):
            return FetchedTranscript()

    class TranscriptApi:
        def fetch(self, video_id, languages):
            raise NoTranscriptFound(video_id, languages, None)

        def list(self, video_id):
            return [Transcript()]

    monkeypatch.setattr(youtube_transcript_api, "YouTubeTranscriptApi", TranscriptApi)

    segments, language = youtube_loader.fetch("https://youtu.be/abcdefghijk")

    assert language == "en"
    assert segments == [{"start": 0.0, "end": 2.0, "text": "Hola"}]

def test_youtube_loader_explains_when_captions_are_absent(monkeypatch):
    import youtube_transcript_api
    from youtube_transcript_api._errors import NoTranscriptFound

    class TranscriptApi:
        def fetch(self, video_id, languages):
            raise NoTranscriptFound(video_id, languages, None)

        def list(self, video_id):
            return []

    monkeypatch.setattr(youtube_transcript_api, "YouTubeTranscriptApi", TranscriptApi)

    with pytest.raises(RuntimeError, match="no captions available"):
        youtube_loader.fetch("https://youtu.be/abcdefghijk")
