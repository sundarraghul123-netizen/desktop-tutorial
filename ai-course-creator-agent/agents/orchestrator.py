from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import logging
import threading
from config import settings
from models import database as db
from models.schemas import Outline
from ingestion.text_cleaner import clean
from rag.chunking import chunk, sample
from rag import vector_store, retriever
from agents import curriculum_agent, lesson_agent, assessment_agent, quality_agent
from services import llm_client

LESSON_GENERATION_WORKERS = settings.MODEL_CONCURRENCY
LESSONS_PER_MODEL_REQUEST = 2
_generation_state_lock = threading.Lock()
_active_generations = set()

def _job(cid, status, progress, msg="", err=""):
    db.x("insert or replace into jobs(course_id,status,progress,message,error) values(?,?,?,?,?)",
         (cid, status, progress, msg, err))

def is_generation_active(cid):
    with _generation_state_lock:
        return cid in _active_generations

def recover_interrupted_generation(cid):
    with _generation_state_lock:
        if cid in _active_generations:
            return False
        jobs = db.q("select status from jobs where course_id=?", (cid,))
        if not jobs or jobs[0]["status"] != "generating":
            return False
        message = "Previous generation stopped unexpectedly. Saved lessons are kept; retry to resume."
        db.x("update courses set status='failed' where id=?", (cid,))
        _job(cid, "failed", 0.0, message, message)
        return True

def _segs(cid):
    return db.q("select start_s as start,end_s as end,text from segments where course_id=? order by id", (cid,))

def create_course(source_type, url, title, segs, language=""):
    segs = clean(segs)
    cid = db.x("insert into courses(title,source_type,source_url,source_title,status) values(?,?,?,?,?)",
               (title, source_type, url, title, "transcribed"))
    db.many("insert into segments(course_id,start_s,end_s,text,language) values(?,?,?,?,?)",
            [(cid, s["start"], s["end"], s["text"], language) for s in segs])
    _job(cid, "transcribed", 0.1)
    return cid

def generate_outline(cid, opts):
    source_chunks = chunk(_segs(cid))
    o = curriculum_agent.outline(source_chunks, opts)
    db.x("update courses set title=?,description=?,difficulty=?,settings_json=?,outline_json=?,status=? where id=?",
         (o.title, o.description, o.difficulty, json.dumps(opts), o.model_dump_json(), "awaiting_approval", cid))
    _job(cid, "awaiting_approval", 0.2, "Review the outline, then approve.")
    return o

def _context(cid, title, chunks, indexed):
    return _contexts(cid, [title], chunks, indexed)[0]

def _contexts(cid, titles, chunks, indexed):
    hit_groups = [[] for _ in titles]
    if indexed and titles:
        try:
            hit_groups = retriever.retrieve_many(cid, titles)
        except Exception as error:
            logging.warning("Lesson transcript retrieval failed for course %s: %s", cid, error)

    fallback = sample(chunks, 6000)
    contexts = []
    for hits in hit_groups:
        if hits:
            contexts.append((
                "\n\n".join(f"[{hit['ref']}] {hit['text']}" for hit in hits),
                [hit["ref"] for hit in hits],
            ))
        else:
            contexts.append((fallback, []))
    return contexts

def _generate_lessons(cid, tasks, course_title, chunks, indexed, opts, total, completed_before, cb):
    if not tasks:
        return

    batches = [
        tasks[index:index + LESSONS_PER_MODEL_REQUEST]
        for index in range(0, len(tasks), LESSONS_PER_MODEL_REQUEST)
    ]

    def generate(batch):
        contexts = _contexts(
            cid, [task[2].title for task in batch], chunks, indexed
        )
        lessons = lesson_agent.generate_batch(
            course_title,
            [task[2] for task in batch],
            [context for context, _ in contexts],
            opts,
        )
        return batch, lessons, [refs for _, refs in contexts]

    failures = []
    completed = completed_before
    with ThreadPoolExecutor(max_workers=min(LESSON_GENERATION_WORKERS, len(batches))) as executor:
        futures = {executor.submit(generate, batch): batch for batch in batches}
        for future in as_completed(futures):
            batch = futures[future]
            try:
                _, lessons, refs_list = future.result()
            except Exception as error:
                failures.append(error)
                continue
            for task, lesson, refs in zip(batch, lessons, refs_list):
                module_title, position, lesson_outline = task
                db.x("insert into lessons(course_id,module_title,position,title,content_json,summary,sources) "
                     "values(?,?,?,?,?,?,?)", (cid, module_title, position, lesson_outline.title,
                                               lesson.model_dump_json(), lesson.summary, json.dumps(refs)))
                completed += 1
                progress = 0.1 + 0.6 * completed / total
                message = f"Lesson {completed}/{total}: {lesson_outline.title}"
                _job(cid, "generating", progress, message)
                cb(progress, message)
    if failures:
        raise failures[0]

def _generate_course(cid, cb):
    c = db.q("select * from courses where id=?", (cid,))[0]
    opts, outline = json.loads(c["settings_json"]), Outline(**json.loads(c["outline_json"]))
    try:
        chunks = chunk(_segs(cid))
        db.x("update courses set status='generating' where id=?", (cid,))
        _job(cid, "generating", 0.02, "Preparing transcript search index.")
        cb(0.02, "Preparing transcript")
        indexed = True
        try:
            vector_store.delete(cid)
            def embedding_progress(done, total, batch_number, batch_total):
                progress = 0.03 + 0.07 * done / max(total, 1)
                message = f"Indexing transcript: {done}/{total} chunks (batch {batch_number}/{batch_total})"
                _job(cid, "generating", progress, message)
                cb(progress, message)

            embeddings = llm_client.embed(
                [item["text"] for item in chunks],
                progress=embedding_progress,
            )
            vector_store.index(cid, chunks, embeddings)
            _job(cid, "generating", 0.1, f"Transcript index ready ({len(chunks)} chunks).")
            cb(0.1, f"Transcript index ready ({len(chunks)} chunks)")
        except Exception as e:  # keep the course; tutor is unavailable until re-indexed
            indexed = False
            logging.exception("Transcript indexing failed for course %s; continuing without tutor index", cid)
            if "RESOURCE_EXHAUSTED" in str(e) or "429" in str(e):
                retry_after = llm_client._retry_delay(e)
                retry_text = (
                    f" Gemini may allow indexing again in about {round(retry_after)} seconds."
                    if retry_after is not None else ""
                )
                message = (
                    "Transcript search indexing hit Gemini's embedding quota. "
                    "Continuing lesson generation without tutor search." + retry_text
                )
            else:
                message = (
                    f"Tutor index unavailable ({type(e).__name__}). "
                    "Continuing with lesson generation."
                )
            _job(cid, "generating", 0.1, message)
            cb(0.1, message)
        done = {r["title"] for r in db.q("select title from lessons where course_id=?", (cid,))}
        total = sum(len(m.lessons) for m in outline.modules)
        tasks = []
        pos = 0
        for m in outline.modules:
            for lo in m.lessons:
                pos += 1
                if lo.title in done:
                    continue  # resume support
                tasks.append((m.title, pos, lo))
        _generate_lessons(cid, tasks, outline.title, chunks, indexed, opts, total, len(done), cb)
        if not db.q("select 1 from questions where course_id=? limit 1", (cid,)):
            cb(0.8, "Creating assessments")
            digest = "\n".join(f"{r['title']}: {r['summary']}" for r in
                               db.q("select title,summary from lessons where course_id=? order by position", (cid,)))
            a = assessment_agent.generate(outline.title, digest, opts["n_mcq"], opts["n_qa"], opts["n_cards"])
            cb(0.9, "Quality checks")
            a, report = quality_agent.check(a)
            for m in a.mcqs:
                db.x("insert into questions(course_id,qtype,prompt,options_json,answer,explanation,topic,difficulty) "
                     "values(?,?,?,?,?,?,?,?)", (cid, "mcq", m.prompt, json.dumps(m.options), m.options[m.answer_index],
                                                  m.explanation, m.topic, m.difficulty))
            for q in a.qas:
                db.x("insert into questions(course_id,qtype,prompt,options_json,answer,explanation,topic,difficulty) "
                     "values(?,?,?,?,?,?,?,?)", (cid, q.qtype, q.prompt, json.dumps(q.marking_points),
                                                  q.suggested_answer, "", q.topic, q.difficulty))
            for k in a.cards:
                db.x("insert into flashcards(course_id,front,back,topic,difficulty) values(?,?,?,?,?)",
                     (cid, k.front, k.back, k.topic, k.difficulty))
            db.x("insert into quality_reports(course_id,report_json) values(?,?)", (cid, json.dumps(report)))
        db.x("update courses set status='ready' where id=?", (cid,))
        _job(cid, "ready", 1.0, "Done")
        cb(1.0, "Done")
    except Exception as e:
        db.x("update courses set status='failed' where id=?", (cid,))
        _job(cid, "failed", 0, "", f"{type(e).__name__}: {e}")
        raise

def generate_course(cid, cb=lambda p, m: None):
    with _generation_state_lock:
        if cid in _active_generations:
            raise RuntimeError(f"Course {cid} is already generating in this app process.")
        _active_generations.add(cid)
    try:
        return _generate_course(cid, cb)
    finally:
        with _generation_state_lock:
            _active_generations.discard(cid)

def delete_course(cid):
    db.x("delete from courses where id=?", (cid,))
    if db.q("select id from courses where id=?", (cid,)):
        raise RuntimeError(f"Course {cid} could not be deleted from the local database.")
    try:
        vector_store.delete(cid)
    except Exception:
        logging.exception("Failed to delete vector index for course %s", cid)
