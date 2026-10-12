from concurrent.futures import ThreadPoolExecutor

from config import settings
from models.schemas import CoursePlan, LessonOutline, ModuleOutline, Outline, SourceTopic, SourceTopicCatalog
from services import llm_client

TOPIC_BATCH_CHARS = 60000
TOPIC_EXTRACTION_WORKERS = settings.MODEL_CONCURRENCY

def _topic_batches(chunks, limit=TOPIC_BATCH_CHARS):
    batches, current, size = [], [], 0
    for chunk in chunks:
        start = int(chunk["start"] or 0)
        timestamp = f"{start // 3600:02d}:{start % 3600 // 60:02d}:{start % 60:02d}"
        text = chunk["text"]
        while text:
            prefix = f"[{timestamp}] "
            available = limit - size - len(prefix) - 1
            if available <= 0:
                batches.append("\n".join(current))
                current, size = [], 0
                continue
            piece, text = text[:available], text[available:]
            current.append(prefix + piece)
            size += len(prefix) + len(piece) + 1
            if text:
                batches.append("\n".join(current))
                current, size = [], 0
    if current:
        batches.append("\n".join(current))
    return batches

def _one_lesson_per_topic(outline, topics):
    if not topics:
        return outline

    modules = list(outline.modules)
    if not modules:
        modules = [ModuleOutline(title="Course Content", description="Lessons covering the video topics.", lessons=[])]

    assignments = {}
    for topic_index, topic in enumerate(topics):
        topic_title = topic.title.strip().casefold()
        for module_index, module in enumerate(modules):
            if any(
                key_topic.strip().casefold() == topic_title
                for lesson in module.lessons
                for key_topic in lesson.key_topics
            ) or any(lesson.title.strip().casefold() == topic_title for lesson in module.lessons):
                assignments[topic_index] = module_index
                break

    for topic_index in range(len(topics)):
        if topic_index in assignments:
            continue
        nearest = min(
            assignments,
            key=lambda assigned_index: abs(assigned_index - topic_index),
            default=None,
        )
        assignments[topic_index] = assignments[nearest] if nearest is not None else 0

    module_lessons = [[] for _ in modules]
    for topic_index, topic in enumerate(topics):
        module_index = assignments[topic_index]
        matching_lessons = [
            lesson
            for lesson in modules[module_index].lessons
            if topic.title.strip().casefold() == lesson.title.strip().casefold()
            or [key_topic.strip().casefold() for key_topic in lesson.key_topics]
               == [topic.title.strip().casefold()]
        ]
        source_lesson = matching_lessons[0] if matching_lessons else None
        objectives = source_lesson.objectives if source_lesson and source_lesson.objectives else [
            f"Explain the key ideas in {topic.title}.",
            f"Apply {topic.title} in an appropriate example."
        ]
        module_lessons[module_index].append(LessonOutline(
            title=topic.title,
            objectives=objectives,
            key_topics=[topic.title],
            estimated_minutes=source_lesson.estimated_minutes if source_lesson else 10,
        ))

    normalized_modules = [
        module.model_copy(update={"lessons": lessons})
        for module, lessons in zip(modules, module_lessons)
        if lessons
    ]
    return outline.model_copy(update={"modules": normalized_modules})

def _deduplicate_topics(topics):
    unique, seen_titles = [], set()
    for topic in topics:
        normalized_title = topic.title.strip().casefold()
        if normalized_title and normalized_title not in seen_titles:
            unique.append(topic)
            seen_titles.add(normalized_title)
    return unique

def _outline_prompt(opts, source_tag):
    return (f"Design a structured, source-grounded course outline from the {source_tag}.\n"
            f"Level: {opts['level']}. Style: {opts['style']}. Depth: {opts['depth']}.\n"
            "State the target audience, prerequisites, measurable course outcomes, and realistic total duration. "
            "Create exactly one separate lesson for each distinct teaching topic covered in the source. Never "
            "combine multiple distinct topics into one lesson, and do not omit topics to satisfy a fixed module "
            "or lesson count. Group related lessons into logically ordered modules, and progress from foundations "
            "through application and review where relevant; adapt to the source. For every module provide its "
            "purpose, measurable objectives, prerequisites, practice tasks, an assessment method, completion "
            "criteria, and estimated minutes. For each lesson give measurable objectives and estimated minutes. "
            "Do not invent source coverage. Keep lessons appropriately sized for the requested depth.")

def outline(chunks, opts):
    batches = _topic_batches(chunks)
    if not batches:
        raise ValueError("The transcript contains no content to outline.")

    if len(batches) == 1:
        source_text = batches[0]
        prompt = (f"{_outline_prompt(opts, 'the complete transcript')}\n"
                  "Return both an ordered catalog containing every distinct teaching topic and the course "
                  "outline. The outline must contain exactly one lesson per catalog topic; each lesson's "
                  "title and key_topics must identify only that topic. Include the supplied timestamp with "
                  "each catalog topic. Ignore non-instructional quiz breaks and promotional transitions.\n"
                  f"<source>\n{source_text}\n</source>")
        plan = llm_client.generate(prompt, schema=CoursePlan)
        return _one_lesson_per_topic(plan.outline, _deduplicate_topics(plan.topics))

    def extract_topics(item):
        index, source_text = item
        p = (f"Extract every distinct teaching topic explicitly covered in transcript part "
             f"{index} of {len(batches)}, preserving source order. For each topic provide a concise "
             "description and the supplied timestamp. Do not invent topics. Keep distinct topics "
             "separate, including separate named sections even when related; ignore quiz breaks and "
             "promotional transitions unless they teach content.\n"
             f"<source>\n{source_text}\n</source>")
        return llm_client.generate(p, schema=SourceTopicCatalog).topics

    with ThreadPoolExecutor(max_workers=min(TOPIC_EXTRACTION_WORKERS, len(batches))) as executor:
        extracted_topics = [
            topic
            for batch_topics in executor.map(extract_topics, enumerate(batches, start=1))
            for topic in batch_topics
        ]
    topics = _deduplicate_topics(extracted_topics)

    catalog_text = "\n".join(
        f"- [{topic.timestamp}] {topic.title}: {topic.description}" for topic in topics
    )
    p = (f"{_outline_prompt(opts, 'the source topic catalog below')}\n"
         "Create exactly one separate lesson for each catalog topic. Each lesson title and key_topics must "
         "identify only its single catalog topic.\n"
         f"<source_topic_catalog>\n{catalog_text}\n</source_topic_catalog>")
    generated_outline = llm_client.generate(p, schema=Outline)
    return _one_lesson_per_topic(generated_outline, topics)
