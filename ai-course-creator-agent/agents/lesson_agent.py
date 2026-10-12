from models.schemas import Lesson, LessonBatch
from services import llm_client

def _instructions(opts):
    return (
        f"Level: {opts['level']}. Style: {opts['style']}. Depth: {opts['depth']}.\n"
        "Teach for understanding and application, not as a summary. Fill the structured lesson fields with "
        "clear, learner-appropriate content. In the explanation, teach the topic in a logical sequence: "
        "define the central idea, explain why it exists and when it is useful, show how it works step by step, "
        "clarify important details and likely points of confusion, and connect it to the lesson objectives and "
        "nearby concepts. State the meaning of the topic and why learners use it. Explain cause and effect "
        "rather than listing facts. Include at least one realistic, real-world use case in the applications "
        "field and a concrete worked example in the examples field; when the source is introductory, use a "
        "believable everyday or workplace scenario. Use concrete examples to make abstract ideas understandable, "
        "and distinguish similar concepts when that helps. For detailed depth, provide a thorough walkthrough "
        "with enough context that a beginner can follow without outside material; for short depth, stay compact "
        "but still explain the complete core idea. Do not pad or repeat content already covered in other fields. "
        "Explain why the topic matters; list 3-5 measurable objectives and prerequisites; define important "
        "terminology; include an analogy only if useful and state its limitation; break difficult procedures "
        "into ordered steps; provide a useful text diagram or other structural representation when it helps; "
        "demonstrate a worked example; identify likely mistakes and corrections; include guided practice "
        "(I do/We do) and a new independent practice task (You do) without its solution; ask 3 "
        "retrieval/knowledge-check questions; give a short summary, a mastery check, and a connection to the "
        "next lesson. Keep it concise for the requested depth and avoid forcing irrelevant sections. Ground "
        "claims in the excerpts. Clearly mark supplementary explanations with '(Added context)'; never invent "
        "source references. Include any available source timestamps in the source references supplied by the "
        "application."
    )

def generate(course_title, lesson_outline, context, opts):
    p = (f"Course: {course_title}\nWrite the lesson '{lesson_outline.title}'.\n"
         f"Objectives: {lesson_outline.objectives}\nTopics: {lesson_outline.key_topics}\n"
         f"{_instructions(opts)}\n"
         f"<source>\n{context}\n</source>")
    return llm_client.generate(p, schema=Lesson)

def generate_batch(course_title, lesson_outlines, contexts, opts):
    if len(lesson_outlines) != len(contexts):
        raise ValueError("Each lesson outline must have exactly one matching source context.")
    sections = []
    for outline, context in zip(lesson_outlines, contexts):
        sections.append(
            f"Lesson title: {outline.title}\n"
            f"Objectives: {outline.objectives}\n"
            f"Topics: {outline.key_topics}\n"
            f"<source>\n{context}\n</source>"
        )
    prompt = (
        f"Course: {course_title}\n"
        f"Create one complete, separate lesson for each of the {len(lesson_outlines)} lesson requests below. "
        "Return them in the same order. For every result, copy its exact lesson title into outline_title and "
        "write that lesson's content in lesson. Do not combine topics or omit any requested lesson.\n"
        f"{_instructions(opts)}\n"
        + "\n\n".join(sections)
    )
    result = llm_client.generate(prompt, schema=LessonBatch)
    expected = [outline.title.strip().casefold() for outline in lesson_outlines]
    actual = [item.outline_title.strip().casefold() for item in result.lessons]
    if len(actual) != len(expected) or set(actual) != set(expected):
        raise ValueError(
            "The model did not return exactly one result for each requested lesson; "
            "the batch was not saved."
        )
    by_title = {item.outline_title.strip().casefold(): item.lesson for item in result.lessons}
    lessons = []
    for outline, title in zip(lesson_outlines, expected):
        lesson = by_title[title]
        lesson.title = outline.title
        lessons.append(lesson)
    return lessons
