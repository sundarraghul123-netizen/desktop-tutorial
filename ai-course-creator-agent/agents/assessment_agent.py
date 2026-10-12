import random
from models.schemas import Assessment
from services import llm_client

def generate(course_title, digest, n_mcq, n_qa, n_cards):
    p = (f"Course: {course_title}\nFrom the lesson digest create {n_mcq} MCQs (exactly 4 distinct options, one "
         f"unambiguous correct answer, explanation, topic, difficulty easy/medium/hard), {n_qa} short/long/scenario "
         f"questions with suggested answers and marking points, and {n_cards} flashcards. No duplicates. "
         "Assess a mix of recall, explanation, application, and analysis, aligned to the lesson topics. "
         "Make distractors plausible but clearly incorrect, and ensure the explanation supports the answer. "
         "Do not claim mastery from a single question or include unsupported facts.\n"
         f"<source>\n{digest[:12000]}\n</source>")
    a = llm_client.generate(p, schema=Assessment)
    for m in a.mcqs:  # shuffle but keep the mapping
        if 0 <= m.answer_index < len(m.options):
            correct = m.options[m.answer_index]
            random.shuffle(m.options)
            m.answer_index = m.options.index(correct)
    return a
