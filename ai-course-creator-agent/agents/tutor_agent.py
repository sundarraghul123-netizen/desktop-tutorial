import json
import logging
import math
import re
from models import database as db
from rag import retriever
from rag.chunking import chunk, fmt
from services import llm_client

_STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "do", "for", "from", "main",
    "how", "i", "in", "is", "it", "me", "of", "on", "or", "that", "the", "this",
    "to", "use", "used", "what", "when", "where", "which", "who", "why", "with",
}

def _tokens(text):
    tokens = re.findall(r"\b[\w'-]+\b", text.casefold())
    return [t[:-1] if len(t) > 4 and t.endswith("s") else t for t in tokens]


def _keyword_retrieve(course_id, question, k=None):
    """Rank transcript chunks by keyword relevance when the vector index is unavailable."""
    terms = [t for t in _tokens(question)
             if t not in _STOP_WORDS and (len(t) > 2 or t == "ai")]
    if not terms:
        return []

    segments = db.q(
        "select start_s as start,end_s as end,text from segments where course_id=? order by id",
        (course_id,),
    )
    documents = chunk(segments)
    if not documents:
        return []

    tokenized = [_tokens(d["text"]) for d in documents]
    doc_freq = {}
    for tokens in tokenized:
        for term in set(tokens):
            doc_freq[term] = doc_freq.get(term, 0) + 1
    avg_len = sum(map(len, tokenized)) / len(tokenized) or 1
    scores = []
    for i, tokens in enumerate(tokenized):
        frequencies = {}
        for term in tokens:
            frequencies[term] = frequencies.get(term, 0) + 1
        length = len(tokens)
        score = 0.0
        for term in set(terms):
            tf = frequencies.get(term, 0)
            if tf:
                idf = math.log(1 + (len(documents) - doc_freq[term] + 0.5) /
                               (doc_freq[term] + 0.5))
                score += idf * (tf * 2.2) / (tf + 1.2 * (0.25 + 0.75 * length / avg_len))
        if score:
            scores.append((score, documents[i]))
    scores.sort(key=lambda item: item[0], reverse=True)
    limit = k or 5
    return [
        {"text": d["text"], "ref": f"{fmt(d['start'])}-{fmt(d['end'])}", "score": round(score, 2)}
        for score, d in scores[:limit]
    ]


def _save_assistant(course_id, text, citations):
    db.x("insert into tutor_messages(course_id,role,content,citations) values(?,?,?,?)",
         (course_id, "assistant", text, json.dumps(citations)))


def answer(course_id, question):
    hist = db.q("select role,content from tutor_messages where course_id=? order by id desc limit 6", (course_id,))[::-1]
    db.x("insert into tutor_messages(course_id,role,content) values(?,?,?)", (course_id, "user", question))
    fallback_notice = ""
    retrieval_error = None
    hits = []
    try:
        hits = retriever.retrieve(course_id, question)
    except Exception as e:
        logging.warning("Tutor vector retrieval failed for course %s: %s", course_id, e)
        fallback_notice = "I searched the transcript directly because the tutor index is unavailable.\n\n"
    if not hits:
        try:
            hits = _keyword_retrieve(course_id, question)
        except Exception as error:
            logging.exception("Tutor transcript fallback failed for course %s", course_id)
            retrieval_error = error
            fallback_notice += (
                "Course search is temporarily unavailable, so the answer below uses general knowledge only. "
                "It is not verified against this course.\n\n"
            )
    if not hits and "general knowledge only" not in fallback_notice:
        fallback_notice += (
            "No matching course excerpts were found; the answer uses general knowledge only and is not "
            "verified against this course.\n\n"
        )
    ctx = "\n\n".join(f"[{h['ref']}] {h['text']}" for h in hits)
    convo = "\n".join(f"{m['role']}: {m['content']}" for m in hist)
    source_instruction = (
        "Ground course-specific claims in the supplied excerpts. If they do not fully answer the question, "
        "say what is missing, then give reliable general teaching context labeled '(Added context)'."
        if hits else
        "No relevant course excerpts were found. Answer the learner's question using reliable general "
        "knowledge when possible, clearly label the answer '(Added context — not covered in the retrieved "
        "course material)', and do not claim that it came from this course. If the question is ambiguous or "
        "you cannot answer reliably, say what needs clarification rather than guessing."
    )
    p = (f"Conversation so far:\n{convo}\n\nQuestion: {question}\n"
         "Act as a patient teaching assistant. "
         f"{source_instruction} Use a clear, concise structure suited to the question: explain the idea in "
         "plain language, define necessary terms, explain why it is useful, and add a relevant practical "
         "example when useful. For procedural questions, show the steps and reasoning. When appropriate, ask "
         "one short retrieval or practice question and let the learner attempt it before revealing its "
         "solution. Correct mistakes constructively and adapt explanation complexity to the conversation. "
         "Do not invent citations; source references are attached separately by the application.\n"
         f"<source>\n{ctx}\n</source>")
    try:
        text = llm_client.generate(p)
    except Exception as e:
        logging.exception("Tutor response generation failed for course %s", course_id)
        detail = f"Course search also failed ({type(retrieval_error).__name__}). " if retrieval_error else ""
        text = (f"{detail}Tutor answer generation failed ({type(e).__name__}: {e}). "
                "Your question is saved; please retry in a moment.")
        _save_assistant(course_id, fallback_notice + text, [h["ref"] for h in hits])
        return fallback_notice + text, [h["ref"] for h in hits]
    cites = [h["ref"] for h in hits]  # citations come from retrieved metadata
    text = fallback_notice + text
    _save_assistant(course_id, text, cites)
    return text, cites
