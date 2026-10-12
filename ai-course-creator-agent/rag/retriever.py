from config import settings
from rag import vector_store
from rag.chunking import fmt
from services import llm_client

def retrieve(course_id, question, k=None):
    return retrieve_many(course_id, [question], k)[0]

def retrieve_many(course_id, questions, k=None):
    if not questions:
        return []
    embeddings = llm_client.embed(questions)
    results = []
    for emb in embeddings:
        hits = []
        for doc, meta, dist in vector_store.search(course_id, emb, k or settings.TOP_K):
            if 1 - dist >= settings.MIN_RELEVANCE:
                hits.append({
                    "text": doc,
                    "ref": f"{fmt(meta['start'])}-{fmt(meta['end'])}",
                    "score": round(1 - dist, 2),
                })
        results.append(hits)
    return results
