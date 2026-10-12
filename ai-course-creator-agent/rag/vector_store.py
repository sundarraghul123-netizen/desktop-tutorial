import chromadb
from config import settings

def _coll():
    return chromadb.PersistentClient(path=str(settings.CHROMA_PATH)).get_or_create_collection(
        "chunks", metadata={"hnsw:space": "cosine", "embedding_model": settings.EMBED_MODEL})

def index(course_id, chunks, embeddings):
    _coll().add(ids=[f"{course_id}-{i}" for i in range(len(chunks))], embeddings=embeddings,
                documents=[c["text"] for c in chunks],
                metadatas=[{"course_id": course_id, "start": float(c["start"]), "end": float(c["end"])} for c in chunks])

def search(course_id, emb, k):
    r = _coll().query(query_embeddings=[emb], n_results=k, where={"course_id": course_id})
    return list(zip(r["documents"][0], r["metadatas"][0], r["distances"][0]))

def delete(course_id):
    _coll().delete(where={"course_id": course_id})
