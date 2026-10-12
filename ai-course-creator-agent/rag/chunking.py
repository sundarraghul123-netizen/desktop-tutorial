from config import settings

def _mk(cur):
    return {"text": " ".join(s["text"] for s in cur), "start": cur[0]["start"], "end": cur[-1]["end"]}

def chunk(segs, size=None, overlap=None):
    size, overlap = size or settings.CHUNK_SIZE, settings.CHUNK_OVERLAP if overlap is None else overlap
    chunks, cur, n, fresh = [], [], 0, 0
    for s in segs:
        cur.append(s); n += len(s["text"]) + 1; fresh += 1
        if n >= size:
            chunks.append(_mk(cur)); fresh = 0
            keep, k = [], 0
            for t in reversed(cur):
                if k >= overlap:
                    break
                keep.insert(0, t); k += len(t["text"])
            if len(keep) == len(cur):
                keep = keep[1:]
            cur, n = keep, sum(len(t["text"]) + 1 for t in keep)
    if fresh:
        chunks.append(_mk(cur))
    return chunks

def sample(chunks, limit=None):
    """Evenly sample chunks so long transcripts fit the prompt budget."""
    limit = limit or settings.MAX_PROMPT_CHARS
    if not chunks:
        return ""
    per = max(200, limit // len(chunks))
    return "\n".join(f"[{fmt(c['start'])}] {c['text'][:per]}" for c in chunks)[:limit]

def fmt(sec):
    sec = int(sec or 0)
    return f"{sec // 60}:{sec % 60:02d}"
