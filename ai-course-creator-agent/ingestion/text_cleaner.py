import re

def clean(segs):
    out = []
    for s in segs:
        t = re.sub(r"\s+", " ", s["text"]).strip()
        if t and (not out or out[-1]["text"] != t):
            out.append({"start": s["start"], "end": s["end"], "text": t})
    if sum(len(s["text"]) for s in out) < 50:
        raise ValueError("Transcript is empty or too short to build a course.")
    return out
