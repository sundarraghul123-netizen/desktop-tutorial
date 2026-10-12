import re

TS = re.compile(r"^\[?(?:(\d+):)?(\d{1,2}):(\d{2})\]?\s+(.*)$")

def parse_transcript(raw: bytes):
    """Accepts plain text or lines like '[01:23] text'. Timestamps are optional."""
    lines = raw.decode("utf-8", errors="ignore").splitlines()
    segs, t = [], 0.0
    for ln in lines:
        ln = ln.strip()
        if not ln or ln.isdigit() or "-->" in ln or ln == "WEBVTT":
            continue
        m = TS.match(ln)
        if m:
            h, mi, s, ln = int(m.group(1) or 0), int(m.group(2)), int(m.group(3)), m.group(4)
            t = h * 3600 + mi * 60 + s
        segs.append({"start": t, "end": t, "text": ln})
    return segs
