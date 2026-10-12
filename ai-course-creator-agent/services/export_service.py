import json
from pathlib import Path
from xml.sax.saxutils import escape
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from config import settings
from models import database as db

FONT = "Helvetica"
_f = Path(__file__).resolve().parent.parent / "fonts" / "DejaVuSans.ttf"
if _f.exists():  # drop DejaVuSans.ttf in fonts/ for Unicode text
    pdfmetrics.registerFont(TTFont("DejaVu", str(_f))); FONT = "DejaVu"

def _styles():
    s = getSampleStyleSheet()
    for k in ("Normal", "Heading1", "Heading2", "Heading3"):
        s[k].fontName = FONT
    s["Normal"].spaceAfter = 6
    return s

def _page(canvas, doc):
    canvas.setFont(FONT, 8); canvas.drawCentredString(A4[0] / 2, 20, f"Page {doc.page}")

def build(cid, kind):
    """kind: notes | questions | answers | flashcards"""
    c = db.q("select * from courses where id=?", (cid,))[0]
    s = _styles()
    P = lambda t, st="Normal": Paragraph(escape(str(t)), s[st])
    story = [P(c["title"], "Heading1"), P(c["description"] or ""), P(f"Source: {c['source_url'] or c['source_title']}")]
    if kind == "notes":
        for l in db.q("select * from lessons where course_id=? order by position", (cid,)):
            d = json.loads(l["content_json"])
            story += [PageBreak(), P(f"{l['module_title']} / {l['title']}", "Heading2"), P(d["introduction"]),
                      P(d["explanation"]), P("Key concepts", "Heading3")] + [P("- " + x) for x in d["concepts"]]
            story += [P("Examples", "Heading3")] + [P("- " + x) for x in d["examples"]]
            story += [P("Terminology", "Heading3")] + [P("- " + x) for x in d["terminology"]]
            story += [P("Summary", "Heading3"), P(d["summary"])]
            refs = json.loads(l["sources"] or "[]")
            if refs:
                story.append(P("Sources (video timestamps): " + ", ".join(refs)))
    elif kind in ("questions", "answers"):
        for i, q in enumerate(db.q("select * from questions where course_id=? order by id", (cid,)), 1):
            story.append(P(f"Q{i}. {q['prompt']}", "Heading3"))
            if q["qtype"] == "mcq":
                for j, o in enumerate(json.loads(q["options_json"])):
                    story.append(P(f"   {'ABCD'[j]}. {o}"))
            if kind == "answers":
                story += [P(f"Answer: {q['answer']}"), P(f"Explanation/marking: {q['explanation'] or q['options_json']}")]
    elif kind == "flashcards":
        for i, k in enumerate(db.q("select * from flashcards where course_id=?", (cid,)), 1):
            story += [P(f"{i}. {k['front']}", "Heading3"), P(k["back"])]
    path = settings.EXPORT_DIR / f"course_{cid}_{kind}.pdf"
    SimpleDocTemplate(str(path), pagesize=A4, title=c["title"]).build(story, onFirstPage=_page, onLaterPages=_page)
    return path.read_bytes()
