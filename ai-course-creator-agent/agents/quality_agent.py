import re

def _norm(s):
    return re.sub(r"\W+", " ", s.lower()).strip()

def check(assessment):
    """Return (clean_assessment, report). Drops invalid or duplicate items."""
    report = {"dropped_mcq": [], "dropped_qa": 0, "dropped_cards": 0}
    seen, good = set(), []
    for m in assessment.mcqs:
        key = _norm(m.prompt)
        opts = [_norm(o) for o in m.options]
        why = ("duplicate" if key in seen else "needs 4 options" if len(m.options) != 4
               else "duplicate options" if len(set(opts)) != 4 else "bad answer index"
               if not 0 <= m.answer_index < 4 else "no explanation" if not m.explanation.strip() else None)
        if why:
            report["dropped_mcq"].append(why)
        else:
            seen.add(key); good.append(m)
    qa, seenq = [], set()
    for q in assessment.qas:
        if _norm(q.prompt) in seenq or not q.suggested_answer.strip():
            report["dropped_qa"] += 1
        else:
            seenq.add(_norm(q.prompt)); qa.append(q)
    cards, seenc = [], set()
    for c in assessment.cards:
        if _norm(c.front) in seenc or not c.back.strip():
            report["dropped_cards"] += 1
        else:
            seenc.add(_norm(c.front)); cards.append(c)
    assessment.mcqs, assessment.qas, assessment.cards = good, qa, cards
    report["kept"] = {"mcq": len(good), "qa": len(qa), "cards": len(cards)}
    return assessment, report
