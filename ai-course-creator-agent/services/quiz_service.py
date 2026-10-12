import random
from models import database as db

WEIGHT = {"easy": 1.0, "medium": 1.5, "hard": 2.0}
MIN_ATTEMPTS = 3      # mastery hidden until this many attempts on a topic
WEAK_BELOW = 60.0

def record(course_id, qid, answer, correct):
    db.x("insert into attempts(course_id,question_id,learner_answer,is_correct) values(?,?,?,?)",
         (course_id, qid, answer, int(correct)))

def mastery_from_rows(rows):
    agg = {}
    for r in rows:
        w = WEIGHT.get(r["difficulty"], 1.0)
        a = agg.setdefault(r["topic"], [0.0, 0.0, 0])
        a[0] += w * r["is_correct"]; a[1] += w; a[2] += 1
    return {t: {"mastery": round(100 * c / tot, 1) if n >= MIN_ATTEMPTS else None, "attempts": n}
            for t, (c, tot, n) in agg.items()}

def mastery(course_id):
    return mastery_from_rows(db.q(
        "select q.topic,q.difficulty,a.is_correct from attempts a join questions q on q.id=a.question_id "
        "where a.course_id=?", (course_id,)))

def target_level(course_id):
    r = db.q("select is_correct from attempts where course_id=? order by id desc limit 5", (course_id,))
    if len(r) < 3:
        return "medium"
    acc = sum(x["is_correct"] for x in r) / len(r)
    return "hard" if acc >= 0.8 else "easy" if acc <= 0.4 else "medium"

def select(course_id, n=10, adaptive=True):
    qs = db.q("select * from questions where course_id=? and qtype='mcq'", (course_id,))
    if not adaptive or not qs:
        return random.sample(qs, min(n, len(qs)))
    m, level = mastery(course_id), target_level(course_id)
    def w(q):
        s = m.get(q["topic"], {}).get("mastery")
        return (2.0 if s is not None and s < WEAK_BELOW else 1.0) * (1.5 if q["difficulty"] == level else 1.0)
    keyed = sorted(qs, key=lambda q: random.random() ** (1 / w(q)), reverse=True)
    return keyed[:n]
