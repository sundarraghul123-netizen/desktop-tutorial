import pandas as pd
from models import database as db

def summary(course_id):
    les = db.q("select count(*) n, coalesce(sum(completed),0) d from lessons where course_id=?", (course_id,))[0]
    att = db.q("select date(attempted_at) day, avg(is_correct)*100 score, count(*) n from attempts "
               "where course_id=? group by day order by day", (course_id,))
    tot = db.q("select count(*) n, coalesce(sum(is_correct),0) c from attempts where course_id=?", (course_id,))[0]
    return {"lessons": les["n"], "completed": les["d"],
            "completion_pct": round(100 * les["d"] / les["n"], 1) if les["n"] else 0.0,
            "attempts": tot["n"], "accuracy": round(100 * tot["c"] / tot["n"], 1) if tot["n"] else None,
            "trend": pd.DataFrame(att)}
