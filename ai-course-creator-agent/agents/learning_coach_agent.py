from models import database as db
from services import quiz_service

def recommend(course_id):
    m, tips = quiz_service.mastery(course_id), []
    for t, v in m.items():
        if v["mastery"] is None:
            tips.append(f"'{t}': only {v['attempts']} attempt(s); need {quiz_service.MIN_ATTEMPTS} before mastery is shown.")
        elif v["mastery"] < quiz_service.WEAK_BELOW:
            tips.append(f"Revise '{t}': weighted score {v['mastery']}% over {v['attempts']} attempts.")
    nxt = db.q("select title from lessons where course_id=? and completed=0 order by position limit 1", (course_id,))
    if nxt:
        tips.append(f"Next lesson: {nxt[0]['title']}")
    return tips or ["Take a quiz to get personalised recommendations."]
