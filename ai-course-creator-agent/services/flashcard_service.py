import datetime as dt
from models import database as db

RATING = {"Again": 1, "Hard": 3, "Good": 4, "Easy": 5}

def sm2(ease, interval, reps, q):
    if q < 3:
        reps, interval = 0, 1
    else:
        reps += 1
        interval = 1 if reps == 1 else 6 if reps == 2 else round(interval * ease)
    ease = max(1.3, ease + 0.1 - (5 - q) * (0.08 + (5 - q) * 0.02))
    return ease, interval, reps

def due(course_id):
    return db.q("select * from flashcards where course_id=? and (due is null or due<=date('now')) "
                "order by due", (course_id,))

def review(card_id, rating):
    c = db.q("select * from flashcards where id=?", (card_id,))[0]
    ease, interval, reps = sm2(c["ease"], c["interval"], c["reps"], RATING[rating])
    nxt = (dt.date.today() + dt.timedelta(days=interval)).isoformat()
    db.x("update flashcards set ease=?,interval=?,reps=?,due=?,last_reviewed=date('now') where id=?",
         (ease, interval, reps, nxt, card_id))
    db.x("insert into card_reviews(card_id,rating) values(?,?)", (card_id, rating))
