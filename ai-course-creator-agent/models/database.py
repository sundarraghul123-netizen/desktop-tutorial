import sqlite3
from contextlib import contextmanager
from config.settings import DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS courses(id INTEGER PRIMARY KEY, title TEXT, description TEXT, source_type TEXT,
 source_url TEXT, source_title TEXT, difficulty TEXT, status TEXT, settings_json TEXT, outline_json TEXT,
 created_at TEXT DEFAULT CURRENT_TIMESTAMP);
CREATE TABLE IF NOT EXISTS jobs(course_id INTEGER PRIMARY KEY REFERENCES courses(id) ON DELETE CASCADE,
 status TEXT, progress REAL, message TEXT, error TEXT, updated_at TEXT DEFAULT CURRENT_TIMESTAMP);
CREATE TABLE IF NOT EXISTS segments(id INTEGER PRIMARY KEY, course_id INTEGER REFERENCES courses(id) ON DELETE CASCADE,
 start_s REAL, end_s REAL, text TEXT, language TEXT);
CREATE TABLE IF NOT EXISTS lessons(id INTEGER PRIMARY KEY, course_id INTEGER REFERENCES courses(id) ON DELETE CASCADE,
 module_title TEXT, position INTEGER, title TEXT, content_json TEXT, summary TEXT, sources TEXT, completed INTEGER DEFAULT 0);
CREATE TABLE IF NOT EXISTS questions(id INTEGER PRIMARY KEY, course_id INTEGER REFERENCES courses(id) ON DELETE CASCADE,
 qtype TEXT, prompt TEXT, options_json TEXT, answer TEXT, explanation TEXT, topic TEXT, difficulty TEXT);
CREATE TABLE IF NOT EXISTS attempts(id INTEGER PRIMARY KEY, course_id INTEGER REFERENCES courses(id) ON DELETE CASCADE,
 question_id INTEGER REFERENCES questions(id) ON DELETE CASCADE, learner_answer TEXT, is_correct INTEGER,
 attempted_at TEXT DEFAULT CURRENT_TIMESTAMP);
CREATE TABLE IF NOT EXISTS flashcards(id INTEGER PRIMARY KEY, course_id INTEGER REFERENCES courses(id) ON DELETE CASCADE,
 front TEXT, back TEXT, topic TEXT, difficulty TEXT, ease REAL DEFAULT 2.5, interval INTEGER DEFAULT 0,
 reps INTEGER DEFAULT 0, due TEXT, last_reviewed TEXT);
CREATE TABLE IF NOT EXISTS card_reviews(id INTEGER PRIMARY KEY, card_id INTEGER REFERENCES flashcards(id) ON DELETE CASCADE,
 rating TEXT, reviewed_at TEXT DEFAULT CURRENT_TIMESTAMP);
CREATE TABLE IF NOT EXISTS tutor_messages(id INTEGER PRIMARY KEY, course_id INTEGER REFERENCES courses(id) ON DELETE CASCADE,
 role TEXT, content TEXT, citations TEXT, created_at TEXT DEFAULT CURRENT_TIMESTAMP);
CREATE TABLE IF NOT EXISTS quality_reports(id INTEGER PRIMARY KEY, course_id INTEGER REFERENCES courses(id) ON DELETE CASCADE,
 report_json TEXT, created_at TEXT DEFAULT CURRENT_TIMESTAMP);
CREATE INDEX IF NOT EXISTS ix_seg ON segments(course_id);
CREATE INDEX IF NOT EXISTS ix_att ON attempts(course_id);
CREATE INDEX IF NOT EXISTS ix_q ON questions(course_id);
"""

@contextmanager
def conn():
    c = sqlite3.connect(DB_PATH)
    c.row_factory = sqlite3.Row
    c.execute("PRAGMA foreign_keys=ON")
    try:
        yield c
        c.commit()
    finally:
        c.close()

def init():
    with conn() as c:
        c.executescript(SCHEMA)

def q(sql, args=()):
    with conn() as c:
        return [dict(r) for r in c.execute(sql, args)]

def x(sql, args=()):
    with conn() as c:
        return c.execute(sql, args).lastrowid

def many(sql, rows):
    with conn() as c:
        c.executemany(sql, rows)
