import json
import pandas as pd
import plotly.express as px
import streamlit as st
from models import database as db
from agents import orchestrator as orc, tutor_agent, learning_coach_agent as coach
from ingestion import youtube_loader, upload_loader
from services import quiz_service as quiz, flashcard_service as fc, analytics_service as an, export_service as ex
from services.lesson_content import display_items, display_text

st.set_page_config("AI Course Creator", layout="wide")
db.init()
if "pending" in st.session_state:
    st.session_state.sel = st.session_state.pop("pending")

st.sidebar.title("AI Course Creator")
courses = db.q("select id,title,status from courses order by id desc")
cid = st.sidebar.selectbox("Course", [c["id"] for c in courses], key="sel", index=None if not courses else 0,
                           format_func=lambda i: next(f"{c['title']} ({c['status']})" for c in courses if c["id"] == i))
if cid is not None and st.sidebar.button("Delete this course"):
    orc.delete_course(cid)
    st.session_state.pending = None
    st.rerun()
st.sidebar.caption("Course data is stored locally. Prompts and transcript excerpts are sent to Google's hosted "
                   "Gemini service for generation and embeddings.")

page = st.radio(
    "Section",
    ["New", "Outline", "Lessons", "Quiz", "Flashcards", "Tutor", "Progress", "Export"],
    horizontal=True,
    label_visibility="collapsed",
)

if page == "New":
    url = st.text_input("YouTube URL")
    up = st.file_uploader("...or transcript file (.txt/.vtt/.srt)", type=["txt", "vtt", "srt"])
    title = st.text_input("Title (for uploaded transcripts)")
    c1, c2, c3 = st.columns(3)
    level = c1.selectbox("Level", ["beginner", "intermediate", "advanced"])
    style = c2.selectbox("Style", ["practical", "theory-focused", "interview-prep", "exam-oriented"])
    depth = c3.selectbox("Lesson length", ["short", "detailed"])
    n_mcq = c1.number_input("MCQs", 5, 50, 15); n_qa = c2.number_input("Q&A", 3, 30, 6); n_cards = c3.number_input("Flashcards", 5, 50, 15)
    if st.button("Fetch transcript and create outline", type="primary"):
        try:
            with st.spinner("Working..."):
                if up:
                    segs, lang, stype, src = upload_loader.parse_transcript(up.read()), "", "transcript", ""
                else:
                    segs, lang = youtube_loader.fetch(url); stype, src = "youtube", url
                    if lang and not lang.lower().startswith("en"):
                        st.info(f"Using available captions in language: {lang}")
                new = orc.create_course(stype, src, title or up.name if up else (title or url), segs, lang)
                orc.generate_outline(new, {"level": level, "style": style, "depth": depth,
                                           "n_mcq": n_mcq, "n_qa": n_qa, "n_cards": n_cards})
            st.session_state.pending = new; st.rerun()
        except Exception as e:
            st.error(str(e))

course = db.q("select * from courses where id=?", (cid,))[0] if cid else None
if not course:
    if page != "New":
        st.info("Create or select a course first.")
    st.stop()

if page == "Outline":
    if course["outline_json"]:
        o = json.loads(course["outline_json"])
        st.header(o["title"]); st.write(o["description"])
        st.caption(f"{o['difficulty']} | ~{o['estimated_minutes']} min | Audience: {o['audience']}")
        st.write("**Objectives:** " + "; ".join(o["objectives"]))
        for m in o["modules"]:
            with st.expander(m["title"]):
                st.write(m["description"])
                if m.get("purpose"): st.write(f"**Purpose:** {m['purpose']}")
                if m.get("estimated_minutes"): st.caption(f"Estimated duration: {m['estimated_minutes']} minutes")
                for label, key in (("Module outcomes", "objectives"), ("Prerequisites", "prerequisites"),
                                   ("Practice activities", "practice_tasks"),
                                   ("Completion criteria", "completion_criteria")):
                    if m.get(key):
                        st.markdown(f"**{label}**")
                        for item in m[key]: st.write(f"- {item}")
                if m.get("assessment"): st.write(f"**Assessment:** {m['assessment']}")
                st.markdown("**Lessons**")
                for l in m["lessons"]:
                    duration = f" (~{l['estimated_minutes']} min)" if l.get("estimated_minutes") else ""
                    st.write(f"- **{l['title']}**{duration}")
                    for objective in l.get("objectives", []): st.write(f"  - {objective}")
        job = db.q(
            "select status,progress,message,error from jobs where course_id=?",
            (cid,),
        )
        if job and job[0]["status"] == "generating":
            if orc.is_generation_active(cid):
                current_job = job[0]
                st.progress(
                    current_job["progress"] or 0.0,
                    current_job["message"] or "Course generation is in progress.",
                )
                st.info("Generation is already running. Wait for it to finish; do not start it again.")
            else:
                orc.recover_interrupted_generation(cid)
                course = db.q("select * from courses where id=?", (cid,))[0]
                job = db.q(
                    "select status,progress,message,error from jobs where course_id=?",
                    (cid,),
                )
                st.warning(
                    "The previous app run stopped before generation finished. "
                    "Your saved lessons are safe; retry below to continue."
                )
        elif job and job[0]["status"] == "failed" and job[0]["error"]:
            st.error(f"Last generation attempt failed: {job[0]['error']}")

        if course["status"] in ("awaiting_approval", "failed"):
            if st.button("Approve and generate course", type="primary"):
                bar = st.progress(0.0, "Starting")
                try:
                    orc.generate_course(cid, lambda p, m: bar.progress(min(p, 1.0), m)); st.rerun()
                except Exception as e:
                    st.error(f"{e}\n\nSaved progress is kept. Click again to resume.")

if page == "Lessons":
    lesson_counts = db.q(
        "select count(*) as total, coalesce(sum(completed), 0) as completed "
        "from lessons where course_id=?",
        (cid,),
    )[0]
    total_lessons = lesson_counts["total"]
    page_size = 25
    page_count = max(1, (total_lessons + page_size - 1) // page_size)
    page_key = f"lesson-page-{cid}"
    if st.session_state.get(page_key, 1) > page_count:
        st.session_state[page_key] = page_count

    _, progress_column = st.columns([4, 1])
    with progress_column:
        with st.container(border=True):
            st.metric(
                "Lessons",
                f"{lesson_counts['completed']}/{total_lessons}",
            )

    current_page = st.selectbox(
        "Lesson page",
        range(1, page_count + 1),
        format_func=lambda number: f"Page {number} of {page_count}",
        key=page_key,
    )
    first_lesson = (current_page - 1) * page_size + 1
    last_lesson = min(current_page * page_size, total_lessons)
    if total_lessons:
        st.caption(f"Showing lessons {first_lesson}-{last_lesson} of {total_lessons}")
    else:
        st.info("No lessons have been generated for this course yet.")

    lessons = db.q(
        "select * from lessons where course_id=? order by position limit ? offset ?",
        (cid, page_size, (current_page - 1) * page_size),
    )
    for l in lessons:
        d = json.loads(l["content_json"])
        with st.expander(f"{'✅ ' if l['completed'] else ''}Lesson {l['position']}: {l['title']}"):
            st.caption(l["module_title"])
            why_it_matters = display_text(d.get("why_it_matters"))
            if why_it_matters:
                st.markdown("**Why it matters**"); st.write(why_it_matters)
            learning_objectives = display_items(d.get("learning_objectives"))
            if learning_objectives:
                st.markdown("**Learning objectives**")
                for item in learning_objectives: st.markdown(f"- {item}")
            prerequisites = display_items(d.get("prerequisites"))
            if prerequisites:
                st.markdown("**Prerequisites**")
                for item in prerequisites: st.markdown(f"- {item}")
            introduction = display_text(d.get("introduction"))
            explanation = display_text(d.get("explanation"))
            if introduction: st.write(introduction)
            if explanation: st.write(explanation)
            analogy = display_text(d.get("analogy"))
            if analogy:
                st.markdown("**Analogy**"); st.write(analogy)
            step_by_step = display_items(d.get("step_by_step_breakdown"))
            if step_by_step:
                st.markdown("**Step-by-step**")
                for item in step_by_step: st.markdown(f"- {item}")
            visual = display_text(d.get("visual_representation"))
            if visual:
                st.markdown("**Visual guide**"); st.code(visual, language=None)
            for k, lab in (("concepts", "Key concepts"), ("examples", "Examples"), ("applications", "Applications"),
                           ("terminology", "Terminology"), ("common_mistakes", "Common mistakes"),
                           ("guided_practice", "Guided practice — I do / We do"),
                           ("independent_practice", "Independent practice — You do"),
                           ("retrieval_questions", "Retrieval practice"),
                           ("knowledge_checks", "Knowledge check"), ("mastery_check", "Mastery check")):
                items = display_items(d.get(k))
                if not items: continue
                st.markdown(f"**{lab}**")
                for item in items: st.markdown(f"- {item}")
            summary = display_text(d.get("summary"))
            if summary: st.info(summary)
            next_lesson = display_text(d.get("next_lesson_connection"))
            if next_lesson:
                st.caption(f"Next: {next_lesson}")
            refs = json.loads(l["sources"] or "[]")
            if refs: st.caption("Source timestamps: " + ", ".join(refs))
            if not l["completed"] and st.button("Mark complete", key=f"c{l['id']}"):
                db.x("update lessons set completed=1 where id=?", (l["id"],)); st.rerun()

if page == "Quiz":
    adaptive = st.toggle("Adaptive selection", True)
    st.caption(f"Current target difficulty: {quiz.target_level(cid)}" if adaptive else "Fixed random selection")
    if st.button("Start quiz (10 questions)"):
        st.session_state.active_quiz = [q["id"] for q in quiz.select(cid, 10, adaptive)]
    ids = st.session_state.get("active_quiz", [])
    if ids:
        with st.form("quiz_form"):
            answers = {}
            for i in ids:
                q = db.q("select * from questions where id=?", (i,))[0]
                answers[i] = st.radio(q["prompt"], json.loads(q["options_json"]), index=None, key=f"q{i}")
            sub = st.form_submit_button("Submit")
        if sub:
            right = 0
            for i, a in answers.items():
                q = db.q("select * from questions where id=?", (i,))[0]
                ok = a == q["answer"]; right += ok
                quiz.record(cid, i, a or "", ok)
                (st.success if ok else st.error)(f"{q['prompt']}\n\nCorrect: {q['answer']}. {q['explanation']}")
            st.metric("Score", f"{right}/{len(ids)}")
    with st.expander("Short and long answer bank"):
        for q in db.q("select * from questions where course_id=? and qtype!='mcq'", (cid,)):
            st.markdown(f"**{q['prompt']}** _({q['qtype']}, {q['difficulty']})_"); st.write(q["answer"])
            st.caption("Marking points: " + "; ".join(json.loads(q["options_json"])))

if page == "Flashcards":
    cards = fc.due(cid)
    st.write(f"{len(cards)} card(s) due")
    if cards:
        k = cards[0]; st.subheader(k["front"])
        if st.toggle("Show answer", key=f"s{k['id']}"):
            st.write(k["back"])
            cols = st.columns(4)
            for col, r in zip(cols, fc.RATING):
                if col.button(r, key=f"{r}{k['id']}"):
                    fc.review(k["id"], r); st.rerun()

if page == "Tutor":
    st.caption("Answers come from retrieved course material and may be imperfect.")
    for m in db.q("select * from tutor_messages where course_id=? order by id", (cid,)):
        with st.chat_message(m["role"]):
            st.write(m["content"])
            if m["citations"] and m["citations"] != "[]": st.caption("Sources: " + ", ".join(json.loads(m["citations"])))
    if qn := st.chat_input("Ask about this course"):
        try:
            tutor_agent.answer(cid, qn); st.rerun()
        except Exception as e:
            st.error(str(e))

if page == "Progress":
    s = an.summary(cid)
    a, b, c = st.columns(3)
    a.metric("Lessons done", f"{s['completed']}/{s['lessons']}"); b.metric("Completion", f"{s['completion_pct']}%")
    c.metric("Quiz accuracy", "-" if s["accuracy"] is None else f"{s['accuracy']}%")
    m = quiz.mastery(cid)
    if m:
        df = pd.DataFrame([{"topic": t, "mastery": v["mastery"], "attempts": v["attempts"]} for t, v in m.items()])
        st.plotly_chart(px.bar(df.dropna(), x="topic", y="mastery", range_y=[0, 100]), use_container_width=True)
        st.caption("Weighted topic score for practice guidance only; it is not a measure of ability.")
    if not s["trend"].empty:
        st.plotly_chart(px.line(s["trend"], x="day", y="score", markers=True), use_container_width=True)
    for t in coach.recommend(cid): st.write("- " + t)

if page == "Export":
    for kind, lab in (("notes", "Course notes"), ("questions", "Question paper"),
                      ("answers", "Answer key"), ("flashcards", "Flashcards")):
        if st.button(f"Build {lab} PDF", key=f"b{kind}"):
            try:
                st.download_button(f"Download {lab}", ex.build(cid, kind), f"{kind}.pdf", "application/pdf", key=f"d{kind}")
            except Exception as e:
                st.error(f"Export failed (course is safe, retry): {e}")
