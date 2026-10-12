# AI Course Creator Agent (v1.1)

## Setup (Windows, PowerShell, Python 3.11/3.12)
```
python -m venv .venv
Set-ExecutionPolicy -Scope Process Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
copy .env.example .env      # then add GOOGLE_API_KEY and verify GEMINI_MODEL
streamlit run app.py
pytest -q
```
## Flow
Paste a YouTube URL (or upload a .txt/.vtt/.srt transcript) -> review outline -> approve -> lessons, MCQs,
Q&A, flashcards -> study, quiz (adaptive), flashcards (SM-2), tutor (RAG), progress, PDF export.
Re-clicking "Approve" after a failure resumes without redoing finished lessons.
YouTube import prefers English captions, translates another available caption track to English when possible,
and otherwise uses the available track's original language. If captions are disabled or absent, upload a
transcript file instead.

## Teaching approach
For transcripts that fit in one request, topic extraction and outline creation share a single model call;
longer transcripts are processed in parallel batches before outline creation. Course outlines create exactly
one separate lesson for every identified topic, grouping lessons into modules without a fixed lesson-count ceiling.
Outlines present measurable outcomes,
prerequisites, module practice, assessments, and completion criteria before generation. Lesson explanations
teach the central idea step by step, explain why and when it is useful, clarify confusing details, and use
examples to connect concepts. Lessons also cover objectives and
prerequisites, a plain-language explanation, useful examples, common mistakes, guided and independent
practice, retrieval questions, a mastery check, and a connection to the next lesson. The tutor adapts its
explanation to the learner's question and invites practice when useful. Source-based claims are grounded in
retrieved excerpts; supplementary explanations are labeled.
Lesson content is generated two lessons per model request to reduce request overhead, with up to six batches
in progress concurrently by default. Topic extraction also runs in parallel. Model requests are paced to stay
below the common 15-requests-per-minute free-tier limit; Gemini quota errors honor the server-provided retry
delay. Set `GEMINI_MAX_CONCURRENCY` or `GEMINI_MIN_REQUEST_INTERVAL` in `.env` to match your API quota.
Transcript indexing reports progress as it embeds each batch, and requests have a configurable timeout.
If Gemini reports a long quota reset, the app stops instead of waiting hours and shows the failure.
Completed lessons are saved individually so retrying after a failure still resumes from saved progress.

## Privacy
Data is stored locally (SQLite + ChromaDB). Generation and embeddings send prompts and transcript excerpts
to Google's Gemini API. Check Google's current terms for your API tier (free-tier content may be used to
improve products). Respect YouTube's terms and copyright.

## Notes
- Unicode PDFs: place `DejaVuSans.ttf` in `fonts/`. Complex scripts (Tamil, Hindi) need HTML-to-PDF instead.
- Changing GEMINI_EMBEDDING_MODEL requires re-indexing (re-approve the course).
- Tutor citations are built from retrieved chunk metadata, never model-written.
