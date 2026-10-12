# AI Course Creator: A Technical Guide in Plain Language

This guide explains what the AI Course Creator does, how its parts work together,
how to run it, and what to expect when something goes wrong. It is written for
readers who may not have a software engineering background.

## 1. What the program does

The AI Course Creator turns a video transcript or transcript file into study
materials. You can:

1. Provide a YouTube video URL or upload a `.txt`, `.vtt`, or `.srt` transcript.
2. Ask the app to build a course outline from the transcript.
3. Review the outline and approve it.
4. Generate lessons, quiz questions, written questions, and flashcards.
5. Study lessons, take quizzes, review flashcards, ask the tutor questions, and
   view learning progress.
6. Export course materials.

The app uses the transcript as its course source. It does not watch or understand
the video itself; it works with the words available in the transcript.

## 2. How the pieces fit together

Think of the app as a small workshop with a screen, a local filing cabinet, and
an online AI service:

- **The Streamlit screen** is the part you click and use in your web browser.
  The program runs on your computer, and Streamlit displays it at
  `http://127.0.0.1:8501`.
- **The course workflow** coordinates the work: save the transcript, request an
  outline, create lessons and assessments, and track progress.
- **The Gemini service** writes or explains content and creates number-based
  representations of text that help find relevant transcript passages.
- **The local database** is the filing cabinet. SQLite stores course details,
  transcript segments, lessons, questions, attempts, flashcards, tutor messages,
  and completion records.
- **The transcript search index** is a separate local store. ChromaDB keeps
  searchable transcript sections so the tutor can look for passages related to
  a question.

In broad terms, the flow is:

```text
YouTube captions or uploaded transcript
                  |
                  v
       Clean and save transcript locally
                  |
                  v
        Gemini helps make an outline
                  |
          You review and approve it
                  |
                  v
 Lessons + quiz + written questions + flashcards
                  |
                  v
        Study, ask the tutor, track progress
```

### Technical architecture

The application is a local Python web application. Streamlit runs the UI and
reruns `app.py` when the user changes a control. The UI calls workflow and
service modules directly; there is no separate HTTP backend or frontend
JavaScript application.

| Layer | Implementation | Responsibility |
| --- | --- | --- |
| UI | Streamlit in `app.py` | Course selection, outline approval, lesson display, quizzes, tutor, progress, and exports. |
| Workflow | `agents/orchestrator.py` | Coordinates course creation, transcript indexing, lesson generation, persistence, progress, failure recovery, and resume. |
| AI agents | `agents/` | Builds curricula, lessons, assessments, quality checks, and tutor answers using typed Pydantic schemas where applicable. |
| Model gateway | `services/llm_client.py` | Calls the Google Gen AI SDK for Gemini generation and embeddings; applies request pacing, timeouts, and bounded retries. |
| Relational storage | SQLite via Python's `sqlite3` module | Stores course state, transcripts, lessons, assessments, learner activity, and job state. |
| Semantic retrieval | ChromaDB in `rag/vector_store.py` and `rag/retriever.py` | Stores transcript vectors and retrieves relevant source passages for tutor and lesson context. |
| Input and output | `ingestion/`, `services/` | Parses YouTube captions and transcript files; provides quizzes, flashcards, analytics, lesson formatting, and exports. |

There is no separate hosted application database. SQLite and ChromaDB data are
kept in local storage directories configured by `DATABASE_PATH` and
`CHROMA_PATH`.

### Main processing sequence

1. `ingestion/youtube_loader.py` fetches captions, or
   `ingestion/upload_loader.py` parses a supplied transcript.
2. `agents/orchestrator.py` cleans and stores transcript segments in SQLite.
3. `rag/chunking.py` prepares transcript chunks. `agents/curriculum_agent.py`
   asks Gemini to identify topics and return a typed course outline.
4. The outline is saved in `courses.outline_json` with status
   `awaiting_approval`; the user reviews it in the UI.
5. After approval, the orchestrator attempts to embed transcript chunks and
   index them in ChromaDB. An indexing failure is logged and lesson creation
   continues with sampled transcript context.
6. Missing lessons are generated in batches of two. Up to
   `GEMINI_MAX_CONCURRENCY` batches can run concurrently. Each completed lesson
   is written to SQLite and job progress is updated.
7. Assessments and flashcards are generated if they do not already exist. The
   job and course statuses are then updated. On failure, saved lessons remain
   available for a later resume.

### SQLite records

`models/database.py` defines the schema. The primary records are:

| Table | Stored data |
| --- | --- |
| `courses` | Course metadata, generation status, settings, and serialized outline. |
| `segments` | Transcript text and start/end timestamps. |
| `jobs` | Per-course generation status, progress, message, and error. |
| `lessons` | Lesson position, module, generated content, source references, and completion flag. |
| `questions` | Quiz and written-assessment prompts, options, answers, explanations, and topics. |
| `attempts` | Learner answers, correctness, and attempt timestamps. |
| `flashcards` | Card content and spaced-repetition scheduling state. |
| `card_reviews` | Flashcard ratings and review timestamps. |
| `tutor_messages` | Saved tutor conversation and source citations. |
| `quality_reports` | Course quality-check output. |

The course ID links its related records. The schema enables SQLite foreign-key
checking, and indexes support transcript-segment, quiz-attempt, and
course-question lookups. Lesson content, outline, settings, and some structured
fields are serialized as JSON text in SQLite.

### Tutor retrieval (RAG)

RAG means retrieval-augmented generation: retrieve likely source text first,
then ask the model to answer with that context. The app splits transcripts into
overlapping chunks, requests embeddings from Gemini, and stores the vectors and
timestamps in ChromaDB. For a tutor question, the same embedding model converts
the question into a vector; ChromaDB returns the nearest transcript chunks.
The tutor uses those chunks to answer and the app builds citations from stored
chunk metadata. If vector retrieval is unavailable or has no sufficiently
relevant result, the app may use keyword search or clearly labeled general
context, depending on the result.

## 3. How course creation works

### Step 1: Get the transcript

For a YouTube URL, the app tries to get English captions first. If English is
not available, it may translate another available caption track to English; if
that does not work, it can use an available track in its original language.
You can instead upload a transcript file.

If captions are disabled, unavailable, or blocked by YouTube, use a transcript
file. The app cannot create accurate transcript-based lessons when it has no
transcript text.

### Step 2: Build and review the outline

The app splits transcript text into smaller sections and asks Gemini to identify
topics and organize them into lessons and modules. It saves the outline so you
can review it before generating the full course.

For a transcript that fits into one model request, topic identification and
outline creation are combined. Longer transcripts are handled in batches. The
outline is intended to provide a lesson for each identified topic, but the AI
can still miss or misunderstand material. Review the outline before approving
it.

### Step 3: Generate the learning materials

After approval, the app creates lessons and assessment materials. Lesson
instructions ask for plain explanations, why a topic is useful, examples,
applications, and practice activities where appropriate.

Lessons are saved as they finish. If generation fails partway through, the app
can keep completed lessons and resume when you try again. Gemini can still
return incomplete or imperfect material, so review important content.

To reduce waiting without sending extra generation requests, the app now asks
Gemini to create two separate lessons in each request. Each lesson is still
saved separately, and the app checks that the response contains both requested
lessons before saving either one. This reduces the number of lesson-writing
requests by about half. It cannot remove time spent waiting for Gemini or bypass
the account's request and output limits.

### Step 4: Study and track progress

The app stores lesson completion, quiz attempts, flashcard review history, and
tutor messages locally. Quiz selection can use recent performance and topic
mastery to emphasize questions that may need more practice. Mastery is not
shown for a topic until there are at least three attempts.

## 4. How the tutor finds answers

When you ask a tutor question, the app first searches transcript sections that
are similar in meaning to the question. If that search does not find anything,
it tries a keyword-based search through the saved transcript.

- When relevant transcript sections are found, the tutor is asked to use them.
  The app attaches timestamp references from those sections.
- When no relevant sections are found, the tutor may answer using general
  knowledge. It should label this as added context that was not verified against
  the retrieved course material.
- If Gemini cannot produce a reply, the app reports the failure and keeps the
  question saved so you can retry.

The tutor is not a guarantee that every answer is correct. Check important
claims against reliable learning materials, especially when an answer is marked
as general knowledge rather than course-based.

## 5. Running the program on Windows

Open PowerShell in the project folder and run:

```powershell
python -m venv .venv
Set-ExecutionPolicy -Scope Process Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Open `.env` in a text editor and put your Gemini API key after
`GOOGLE_API_KEY=`. Keep the key private: do not paste it into chat, screenshots,
source code, or a public repository.

Then start the app:

```powershell
streamlit run app.py
```

Open `http://127.0.0.1:8501` in your browser. To run the focused tests:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_core.py -q
```

## 6. Settings you may see in `.env`

`.env` is a local settings file. `.env.example` lists the available settings.
The main ones are:

| Setting | In everyday terms |
| --- | --- |
| `GOOGLE_API_KEY` | The private credential that lets the app contact Gemini. |
| `GEMINI_MODEL` | Which Gemini model writes course content and tutor replies. |
| `GEMINI_EMBEDDING_MODEL` | Which Gemini model turns text into searchable number patterns. |
| `GEMINI_MAX_CONCURRENCY` | The maximum number of lesson batches the app can work on at once; each batch contains two lessons. |
| `GEMINI_MIN_REQUEST_INTERVAL` | The minimum delay, in seconds, between model requests. |
| `GEMINI_REQUEST_TIMEOUT_MS` | How long a Gemini request may wait before the app stops waiting and reports an error; the default is 120,000 milliseconds (2 minutes). |
| `DATABASE_PATH` | Where the local SQLite course database is kept. |
| `CHROMA_PATH` | Where the local transcript search index is kept. |
| `EXPORT_DIR` | Where exported files are saved. |
| `TOP_K` | How many likely transcript matches to inspect for a question. |
| `MIN_RELEVANCE` | How similar a transcript match must be before it is accepted. |
| `CHUNK_SIZE` | The approximate size of transcript sections used for search. |
| `CHUNK_OVERLAP` | How much neighboring sections repeat so a split does not lose context. |

Change settings only when you know why you need to. For example, allowing more
simultaneous requests may speed up generation, but it does not increase the
Gemini account's quota and may make quota errors more likely.

## 7. API keys, speed, and quota errors

The app uses the Gemini key configured in `GOOGLE_API_KEY`. It spaces out model
requests and observes retry delays reported by Gemini when possible. This can
make generation more reliable, but it cannot make a provider quota unlimited.
Transcript indexing now reports chunk-by-chunk progress. Gemini requests also
have a configurable timeout so a network call does not wait indefinitely. If
Gemini reports that the quota will reset hours later, the app reports the error
instead of keeping the screen waiting for hours.
If transcript embeddings reach their quota, lesson generation can continue
without tutor search. If the app closes during generation, reopening it marks
that run as interrupted and offers a safe resume using the lessons already
saved.

An HTTP `429` or `RESOURCE_EXHAUSTED` message means Gemini is temporarily refusing
requests because a usage limit has been reached. Wait for the retry period shown
in the message, check the current limits for the Gemini project and model, or
review the billing and quota settings for that project. Multiple keys belonging
to the same Google project may share project-level limits; adding keys does not
necessarily add capacity.

## 8. Where data goes

Course records and study progress are stored on the computer in the configured
SQLite database. Transcript search data is stored locally in the configured
ChromaDB folder.

However, the app sends prompts and transcript excerpts to Google's hosted Gemini
service when it requests generated content or embeddings. Do not use private or
sensitive transcripts unless you are comfortable sending the relevant text to
that service and have permission to do so. Check the current Google Gemini terms
and your account's data settings for the rules that apply to your account.

Deleting a course removes its database record and its transcript search entries.
Keep a backup of the storage folder if you need to preserve course data.

## 9. Common problems

| What you see | What it usually means | What to try |
| --- | --- | --- |
| Captions are unavailable | The video has no accessible captions, or YouTube blocked the request. | Upload a `.txt`, `.vtt`, or `.srt` transcript. |
| `GOOGLE_API_KEY is missing` | The app cannot find a key in `.env`. | Create `.env` from `.env.example` and add the key. |
| `429` or `RESOURCE_EXHAUSTED` | The Gemini project or model has reached a request or usage limit. | Wait as instructed, then check quota and billing settings. |
| Tutor says it found no course excerpts | Search did not find transcript sections matching the question. | Ask using words from the lesson or transcript; treat any added-context answer as general knowledge. |
| Course generation stops partway through | A transcript, model, network, or quota request may have failed. | Read the displayed error and retry; completed lessons are saved for resuming. |
| A lesson looks incorrect or incomplete | AI-generated content can misunderstand or omit material. | Compare it with the transcript and edit or regenerate as appropriate. |

## 10. Project map

These are the main places to look when maintaining the project:

| File or folder | What it is responsible for |
| --- | --- |
| `app.py` | The browser interface and its tabs. |
| `agents/orchestrator.py` | The steps that create, save, resume, and delete courses. |
| `agents/curriculum_agent.py` | Transcript topic extraction and course outline creation. |
| `agents/lesson_agent.py` | Instructions used to generate an individual lesson. |
| `agents/tutor_agent.py` | Tutor search, answer generation, citations, and conversation saving. |
| `services/llm_client.py` | The shared Gemini connection, request pacing, and retries. |
| `config/settings.py` | Reads local environment settings and prepares storage folders. |
| `models/database.py` | SQLite tables and basic database operations. |
| `rag/` | Breaks up transcript text, indexes it, and searches it. “RAG” means finding source passages before asking the AI to answer. |
| `ingestion/` | Reads YouTube captions and uploaded transcript files. |
| `services/` | Quiz, flashcard, progress, lesson display, and export helpers. |
| `tests/test_core.py` | Focused checks for important app behavior. |

## 11. Important limitations

- Course quality depends on the transcript quality and the AI's interpretation.
- The app cannot recover speech that is missing from the transcript.
- Transcript search may find no match even when a related idea is present.
- General-knowledge tutor answers are not evidence that the video covered the
  topic.
- Generation speed and availability depend partly on the network and Gemini
  quotas.
- This application is a learning aid, not a substitute for checking important
  information against trusted sources.
