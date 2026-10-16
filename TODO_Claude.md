# TODO_Claude.md — Improvement Plan for Researcher, Tutor & Composer

Generated: 2026-05-23 | Project: ai-agent v2.2.0 | Branch: developer
Updated: 2026-10-14 — see TOMORROW at the top: multi-user queueing, tag, confirm quiz.
Statuses re-verified against the code, not against previous checkboxes.

---

## ▶ NEXT — composer control and keeping the score

Asked for 2026-10-15, after v3.1.0 landed.

### 0. First, two things we are getting wrong today

Found while reading ACE-Step's own parameter docs, not yet applied, because
v3.1.0 was tagged on what Bogdan actually heard:

- **`shift=3.0` is required for turbo and we do not send it.** INFERENCE.md is
  explicit: `guidance_scale` is auto-corrected for turbo models but `shift` is
  **not**, the default 1.0 is applied as-is, and "the only timestep/guidance
  parameter you need to set manually for turbo is `shift=3.0`". We run
  `acestep-v15-turbo`, so every song so far was generated outside its
  documented configuration. One line, and it should improve everything.
- **`instrumental: bool` exists** and we do not use it. We infer "instrumental"
  from empty lyrics, which is a guess; the model has a switch. The documented
  convention for the lyrics field is the literal `[Instrumental]`.

### 1. Real control over the composition

The honest finding: ACE-Step has **no structured fields** for instruments or
voice. `caption` is free text capped at 512 characters, and that is where
genre, mood and instrumentation go. Only these are structured: `bpm`,
`keyscale`, `timesignature`, `duration`, `vocal_language`, `instrumental`.

So "more control" is two separate jobs:

**a. Build the caption rather than making the person write it.** A form —
instruments (multi-select, as the Score tab already has), ensemble size, mood,
era, production style, mix notes — composed into one caption under the 512
character cap, with the raw text still editable underneath. The composer
session already holds genre, key, time signature and tempo, and those should
prefill rather than be retyped. This is the highest value for the least work.

**b. Voice is a reference-audio problem, not a parameter.** There is no
timbre, gender or singer field. The lever is `reference_audio`, which is
*separate* from `src_audio`: `src_audio` carries the structure for a cover,
`reference_audio` guides the style. So a "voice" feature means letting someone
keep a small library of reference clips and pick one — which also raises a
question worth settling before building it: whose voice, and recorded with
whose consent. For a household that is a real conversation, not a checkbox.

Also worth exposing once the form exists: `inference_steps` (8 for turbo,
32-64 for base) as a quality/time dial, and `audio_cover_strength` is already
there as "Faithfulness".

### 2. Keeping the score

Two cases, very different in cost.

**a. Cover mode already has the score and throws it away.** `/composer/song`
computes the MusicXML to render the MIDI and then discards it. Persisting it
is small: `save_composition` already stores `musicxml`, so the song route
should save the composition it performed and link the audio to it. The
`compositions` table needs one new column for the audio path — it has
`musicxml` but nothing for rendered audio — so this is a migration plus a
save call. After that "save this song" gives you both the notation and the
recording, which is exactly the right artefact pair: the score is correct, the
audio is a performance.

**b. text2music mode has no score at all**, and getting one means audio to
notation — a genuinely hard, separate problem. Options to look at: Spotify's
`basic-pitch` (permissive, pip-installable, monophonic-ish), or ACE-Step's own
**Track Separation** to get stems first and transcribe each. Expect
approximate results: polyphonic piano transcription is still research. Worth
prototyping before promising it in the UI, and worth saying plainly in the UI
that a transcription is an approximation while the cover-mode score is exact.

### 3. Housekeeping that will bite

- Nothing prunes `static/generated/`. Songs are ~350 KB for 20s, so a few
  minutes each; with a family using it this grows without bound.
- `stop_if_idle()` exists and nothing calls it on a timer. The engine is
  stopped when someone else needs the card or from the Free GPU button, which
  works but means it can sit holding ~10 GiB if nobody else asks.

---

## ▶ RELEASED v3.1.0 (2026-10-15) — composer song audio (ACE-Step)

Implemented and released 2026-10-15. Bogdan tried it and approved the tag.

### What it does

A **Song** tab in the composer. The score stays the thing that is *correct*;
this renders a performance of it. With "Perform my score" ticked, the last
generated score goes through the existing path and the result is the source
audio for an ACE-Step `cover`:

    ABC -> MusicXML -> MIDI -> fluidsynth WAV -> ACE-Step cover -> MP3

Measured with a real 8-bar study: MusicXML 7923 chars, MIDI 336 B, WAV 3.9 MB,
MP3 353 KB — at the score's own tempo and key (96 bpm, C major, 4/4 came back
in the metadata). Without a score it falls back to text-to-music from the
description, which is a parallel generator rather than a renderer.

### Why it is a separate process, not an import

Yesterday's note said it "integrates like the diffusers pipelines". That was
wrong. ACE-Step pins `torch==2.10.0+cu128` and `transformers<4.58`; we run
torch 2.13.0+cu130 and transformers 5.14. Installing it here would downgrade
torch, CUDA and transformers — breaking the image and video pipelines and
almost certainly CrewAI. It gets its own checkout and `uv sync` venv
(7.6 GB) at `/home/bogdan/ai/ACE-Step-1.5`, and we talk to its REST API.

### Why the broker owns its lifecycle

ACE-Step has **no unload endpoint** — `offload` is init-time only. A
long-lived server would hold VRAM with nothing able to reclaim it, which on
one card means the tutor and researcher get pushed to the CPU. So `Need("song",
"acestep")` is its own residency kind, and `_make_resident` stops the process
when anything else wants the card. Measured:

    engine holding weights          5.0 GiB free
    tutor task arrives, broker stops it
    after                          15.1 GiB free

`--no-init` means the server answers /health in ~4s holding *no* VRAM, and
loads weights on the first request — inside the reservation, where it belongs.
No LM is ever loaded: `cover` skips it entirely and text2music with thinking
off does not need it.

### Timings on the 5070 Ti

    server start                    4.0s
    first song (model load)         141s
    cover from a rendered score      42.7s
    through the HTTP route          115s (cold engine)
    VRAM held while generating      ~9.8 GiB

### The four bugs, all mine, all found by running it

1. **stdout to a pipe killed the model download.** `stdout=subprocess.PIPE`
   with no reader: once the parent was gone every write failed with EPIPE, and
   the download died of its own progress output — "Failed to download main
   model: Both HuggingFace and ModelScope downloads failed. HF: [Errno 32]
   Broken pipe". Output goes to `logs/acestep.log` now.
2. **`task_ids` instead of `task_id_list`** on `/query_result` returns
   `{"data": []}` with HTTP 200, which polls for ever. The field name is in
   docs/en/API.md; I guessed instead of reading.
3. **Environment leakage caused a cuDNN mismatch.** Passing our `PYTHONPATH`
   and `VIRTUAL_ENV` to the subprocess made it load a mix of two cuDNNs:
   `CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH`, on the cover path only, because
   that is the one that runs convolutions over the source audio. Scrubbing
   those variables fixed it outright.
4. **The real error was buried.** `status: 2` at the top level, the reason
   inside the JSON-string `result`. Reading only the top level reported
   "generation failed" and nothing else, which hid bug 3 for a whole run.

Also: `uvicorn --reload` orphans the engine on every code change, and `stop()`
could only kill a process it started — so it now matches `acestep.api_server`
on our port and reclaims orphans too.

### Still to decide

- **Retention.** Songs land in `static/generated/` beside the images and
  video, which is gitignored and never pruned. A few minutes of audio per
  song will add up; nothing deletes them yet.
- The idle window is 300s (`ACESTEP_IDLE_SECONDS`), but nothing sweeps it —
  `stop_if_idle()` exists and is not called on a timer. Today the engine stops
  when someone else needs the card, or from the Free GPU button.
- The XL 4B model would sound better and fits at ≥12 GB with offload. Untried.

---

## ▶ RELEASED v3.0.0 (2026-10-14)

Major, because the model a request runs on and the order requests run in both
changed shape.

### Multi-user with one GPU — done

`gpu_broker.py` owns GPU residency and the queue. A task declares what it
needs (`Need("ollama", "gpt-oss:20b")`, `Need("torch", "zimage")`); when its
turn comes the broker makes that true, runs it, and stands aside. No restore
step — the next task declares its own need, which is what stopped the old path
reloading 10.5 GiB nobody wanted.

Order is first come, first served. Priority is self-declared and carries a
reason everyone waiting can read; whoever is running is never interrupted.

Verified with two real users on two models through the Research tab:

```
  resident: ollama:gpt-oss:20b
  ▶ rbogdan — research, 24.8s in
  · rraluca — research, waiting 20.9s
  ollama ps -> gpt-oss:20b only, count: 1
```

Before: both models loaded and Ollama spilled one to CPU — `ollama ps` showed
`21%/79% CPU/GPU` with 1.6 GB resident.

**Every path queues**, which is the part that matters. A queue covering some
paths is worse than none, because it looks like it works. `test_gpu_coverage.py`
walks the AST and fails if any `run_in_executor` sits outside a reservation
without an explicit `# gpu-queue-exempt: <reason>`.

### The three bugs that made it look broken

1. **The GUI model choice was ignored.** The picker wrote to `users.model` *and*
   to `app.state.current_model`; crews were built from the latter, which is
   re-read from `.env` on every restart. After a restart the picker showed the
   saved model while crews ran the one from `.env` — deleted months earlier,
   hence a 404 for `huihui_ai/mistral-small-abliterated`. The global was also
   process-wide, so the last person to touch the picker changed everybody
   else's default. `user_crews.py` now resolves per-user from the database.

2. **`chat_template_kwargs: {"enable_thinking": ...}` is silently ignored by
   Ollama.** It was used in three places. The switch Ollama honours is
   top-level `think`. Measured on qwen3.5:9b, `num_predict: 24`, prompt "Say
   exactly: READY":

   | setting | visible output |
   |---|---|
   | default | `''` |
   | `chat_template_kwargs` | `''` — ignored |
   | `think: False` | `'READY'` |

   So thinking was on for every thinking-capable model regardless of intent,
   and the model spent its whole budget in a channel Ollama returns
   separately. gemma4:26b: 2031 chars of thinking, zero of response,
   `done_reason: length`. From outside that is an agent that writes a plan and
   stops. All three crews now share `llm_options.py`.

3. **`num_gpu: 99`** pinned every layer to the card. gemma4:26b then asks for a
   17 GB buffer on 15.5 GiB: `cudaMalloc failed: out of memory`, or the
   lopsided split above. Ollama decides the split now.

### Tool calling — by capability, not by name

P1 forced every Ollama model into ReAct text. Measured with thinking off, the
tools-capable models all emit correct structured calls:

| model | capability | native call |
|---|---|---|
| gpt-oss:20b | tools | yes |
| gemma4:26b | tools | yes |
| gemma4:12b | tools | yes |
| qwen3.5:9b | tools | yes |
| phi4:14b | — | no tool support |

Forcing them into ReAct broke two of them. gpt-oss emitted a structured call
anyway, CrewAI returned the list, and it was stringified into the answer —
`[ChatCompletionMessageFunctionToolCall(...)]`, a run that "succeeded" in two
seconds having done nothing. gemma4 wrote a plan, never an `Action:`, and
looped until its step budget ran out.

P1 now asks Ollama (`/api/show` → `capabilities`), cached, falling back to
ReAct when Ollama cannot answer. `_FORCE_REACT_MODELS` is the escape hatch for
a model that advertises tools but answers in prose. P6 translates a native
call into ReAct text as a net for anything still in ReAct mode.

On the same prompt: gemma4 went from one Thought to 5839 chars with
comparison tables; gpt-oss from a raw tool-call object to 4351 chars.

### Also fixed

- `reset_memory()` deleted CrewAI's kickoff-outputs database *file*; the next
  `kickoff()` reconnected, SQLite made an empty database, and CrewAI's DELETE
  found no table and raised — killing a run that had already produced an
  answer. Roughly every second request with two people taking turns. Now
  cleared through CrewAI's own API.
- `generate_ai_image.result_as_answer = True`, so one call ends the run and
  the image becomes the whole answer — and the description never said when not
  to use it. gemma4 drew a picture for a travel question and stopped. Both
  media tool descriptions now say so explicitly.
- The GPU panel was a fixed overlay and covered the typing box on all four
  tabs. It is in the document flow now, last flex item of `<body>`.
- The collapsed reasoning box moved above the answer, where the live one
  already is. Below, it jumped across the answer when the result arrived.

### The lesson, again

A green suite is not evidence a feature works, and this time it was sharper
than usual: the queue was correct and `TestClient` was buffering the stream,
so my own measurement said it was broken. Two concurrent `curl -N` against a
real uvicorn proved it worked. **Verify streaming against a real server, never
through TestClient.**

### Still open

- ACE-Step integration is v3.1.0, agreed 2026-10-14. Plan at the top of
  this file; the broker reservation it needs is the only GPU work left.
- The composer's long-form score path reserves the GPU but does not render a
  queue position.
- `ollama ps` can read zero models mid-swap; `/gpu/queue`'s `resident` is the
  reliable answer.

---

## ▶ DONE 2026-10-13 — superseded

### The one real task: make sheet generation stream

**Symptom:** pressing "Write the lesson" shows nothing in the page. The thinking
appears only in the server terminal, and so does the result.

**Cause, confirmed:** `/tutor/sheets/lesson` and `/tutor/sheets/story` are
blocking POSTs. CrewAI's `verbose=True` writes progress to the server's
**stdout**, not into the HTTP response, so the browser sees nothing until the
call returns 30-240s later. Every other model-backed tutor feature — chat,
`/tutor/lessons`, `/tutor/passages`, quizzes — goes through
`_stream_tutor_crew` in `routes/tutor.py`, which captures stdout with
`config._QueueWriter` and emits SSE. The sheet routes bypass all of it.

**Fix:** reuse that machinery.
1. Read `_stream_tutor_crew` and `_QueueWriter` (routes/tutor.py ~line 100-200).
2. Give the sheet routes an SSE variant that streams the same thinking events
   and ends with a `result` event carrying the markdown / story JSON.
3. In `js/tutor-sheets.js`, consume the stream and show it where the spinner is
   now (`#sheets-working`), then build the PDF from the final payload.
4. Keep the blocking routes as they are — the tests use them, and they are the
   right shape for a scripted call.

Do not start by rewriting the UI. The streaming path is the whole problem.

### What was fixed today and IS verified

All verified end to end against the real model (mistral-small-abliterated),
not mocked:

- **Content-Disposition crashed on non-ASCII filenames.** HTTP headers are
  latin-1; the default lesson title was `f"{language} — Lesson"` and that em
  dash is U+2014. Starlette raised *after* the PDF was built, so a good
  response became a 500. Grammar sheets were named `French-homework-21.pdf`
  (ASCII) which is exactly why they worked and lessons did not.
  Fixed with RFC 6266 in `researcher/downloads.py`: ASCII fallback in
  `filename`, real name percent-encoded in `filename*`.
  **This was my bug, introduced in v2.12.0 and v2.14.0** — the pre-existing
  code in `ingestion.py` sanitises filenames properly and was never affected.
- **The math tutor had the same latent fault** and is fixed by the same change
  (one line in `_pdf_response`). A saved worksheet titled "Contrôle — Théorème
  de Pythagore" now downloads; before it raised UnicodeEncodeError.
- **pdflatex stdout is not always UTF-8.** It echoes source lines in
  overfull-box warnings in its own encoding, so an accented French word in a
  long line made `subprocess` raise, and `compile_pdf` threw away a PDF that
  had built fine. Now `errors="replace"`.
- **Empty level.** `loadSheetSkills()` is async and was not awaited, so
  generating straight after opening the panel sent `level:""` and the model
  spent minutes writing for a "-level student". Guarded on both sides: the
  client awaits the dropdowns, the server rejects a bad level with 400 before
  starting the model.
- **A lesson took 241s.** The full `lesson_task` is a teacher's scheme of work.
  Added `lesson_sheet_task` — compact, **37s** — now the default, with a
  "Full lesson" checkbox for the thorough one.

Full browser sequence, timed, with an accented title:
`write 200 (33s) -> sheet PDF 200 (105 KB) -> answer key 200 (79 KB)`.

### NOT verified

Whether any of this renders in Bogdan's browser. He has not yet confirmed a
PDF appearing in the preview pane. The backend is proven; the front end is not.
**Ask him to confirm before assuming the UI half works.**

### Standing instruction

Do not commit UI work until Bogdan has clicked it. A green suite is not
evidence — 763 tests passed while the Lesson tab was completely broken,
because none of them pressed a button.

### The lesson I kept re-learning

Three encoding bugs in a row, each treated as a one-off: the `°` that printed
as `ř` (T1 font encoding), pdflatex's stdout, and the HTTP header. Bogdan had
to point out that it was a pattern. **When an encoding bug appears, check the
other boundaries in the same path** — file write, subprocess, HTTP header,
font encoding — rather than fixing the one instance.

### Release state

Released as `v3.0.0`. The previous tag `v2.16.0` does not include any
of this work.

### Also noticed

- CrewAI 1.15.14 installed, 1.15.27 available. `patches.py` pins behaviour
  against the tested range, so read it before upgrading.
- Crew `verbose=True` floods the terminal and buries the `[tutor]` log lines.
  Consider `verbose=False` for sheet crews once streaming carries the progress.

---

## ▶ STATE OF PLAY (2026-10-12, after v2.15.0)

Both tutors are feature-complete for what was asked. Everything below is
backlog, not blockers.

### Math tutor — done

53 topics Grade 1–12 at IB level, 32 with drawn figures, English/French/
Romanian plus bilingual pairs, printable worksheet + answer key, saved sheets
that reprint identically, login. Answers computed with SymPy; no model.

### Language tutor — done

Printable homework, tests, exams and challenges for French, German and
Romanian, plus model-written story + summary sheets. Grammar answers come from
checked-in language packs (~60 verbs / 60 nouns / 30 adjectives each); regular
verbs are one line and the paradigm is derived by conjugator.py, verified
against the hand-written entries. Chat, voice, FSRS vocabulary, pronunciation
scoring and everything else are untouched and tested to still answer.

**Verified against the real model on 2026-10-12** (GPU back, driver 615.71.09):
mistral-small-abliterated returned clean fenced JSON, parsed and printed
first time. It undershoots the word count — 126 words for a requested 180 —
so ask for more than you want.

### Backlog, in rough priority order

1. **Math marking UI.** `/math/results` and `/math/progress` exist and are
   tested, but nothing posts to them, so the Progress panel can never fill.
   Build the marking screen or remove the button — leaving it is the worst
   option.
2. **Worked steps for 16 of 53 math topics**, which currently print the answer
   with no method. Fine for `times_tables`; wrong for `simultaneous`,
   `differentiation`, `complex_numbers`, `circle_equations`, where the method
   is what gets marked.
3. **Marks on math sheets.** The language tutor has a mark scheme
   (`printing.Section.marks_each`, subtotals, a total box); the math tutor
   does not. It should borrow it.
4. **3D figures for `vectors_3d`**, which is flagged `geometry: true` but
   draws nothing.
5. **Romanian has 55 verbs to French's 72** — deliberately conservative where
   vowel alternation made the forms uncertain (`a mânca` → `mănânc`). Adding
   more means hand-writing those tables.
6. **Model choice.** `MODEL=mistral-small-abliterated` is a poor fit for
   reasoning; `qwen3.6:27b` is already pulled and is the better default. Only
   matters for the story sheets and the chat tutor — the grammar and math
   sheets use no model at all.

### Things that cost time to rediscover

- The app is launched as `uvicorn researcher.main:app --reload` under VS Code
  debugpy, NOT via the `__main__` block — so `uvicorn.run(...)` arguments have
  no effect. Behind HAProxy it wants `--proxy-headers
  --forwarded-allow-ips=<haproxy-ip>` on the command line.
- `https://ai-agent.hlevca.com` is live and serving with a valid certificate.
- Pronunciation and diction stay on screen; that was a decision, not an
  omission.

---

## Session notes — 2026-10-12 (v2.13.0) — GEOMETRY AND LANGUAGES

### Geometry rebuilt

The weakest part of the math tutor, and the part that decides whether a
curriculum teaches reasoning or skips it. Eight new topics, all drawn, all
computed: circle_theorems (with the theorem stated as the reason),
inscribed_shapes, sectors_arcs, similar_triangles, angle_rules,
analytic_geometry, circle_equations, geometry_3d. Eight new renderers.
53 topics now, 32 with figures.

### English, French, Romanian — and pairs

~150 whole sentence frames in phrases.yaml, translated once and checked in.
Nothing is translated at run time, deliberately:

- A model in the generation path breaks "a seed is a worksheet" — the French
  wording would drift between runs, so a reprint would not match the sheet.
- Models mangle LaTeX.
- Terminology: fr "coefficient directeur" not "gradient", "médiatrice" not
  "bissectrice perpendiculaire"; ro "mediatoarea", "rația", "patrulater
  inscriptibil". A general translator gets these plausibly wrong.

Frames are sentences, not words, so a translation can reorder placeholders.
T() wraps only the literal parts in \\text{}, leaving the mathematics in math
mode. A bilingual spec ("fr+en") renders both and generate() splits them into
question/question_alt, so generators stay unaware of how many languages exist.

Verified: pdflatex + T1 + lmodern renders ș ț ă â î and é è à correctly —
tested by rendering, not assumed. (pdftotext extracts them badly, which is a
red herring.)

### Bugs found by printing the sheet

- A literal ° reached LaTeX and printed as "45ř" — T1 maps that byte to
  r-caron. It compiled cleanly. Now ^\\circ.
- Math mode does not line-break, so long worded questions ran off the right
  margin. Questions are now prose with inline mathematics.
- SymPy writes the segment area as "-36 + 60\\pi"; exact values now lead with
  the positive term.
- \\documentclass[14pt] made primary type SMALLER: article silently discards
  anything but 10/11/12pt. Fixed with extarticle.
- fractions_unit and percentages were emitting an untranslated " of " — caught
  by the English-leak test, not by reading.

### Still to do

- **Language tutor printables** — homework, tests, exams, grammar and logic
  exercises, storytelling with summary questions, printable sheets and answer
  keys, reusing worksheet.py. Requested twice now; still not started.
- A marking UI (the /math/results endpoint and progress view exist, but nothing
  posts to them yet).
- 3D figures for vectors_3d.
- Exam papers with marks per question and a total.

## Session notes — 2026-10-12 (v2.12.0) — MATH TUTOR

A new third tutor: printable IB-level mathematics, Grade 1 to Grade 12,
with drawn geometry. Requested as higher priority than the composer.

### The governing rule

**No language model produces a question or an answer.** SymPy computes every
answer; matplotlib draws every figure from the same parameters. This is a
sharper version of the principle the composer arrived at (deterministic code for
correctness, LLM for variety) and it is non-negotiable here for a reason worth
writing down: a wrong note in a composition is a bad listen, but a wrong answer
on a worksheet teaches a child something false, and whoever marks it then
reinforces the error. An LLM that is right 97% of the time is useless for this.

### Delivered

- `mathtutor/curriculum.yaml` — 12 grades, 68 topic slots, every one backed by a
  working generator (enforced by a test). Each grade carries its IB equivalent.
- `mathtutor/generators.py` — 32 generators. Arithmetic through number bonds,
  fractions, ratio, indices, quadratics, surds, sequences, probability,
  functions, calculus, logs, binomial, complex numbers, integration by parts,
  **proof by induction, implicit differentiation, first-order differential
  equations, 3D vectors, Maclaurin series, discriminant theory, trig identities**.
- `mathtutor/geometry.py` — 14 figure-drawing generators and 7 renderers
  (right triangle, rectangle, circle, compound, SAS triangle, coordinates,
  regular polygon). SVG for the browser, PDF for print, black line work only.
- `mathtutor/worksheet.py` — LaTeX typesetting. Worksheet and answer key from
  one problem set, so the key cannot drift from the sheet.
- `mathtutor/storage.py` — worksheets stored as *recipes*, plus per-question
  marking and topic-accuracy reporting.
- `routes/mathtutor.py` — 11 endpoints; `static/math.html` + `js/math.js` +
  `css/math.css`.
- `tests/test_mathtutor.py` (18) and `tests/test_mathtutor_api.py` (22).

### Decisions worth remembering

- **A seed is a worksheet.** Generation is deterministic, so storage keeps only
  (grade, topics, count, seed). A sheet reprints identically months later and
  its key always matches. ~100 bytes per saved worksheet.
- **The preview is the actual PDF in an iframe**, not an HTML rendering. The app
  had no KaTeX/MathJax, and adding a CDN dependency to an offline-first tool to
  produce a *worse* approximation of the thing it already generates in 0.1s was
  the wrong trade. What you see is what prints.
- **Both namings are selectable.** `grade8`, `MYP3`, `Grade 8` and `8` all
  resolve to the same level, and both appear on the printed sheet. PYP/MYP
  number from the start of the programme — Grade 8 is MYP 3, not MYP 8 — so the
  equivalence has to be shown rather than assumed.
- **Answers stay exact.** Surds, not decimals: Pythagoras deliberately mixes
  triples (whole answers) with surd answers. A test fails if a decimal leaks in.

### Bugs found by running it rather than reading it

Every one of these would have survived a static review:

1. **Geometry dedup discarded nearly every problem.** Geometry questions share
   one question text ("Find the hypotenuse") and differ only in the figure, so
   keying the duplicate check on text alone threw them away. Key now includes
   the figure and the answer.
2. **`\documentclass[14pt]` made primary type SMALLER.** `article` accepts only
   10/11/12pt and *silently discards* anything else, falling back to 10pt. Fixed
   with `extarticle` (extsizes) plus a graceful fallback.
3. **`\text{cm^2}`** — a superscript inside `\text{}` is invalid LaTeX; the
   answer key for every area topic failed to compile.
4. **`\color` without `xcolor`** — same, silent until the key was built.
5. **Grade 1 was asking "Convert 81 km to m"** and "How many vertices does *a*
   octagon have?" with no picture. Generators now scale on grade, and shapes are
   drawn so a six-year-old can count them.
6. **"A ribbon is 9 L long"** — the word problem's noun was chosen independently
   of the unit. Teaching a child that litres measure length is worse than asking
   nothing.
7. **SymPy's notation is not an examiner's.** `\sin{\left(x \right)}` and
   `\operatorname{atan}` became `\sin x` and `\arctan`.
8. **Implicit-differentiation questions read "x³ + y³ + 10 = 10"** — the constant
   was added back onto the wrong side for templates that never had one there.

### Still to do

- **Language tutor printables** (homework, tests, exams, challenges,
  storytelling + summary) reusing `worksheet.py`. Requested in the same message
  as the math tutor; not started.
- Mixed-topic *exam papers* with marks per question and a total.
- A marking UI — the `/math/results` endpoint and the progress view exist, but
  nothing in the page posts to it yet.
- 3D geometry figures (vectors_3d is flagged as geometry but prints no diagram).
- Worked solutions for topics where `steps` is currently empty (simultaneous
  equations, differentiation).

## Media generation — Muse research outcome (2026-08-11)

**Muse Image and Muse Video are real** (Meta Superintelligence Labs, 7 July 2026) but
**there is no developer API for either**. Meta's announcement lists consumer surfaces only:
the Meta AI app, meta.ai, Instagram Stories (US), WhatsApp (limited countries). Muse Video is
"coming soon" even there. They cannot be called from this app at any price today.

Two claims in the brief that prompted this work were **fabricated** — do not act on them:

- `pip install "museq[server,images]"` / `muse pull` / `muse serve` — **no such package**.
- `https://api.meta.ai/v1` — not a documented base URL. There *is* a Meta Model API in
  public preview and it *is* OpenAI-compatible, but it serves **Muse Spark 1.1**, a
  text/multimodal LLM. Image and video are separate products, not on that API.

What we did instead — **copy Muse's method rather than call it**. Muse's advantage is not
better weights, it is that it inspects and improves its own output. That loop is reproducible
locally, and now is:

- [x] **Critique-and-retry image loop** (`image.py`). Generate → a vision model inspects the
      result against the request → rewrite the prompt → re-render, up to
      `IMAGE_REFINE_ROUNDS` (default 2). `IMAGE_CRITIC_MODEL` defaults to `qwen3-vl:8b`.
      Verified: the critic accepted a matching prompt and correctly diagnosed a mismatch
      ("three red circles instead of a blue sailboat").
      Guards: a broken or missing critic degrades to plain generation, a failed retry keeps
      the earlier image, and a critic that faults the image without offering a better prompt
      stops rather than re-rendering an identical prompt.
- [x] **Video generation** (`video.py`) — Wan 2.2 TI2V-5B (default) or LTX-Video behind
      `VIDEO_BACKEND`, reusing the existing `_gpu_inference_ctx` VRAM choreography, new
      `GenerateAIVideo` tool, `<video>` rendering in the chat UI. Weights are **not**
      downloaded during a chat request: an uncached model returns `hf download <model>`
      instructions instead of hanging for several GB.
- [ ] **Remote/OpenAI-compatible media backend** — deliberately deferred. It delivers nothing
      until a Muse API exists, and is about an hour's work once one does.

**Verified on hardware since (2026-08-20/22).** The refine loop and video generation have
both run end-to-end on the GPU. Video moved to Wan 2.2 TI2V-5B at 1280x704 and gained clip
stitching; see the v2.9.0 session notes.

---

## Status at a glance (verified 2026-08-10)

| ID | Item | Status |
|----|------|--------|
| R1 | Monkey-patch isolation | ✅ done (v2.4.0) |
| R2 | Multi-user crew isolation | ✅ done — except 409 Conflict |
| R3 | Request timeouts & cancel | ⚠️ timeout works; **cancel does not actually cancel** |
| R4 | Consistent error responses | ❌ not started |
| R5 | Structured logging | ❌ not started — 94 `stderr.write` vs 82 logger calls |
| R6 | VRAM spike after image gen | ❌ not started |
| R7 | Input sanitization | ⚠️ attachments capped; prompts unvalidated |
| R8 | Static file caching | ❌ not started |
| T1–T6 | Tutor feature set | ✅ done (v2.3.0) |
| T-A | FSRS replaces SM-2 | ✅ done (v2.6.0-dev) |
| T-B | Comprehensible input | ✅ done (v2.6.0) |
| T-C | Feedback loop / phoneme scoring | ✅ done (v2.9.0) — phoneme-level scoring |
| T-D | TTS first-reply language bug | ✅ done (v2.9.0) — root cause found |
| C1 | Modulation / key changes | ⚠️ harmony now extended; true modulation still open |
| C2 | Dynamics & articulation | ❌ not started |
| C3 | Multi-instrument | ✅ done — entry was stale |
| C4 | Browser score playback | ✅ MIDI + WAV playback in-browser (v2.9.0) |
| C5 | JSON repair hardening | ✅ obsolete — JSON path replaced by ABC (v2.9.0) |
| C6 | Harmonization feedback UI | ❌ endpoint exists, no comparison view |
| C7 | Ear training | ❌ not started |
| C-O1 | Instrument ranges & transposition | ✅ done (v2.11.0) |
| C-O2 | Sampled instruments (sfizz + SFZ) | ⚠️ partly solved — see below |
| C-O3 | Choir / singing voices | ❌ not started |
| X1 | Shared LLM factory | ❌ not started — duplication has grown since |
| X2 | Central settings | ❌ not started — `pydantic-settings` installed but unused |
| X3 | Per-user concurrency | ✅ done — entry was stale |
| X4 | DB migrations | ❌ hand-rolled `PRAGMA` checks |
| X5 | Documentation / diagram | ❌ no architecture diagram |

**Delivered since this file was written, and not in the list above:** faster-whisper STT,
PDF/image OCR with automatic DOCX/ODT export, the extraction fast path, the CrewAI executor
pin, and `max_iter` as a tunable parameter. See the session notes below.

---

## Session notes — 2026-08-22 (v2.9.0)

### Delivered

**Video**
- Wan 2.2 TI2V-5B at its native 1280x704. Measured better than 832x480 by eye;
  my Laplacian "detail" metric said the opposite and was simply the wrong tool —
  it rewards grain and punishes clean gradients, so it cannot compare across
  resolutions. The frames settled it in seconds.
- **Clip stitching**, conditioned on the previous clip's last frame. 3 clips ->
  193 frames -> 8.0s at 720p in 22.6 min, joins measurably smooth (2.2 / 3.7
  against a 3.4 average). ~11 clips gives 30s in ~83 min.
- Duration is the WRONG lever for length: at 5s motion collapsed from 12.2 to
  4.2 as the model ran out of coherent movement. Stitching is the right one.
- Progress streams to the reasoning panel; an 80-minute silent request is
  indistinguishable from a hang.

**Tutor**
- T-D fixed. `base_lang` was inferred from the selected VOICE, so a French reply
  with an English voice selected was scored against English. Short openers split
  into 2-4 word fragments that lose to the base-language bias — which is why only
  the FIRST line was mispronounced. The tutor now declares its target language.
- T-C fixed. Word-level WER returns 0.00 for "perfect but spelled differently"
  AND for "entirely wrong word" — indistinguishable to the learner. Phoneme
  scoring via epitran + panphon separates them and names the substitution
  ("/ʒ/ -> /z/"). Neither metric worked alone: pure feature distance rated that
  error 0.98, since those sounds differ by ~one articulatory feature, so the
  score combines acoustic closeness with segment accuracy.

**Composer** — JSON replaced by ABC end to end, with a validate-and-correct loop.
See the COMPOSER section below, which has been rewritten.

### Late additions (same session)

**Composer specialist.** `COMPOSER_MODEL` now sits beside `MODEL` exactly as
`OCR_MODEL` does. ChatMusician is called directly, not through CrewAI: it has no
tools, needs no planning loop, and a ReAct wrapper adds minutes plus a format it
does not speak. Measured against the general model on the same jazz task —
**7 distinct rhythms vs 2, 11/12 pitch classes vs 7**. Two rhythm values is
exactly the mechanical quality that made earlier output unlistenable.

Its quirks, all found by running it and all handled: the Ollama template is a
bare `{{ .Prompt }}` so `Human:`/`Assistant:` framing must be supplied or it
behaves as a raw completion engine; it restates an 8-bar request three times, so
output is truncated; Ollama returns empty mid-model-swap, so one retry is issued.

**Long form: 15 s → 2.6 min.** A single call holds together for 8-16 bars then
repeats, so length is **assembled, not requested**. The specialist writes an
8-bar seed in its own idiom; `build_long_plan()` develops that motif across a
recurring form with genre-specific extended harmony (`IMaj7 vi7 ii7 V9`,
`V7/ii`, `iv7 bVII7`, `IV Ger+65 V I` — every figure verified to realise in
music21). Measured: 64 bars, 2.6 min, 8 sections, ~5 s. Recurrence is deliberate
and tested: three minutes of *new* material is three minutes of mush.

**Notation sanitising.** The specialist emits ABC that music21 accepts and
stricter renderers reject — `[M:3/4]` dropped mid-bar, `C4 >z` (a broken-rhythm
marker on a rest, which is meaningless), unbalanced slurs. This is the sharpest
form of lesson 4 below: **music21 parsing is not validation.** It read 63 notes
from a tune the browser refused outright, so the validator passed something
unusable.

### Composer — what is still missing

**Measured diagnosis of "repetitive and simple" (2026-08-24).** On a 2.5-minute
jazz piece the output contains:

| | value | |
|---|---|---|
| distinct pitches | 11 | acceptable |
| **distinct rhythms** | **3** | **the core problem** |
| literal repeats of the opening 8-note cell | **5** | too many |

The realisation engine develops the motif by *transform* — transpose, invert,
augment — but its **rhythm never changes** and **no new material is ever
introduced**. music21 is being asked to repeat, not to compose. Three rhythmic
values across three minutes is what the ear hears as mechanical, and it is the
same fault the general chat model had (2 rhythms) that ChatMusician fixed for
short passages (7 rhythms). The long-form path throws that advantage away by
using only the seed's *pitches*.

**The specific fixes, in order of expected effect:**

- [ ] **Carry the specialist's rhythm, not just its pitches.** `extract_motif()`
      takes `(pitch, duration)` pairs but every development preserves the
      original durations. Ask the specialist for a *second* contrasting idea and
      alternate them; vary durations per section (dotted, syncopated, triplet)
      rather than only pitch.
- [ ] **Introduce genuine new material at B and C sections.** Today every
      section is the same motif transformed. A rondo needs episodes that are
      *not* the refrain. Generate one fresh 8-bar idea per contrasting section
      from the specialist and develop that.
- [ ] **Harmonic rhythm should vary.** Currently every section spreads its four
      chords evenly. Real music lingers and then moves — two bars of I, then
      four chords in one bar approaching a cadence.
- [ ] **Voice-lead the parts against each other.** `find_voice_leading_problems`
      only *reports*; nothing acts on it. music21 can resolve sevenths and
      leading tones correctly — the theory is available and unused.
- [ ] **Let the interface cues reach the realisation, not just the prompt.**
      Tempo, key and genre shape the prompt but not the *structure*: a slow
      piece should have longer phrases and slower harmonic rhythm than a fast
      one. This is the "more compositional leverage" ask — the theory is in
      music21, it simply is not being consulted.
- [ ] True modulation (C1): harmony is extended but never leaves the home key.
      music21 realises pivot chords and secondary dominants already; the plan
      has no key-change mechanism to use them.
- [ ] Dynamics, articulation, slurs (C2) — still entirely absent from output.

**Original list, still open:**


- [ ] **The long-form melody is the developed motif, not the specialist's full
      surface.** Structurally coherent, but ChatMusician's idiomatic detail only
      reaches the seed. Carrying it into every section is the biggest remaining
      quality win.
- [ ] Scale-run repair lands *at* the limit (6), not comfortably below it.
- [ ] `polish()` reports on the short path but the polished score is not yet what
      the UI receives — the ABC is.
- [ ] True modulation (C1): harmony is extended but stays in one key.
- [ ] Dynamics, articulation, slurs (C2).
- [ ] Counterpoint beyond parallel-fifth detection.

### What this session taught, and what to do differently

1. **A tool the agent does not need actively harms it.** The composer kept
   emitting JSON because `CompositionPrep` returns a JSON skeleton and the agent
   kept reaching for it. Removing the tool fixed what three prompt rewrites could
   not. The same shape as the dialog-mode bug in v2.4.0.
2. **Instructions do not fix what the model cannot do.** A prompt that explicitly
   banned runs longer than five notes still produced a twelve-note scale, and
   three generations produced wrong-length bars after being given the arithmetic.
   Measure the fault, feed it back specifically, and if it still fails, do the
   mechanical part in code.
3. **Keep the best attempt, not the last.** A correction pass made the tune worse
   (7 short bars vs 6) more than once. Any retry loop needs best-of selection.
4. **Verify metrics before trusting them.** The Laplacian detail score, and the
   first bar-repair implementation (which left 11 bars wrong after "fixing" 5),
   both looked reasonable and were wrong. Cheap checks catch this.
5. **Undeclared dependencies keep biting.** music21 vanished mid-session because
   it was pip-installed and a later `uv sync` pruned it — the same failure that
   broke the app in v2.4.0. Everything imported is now declared.

### Proposals for betterment

**Researcher**
- R3 remains the worst open bug: `/chat/cancel` sets an event nothing reads, so a
  "cancelled" run still holds the GPU. On a single-GPU box that is expensive.
- The extraction and media fast paths showed their value — consider a general
  intent router so obvious jobs never pay for planning.
- X1/X2 (shared LLM factory, central settings) keep growing more expensive: every
  new parameter is added by hand in three crew files.

**Tutor**
- The per-learner **error profile** from T-C was never built. Phoneme scoring now
  produces exactly the data for it: recurring substitutions per learner, fed back
  into quiz and scenario generation so practice targets real weaknesses.
- Feedback **strictness** (soft/balanced/strict) is still not implemented.
- The i+1 metric needs lemmatisation and proper-noun handling to be trustworthy.

**Composer**
- Quality is now bounded by the model, not the plumbing. The obvious next step is
  a music-specialist model: **NotaGen** (516M, ABC-native, conditioned on period
  and composer) is small enough to sit alongside everything else, and
  ChatMusician is ABC-native too. That is the single biggest remaining lever.
- Multi-voice writing: the ABC path currently produces one melodic line with
  chord symbols. Real accompaniment needs `[V:1]`/`[V:2]` voices.
- `report["ok"]` describes the pre-repair ABC, so it can read false when the
  delivered MusicXML is correct. Tidy so status is not misleading.
- Playback in the browser (C4) is still the gate on iterating by ear.

---

## Session notes — 2026-08-09 (v2.4.0)

### What broke, and why (post-mortem)

A `uv lock --upgrade` moved the venv off a coherent dependency set and took the whole app
down. Root causes, in the order they bite:

| Symptom | Actual cause |
|---|---|
| App would not import at all | `crewai` 1.14.5 → 1.15.14 tripped an exact-equality `assert` in `patches.py` |
| Dialog mode ran to the 600 s timeout | CrewAI 1.15.14 changed Agent's **default executor** from `CrewAgentExecutor` to the Flow-based `crewai.experimental.AgentExecutor` |
| P3 context-length patch silently dead | Same executor change — the patched `_invoke_loop_*` methods still exist but are never called |
| Every image backend dead | `tokenizers` 0.23.1 vs `transformers` 5.8.1 (requires ≤ 0.23.0) |
| Whisper dead | `numpy` 2.5 vs `numba` (requires ≤ 2.4) |
| Voice dead *before* the upgrade | `speech.py` used `os.getenv` without `import os` (introduced in `c826f3e`) |

**Lesson for the next upgrade.** The structural guard in `patches.py` checks that patched
symbols *exist*. That is necessary but not sufficient — the executor swap kept every symbol
alive while removing them from the execution path. When upgrading CrewAI, check
**reachability** too: which executor `Agent` actually instantiates, and whether the patched
methods are on it. `patches._check_executor_reachability()` now logs this at startup.

### Deferred — pick these up next

- [ ] **TTS reads the first reply in the wrong language.** Observed in the tutor: the opening
      line `"Bonjour ! Ça va très bien, merci ! Et toi ? Je m'appelle [Assistant], je suis ton
      tuteur de français."` was spoken with an English voice, mangling the French. Every
      sentence *after* the first was detected correctly. Suspect the language sniff in
      `tts.py` runs before the target language is known, or `langdetect` is being fed too
      short a sample and losing to the bracketed `[Assistant]` placeholder. Fix: pass the
      session's target language explicitly to `/tts/speak` instead of sniffing, and fall back
      to detection only when no target language is set.
- [ ] **Port P3 to the new executor, then unpin.** Agents currently pin
      `executor_class=CrewAgentExecutor` via `patches.executor_kwargs()` because all five
      patches target it, and it is what the app was tested against. `CrewAgentExecutor` is
      deprecated upstream and will be removed. Port the context-length handling to
      `crewai.experimental.AgentExecutor`, verify with `CREWAI_EXECUTOR=experimental`, then
      drop the pin.
- [ ] **`tests/test_titi.py` hits a live Ollama model at collection time**, so it fails the
      whole run when `llama3.1:8b` is absent. All other suites are excluded from this problem.
      Mark it `@pytest.mark.integration` and move the model call inside the test body.
- [ ] Pytest exits 0 but aborts at interpreter teardown (`terminate called without an active
      exception`) once the app is imported. Cosmetic, pre-existing, not STT-related.

### Corrections to the older sections below

- **R2 (per-user concurrency) is DONE** — `main.py` holds per-user dicts for crews,
  semaphores and cancel events.
- **C3 (multi-instrument) is DONE** — `musicxml_builder.py` already emits a full `part-list`
  with per-part `score-part` and `midi-instrument`.
- **R1 (monkey-patch isolation) is DONE** — patches live in `patches.py` with tests.
- **R3 is still open and still real**: `/chat/cancel` sets an event that `_run_crew` never
  reads, so the worker thread runs to completion holding the GPU, and the early-exit `return`
  releases the semaphore while that thread is still alive.

---

## Where to spend effort next — a candid read (2026-08-31)

Raised by the user: *"Perhaps I am pulling in too many directions and should
focus on the researcher and the language tutor to make both better."*

**That instinct is right, and the evidence supports it.**

### Why the composer is the wrong place to keep pushing

Professional-quality composition is not a tuning problem, it is an open research
problem. The honest position after several sessions:

- The specialist model (ChatMusician) genuinely helped — 7 distinct rhythms
  against 2, 11/12 pitch classes against 7 — but it writes 8-16 good bars and
  then repeats or drifts. That is a property of the model, not of our prompt.
- `realize.py` supplies structure deterministically, which is why the output is
  coherent at all. But code can only *develop* material it is given; it cannot
  invent a good theme. Every improvement so far has been "enforce another rule",
  and each has diminishing returns.
- Assessment across sessions moved from "kindergarten" to "senior kindergarten".
  That is real progress and still a long way from useful.

**What would actually be needed:** a model trained on full scores with structural
supervision, not an 8-bar-window generator. Nothing in that class runs in 16 GB
today. The next honest step is C-O2 (better samples), which improves how it
*sounds* without touching how it *composes*.

### Where the same effort pays off

- **Researcher** — already the strongest component and used daily. R3 alone
  (cancel does not actually cancel; a "cancelled" run still holds the GPU) is a
  concrete defect with a known cause. R4/R5 (error format, structured logging)
  are ordinary engineering with predictable results.
- **Tutor** — FSRS, comprehensible input and phoneme scoring have landed, and
  each was measurable. What remains (per-learner error profile feeding quiz
  generation, curriculum-aware review) is *also* measurable, because language
  learning has ground truth: did the learner recall the word or not.

The difference that matters: in the researcher and tutor, "better" is defined
and testable. In the composer it is a matter of taste that a local model cannot
currently reach.

### Recommendation

Put the composer in maintenance — finish C-O2 for sound quality, keep it usable,
stop trying to make it a professional composition tool on this hardware. Spend
the recovered time on the researcher and tutor, where the same hours produce
something demonstrably better.

Revisit the composer when a score-level open model appears that fits in 16 GB.

---

## Composer rhythm — resolved 2026-08-24 (v2.11.0)

The long-form output had **3 distinct note lengths across 2.5 minutes** and
repeated its opening 8-note cell 5 times. Diagnosis: every transformation in
`develop_motif` changed PITCH and copied `quarterLength` verbatim —
`sequence`, `inversion`, `fragment`, `retrograde` — while `augmentation` scaled
everything uniformly, which is not variety either. The motif came back
transposed and inverted but always in the same rhythm, so the ear heard one idea
repeated rather than developed. Three of the eight sections were literally
`"original"`: the motif pasted back unchanged.

Fixed by adding rhythm as a development dimension of its own:

- `vary_rhythm()` — dotted, diminution, syncopate, even, long_short_pairs.
  Pitches and total length are preserved, so the idea stays recognisable and the
  phrase still fills its bars exactly.
- Combined transforms (`sequence_up+dotted`) so a returning section differs in
  both shape and timing.
- Only the opening statement is now `"original"`; every later return of A is
  re-timed.
- **Durations snap to a readable grid.** The arithmetic produced values like
  0.188 and 1.125 quarter-notes — legal MIDI, but they notate as knots of tied
  32nds. The final note absorbs the exact remainder so sections stay aligned to
  the beat.

Measured on the same 2.5-minute jazz piece:

| | before | after |
|---|---|---|
| distinct rhythms | 3 | **6**, all notatable |
| verbatim repeats (pitch AND rhythm) | 5 | **4** |
| same pitches, re-timed | — | **15** — development rather than repetition |

Honest limits: 4 verbatim returns of the theme remain, which is defensible for a
recurring subject but is not zero; and the B/C sections still develop the A
motif rather than introducing genuinely new material. Harmony still never leaves
the home key, and dynamics and articulation are still absent (C2).

---

## COMPOSER — orchestration and voices (roadmap, 2026-08-24)

*Answering "can we make this professional for instrument composing, and add
choir/voices?" Three separate problems that get conflated; only one is about
sound.*

**Where it stands:** audio renders through timidity + `FluidR3_GM.sf2` — plain
General MIDI. Adequate for auditioning, and the reason output sounds plastic
regardless of how good the notes are. No sampled instruments, no articulations,
and the composer does not know which instrument plays what it writes.

### C-O1 — Instrument ranges and transposition — ✅ DONE (v2.11.0)

The composer emits notes with no instrument awareness. Nothing stops it writing
below a violin's G3, and transposing instruments are written at concert pitch —
which is *wrong on the page* to any player, however good the MIDI sounds. This
is the single most unprofessional thing in current output.

music21 already carries the data, unused:

    Violin  range=G3..    Clarinet  transposition=M-2
    Viola   range=C3..    Trumpet   transposition=M-2
    Cello   range=C2..    Horn      transposition=P-5

- [x] `orchestration.py`: explicit ranges for 34 instruments and voices
- [x] Out-of-range notes move by **whole octaves**, never clamped — that
      preserves the pitch class and therefore the harmony. A test asserts pitch
      classes are unchanged.
- [x] Transposing instruments carry their transposition (clarinet/trumpet -2,
      horn -7, double bass/tenor -12), so the part is right **on the page**
- [x] Breathing check for winds, brass and voices; strings correctly exempt
- [x] Vocal ranges supplied — music21 returns `None..None` for all of them
- [x] **Bug found:** `realize.py` hardcoded `instrument.Piano()` for every part,
      so the UI's instrument picker was silently discarded. It now reaches the
      score, and out-of-range notes are fixed against the chosen instrument.
- [x] **Bug found:** attaching an instrument without removing the Piano
      placeholder left every part reading "Piano" while the report claimed
      otherwise — music21 honours the FIRST instrument it finds.
- [ ] Still open: string double stops, brass endurance beyond breath length

*Deliberately ahead of better samples: a beautifully sampled cello playing an
unplayable line is still amateur; a plain GM cello playing idiomatic writing is
not.*

### C-O2 — Sampled instruments instead of General MIDI (MEDIUM)

**Partly solved 2026-08-31, and the cause was not what this entry assumed.**
Playback sounded like a harpsichord regardless of orchestration because the
system `timidity.cfg` sources only `piano+guitar.cfg` and `drums.cfg` — a
minimal patch set with **no strings, brass or winds at all**. A violin part
(GM program 40) therefore fell back to the nearest available patch. The full
`fluidr3_gm.cfg`, backed by the 148 MB `FluidR3_GM.sf2` already installed, was
sitting unused. Passing `-c` fixed it: measured spectral centroid moved from
2771 Hz to 9461 Hz on the same violin/cello score. Override with
`TIMIDITY_CONFIG`.

So playback now uses real sampled instruments. A dedicated SFZ orchestra would
still be better, but the gap is far smaller than it was.

Remaining, for genuine realism rather than correctness:

- [ ] Install `sfizz` (neither sfizz nor fluidsynth is currently present;
      timidity is)
- [ ] Add a free SFZ library — Sonatina Symphonic Orchestra or Virtual Playing
      Orchestra (~400-500 MB)
- [ ] Route `/composer/convert-audio` through it, keeping timidity as fallback
- [ ] `AUDIO_BACKEND` / `SFZ_LIBRARY` env vars, same pattern as the image and
      video backends

### C-O3 — Choir and human voices (LOW/LARGE — three different technologies)

| Approach | Gives you | Cost | Keeps the score? |
|---|---|---|---|
| Sampled choir (SFZ "ahh"/"ooh") | Sustained vowels, no words | Low — same MIDI path | yes |
| Singing synthesis (DiffSinger, NNSVS, OpenUtau) | Real singing **with lyrics** | High — voicebank, GPU, phoneme timing | yes |
| Text-to-song (YuE, DiffRhythm, ACE-Step) | Finished audio from a prompt | Medium | **NO** |

- [ ] (a) Sampled choir as just another SFZ instrument, once C-O2 lands
- [ ] (b) Singing synthesis with lyrics — a genuine project; needs a lyric line
      in the score, phoneme alignment, and a voicebank

**Do NOT take the third row.** Those models make impressive audio but they do
not perform *your* score — they replace it. For a tool whose point is an
editable composition, that is the wrong product, however good the demos sound.

### Ordering, and why

**C-O1 → C-O2 → C-O3a → C-O3b.** Correctness before sound before voices.

But **all of it is behind the open rhythm problem** (3 distinct durations, motif
repeated 5x — see the v2.10.0 notes). Better samples applied to repetitive music
only make the repetition easier to hear.

---

## COMPOSER — why the music is weak, and what would actually fix it

*Revised 2026-08-09 after a literature scan. The headline finding below supersedes the
music21-first plan I originally wrote: the **representation** is the primary problem, and
fixing it is cheaper than anything else here.*

The composer is *functional* but musically naive. This is not a bug in
`musicxml_builder.py` (which is fine) and not a prompting problem that more instructions will
solve. It is structural, and there are two compounding causes.

### Cause 1 — we picked the worst possible format for an LLM (fix this first)

We ask the model to emit **note-by-note JSON**. Research on symbolic music generation is
blunt about this: LLMs are *inherently* more compatible with **ABC notation** than with
MIDI-style event lists, because ABC is text that aligns with how language models actually
work. Note-by-note JSON is arguably worse than MIDI — it is the most verbose option and
carries no musical structure at all.

The measured consequences of ABC:

- ~288 tokens per song, roughly **38% of the token count** of MIDI-style representations.
  Shorter sequences mean the model can hold a whole piece in context instead of losing the
  thread after four bars.
- ABC **intrinsically encodes repetition and structure** (repeats, sections, phrase marks).
  Structure is carried by the notation rather than something the model must remember.
- MuPT, trained on ABC, produced music that listeners **preferred over GPT-4**, with high
  intra-texture similarity and repetition scores.

This explains the "kindergarten" quality directly: a general model emitting one JSON note at a
time cannot hear bar 4 by the time it writes bar 12, so it produces locally plausible,
globally incoherent sequences — no motivic development, no phrase structure, no harmonic
direction.

**Action — DONE (v2.9.0):**
- [x] Composer output switched from JSON to **ABC notation**.
- [x] Parsed with `music21`, which reads ABC natively and converts to MusicXML.
- [x] `validate_abc()` measures bar lengths, longest directional run and stepwise
      proportion — faults music21 accepts silently.
- [x] Generate → validate → correct loop with best-of-N selection.
- [x] Deterministic re-barring, because the model never counted bar durations correctly
      in any attempt even when given the arithmetic.
- [ ] `musicxml_fix.py` (506 lines) is now dead code on the ABC path — delete once the
      old JSON endpoints are retired.

### Cause 2 — a general chat model has no musical training

Even in ABC, `mistral-small` has no idea what a good melody is. Two specialist models run
locally and both speak ABC natively:

| Model | Size | Notes |
|---|---|---|
| **ChatMusician** (`m-a-p/ChatMusician`) | LLaMA2-based | Continually pre-trained on ABC. Composes well-structured, full-length music conditioned on **text, chords, melodies, motifs and musical forms**. Pure text tokenizer — no multimodal stack, so it can run under Ollama/llama.cpp like any other model. |
| **NotaGen** (`ElectricAlexis/NotaGen`) | **516M params** | Pre-trained on 1.6M ABC sheets. Conditioned on **period, composer and instrumentation**. Tiny — fits alongside Ollama and the image pipelines on the 16 GB card with room to spare. |

Note the symmetry with what we just did for OCR: a small specialist model beat the big general
one decisively. `deepseek-ocr:3b` was 5× faster and produced 4× more text than `qwen3-vl:8b`.
The same pattern very likely holds here.

**Action:**
- [ ] Add `COMPOSER_MODEL`, independent of `MODEL`, exactly as `OCR_MODEL` is today.
- [ ] Evaluate NotaGen first — 516M is cheap to try and it is purpose-built for notation.

### Still worth doing, after the above

1. **`music21`** for what code does better than any model: voice-leading and range validation,
   realising Roman-numeral progressions into voiced chords, and motivic development
   (transposition, inversion, retrograde, augmentation) — one-liners that are exactly what
   makes music sound *composed* rather than sampled.
2. **Two-stage generation.** LLM returns a plan (form, key plan, a 2-bar motif, harmonic
   rhythm); code expands it. The model never emits a long note list.
3. **In-browser playback (OSMD + Tone.js).** Still the gate on everything: until a change can
   be heard in two seconds instead of a MuseScore round-trip, tuning musical quality is
   impractical.
4. Missing MusicXML expression: dynamics, articulations, slurs, and mid-score key changes
   (`key_changes`) — see C1/C2 below.

**Sources:** [MuPT](https://arxiv.org/abs/2404.06393) ·
[ChatMusician](https://huggingface.co/papers/2402.16153) ·
[NotaGen](https://www.ijcai.org/proceedings/2025/1134.pdf) ·
[MMM](https://arxiv.org/pdf/2008.06048)

---

## TUTOR — competitive scan and what to steal

*Added 2026-08-09. The tutor is feature-rich but built almost entirely around **output**
(quizzes, vocabulary drills, scenarios). The research consensus is that this is the smaller
half of language acquisition.*

### T-A — Replace SM-2 with FSRS — ✅ DONE (v2.6.0-dev)

`tutor/storage.py` implements SM-2, the 1987 algorithm with a single ease factor per card.
Anki replaced it with **FSRS**, which models three variables per card — difficulty, stability
and retrievability — fitted to **700M+ real reviews**.

- **~20–30% fewer reviews for the same retention.**
- More accurate recall prediction in **~99.6%** of collections.
- Free, open-source, available as a Python library from the Open Spaced Repetition project.

This is the highest value-per-effort item in the tutor: a drop-in scheduler swap behind the
existing `vocabulary` table. The `lapses` column added in T6 is already one of the inputs FSRS
wants.

- [x] Added `fsrs` 6.3.2 (MIT); `vocabulary` gained `fsrs_stability`, `fsrs_difficulty`,
      `fsrs_state`, `fsrs_step`, `last_review`
- [x] SM-2 kept behind `SRS_ALGO=sm2`; `schedule_review()` dispatches
- [x] Existing cards are seeded from their SM-2 history rather than reset —
      `stability ← interval_days`, `difficulty ←` inverted ease factor, nudged up by lapses
- [x] Added `SRS_MAX_INTERVAL` (default 365). FSRS's own default is 36500 days, which is
      meaningless for language study.
- [x] 9 tests in `tests/test_tutor.py::TestFsrsScheduler`

**Bug found and fixed during implementation** — worth remembering, because it is silent:
seeding a migrated card with `last_review = NULL` makes FSRS compute zero elapsed time, so a
long-interval card looks perfectly retained. A 90-day card rated *Easy* was scheduled
**4994 days** out. Setting `last_review` to one interval ago (which is simply true) gives 447
days. Pinned by `test_migration_does_not_explode_the_interval`.

*Not yet verifiable:* the ~20–30% review reduction is a property of scheduling accuracy over
weeks of real reviews. The tests prove correctness and continuity, not that benefit.

### T-B — Add comprehensible input — ✅ DONE (v2.6.0)

Current design is ~100% output and review. The research points the other way:

- Learners spending **80% of study time on comprehensible input** (real content at their level
  plus ~10% unknown words — Krashen's *i+1*) and 20% on active review progress significantly
  faster.
- Authentic content + SRS gives **3.2× higher retention** than drill-based apps, and reaches
  intermediate comprehension **40% faster**.
- App study combined with authentic media improved **listening comprehension 47% faster** than
  apps alone.

We already have every piece needed to build this and have not connected them:

- [x] **Levelled reading passages** — new 📚 Reading tab. `passage_task` in `tasks.yaml`
      explicitly forbids grammar tables and exercises: this is prose to be read, not a lesson.
      Defaults to the session's current curriculum topic when none is given.
- [x] **Seeded with due vocabulary** — `list_vocabulary_due()` feeds the prompt, which
      instructs the model to use each word naturally and *not* draw attention to it.
- [x] **Read aloud** via the existing edge-tts. `_passageProseOnly()` strips the glossary and
      comprehension questions first, so only the prose is spoken.
- [x] **Unknown-word ratio** computed server-side per passage, with an i+1 verdict in the UI.
- [x] Endpoints: `POST /tutor/passages` (SSE), `/tutor/passages/save`, list / get / delete;
      new `passages` table.

**Two corrections made during implementation — both worth remembering:**

1. *The first ratio was useless.* It reported **95.7% unknown** on a clean A1 passage because
   `le`, `la`, `il`, `est`, `de` counted as unknown vocabulary. Function-word stop lists for
   French, Spanish, German, Italian, Portuguese, English and Romanian now exclude them.
2. *The ratio is measured against **saved** vocabulary*, which is only a proxy for what the
   learner knows. Below 50 saved words the UI now says "save more vocabulary to gauge
   difficulty (n/50)" rather than always declaring a passage "hard". Expect the metric to look
   unhelpful on day one and to become meaningful as the word list grows.

**Known limitations:** proper nouns count as new words (*Léo*), and there is no lemmatisation,
so *dort* and *dormir* are counted separately. Both would need a per-language NLP dependency
for modest gain.

### T-C — Tighten the conversation feedback loop (MEDIUM)

Scenario roleplay (T1) is the right shape; the market leaders differ in the *feedback*, not
the roleplay:

- Praktika's core loop is explicitly **speak → get feedback → improve**, with user-selectable
  strictness: **soft / balanced / strict**.
- Leaders **diagnose pronunciation at phoneme level** and adapt to the learner's actual
  errors rather than a preset script.
- Persistent **contextual memory** across sessions, so the tutor remembers earlier mistakes.

Against that, ours corrects errors inline but has no strictness control, no error history and
only word-level scoring:

- [ ] Add a **feedback strictness** setting to scenario and chat modes.
- [ ] Upgrade pronunciation scoring from WER to **phoneme-level**. Current `_wer()` scores
      *word recognition*, so a heavy accent Whisper still decodes scores 1.0 — see T3 note
      above. Phoneme alignment gives genuinely actionable feedback ("your *u* is fronted").
- [ ] Persist a **per-learner error profile** (recurring grammar and phoneme errors) and feed
      it into scenario and quiz generation, so practice targets real weak spots.

### T-D — Deferred bug (carried from the v2.4.0 notes)

- [ ] TTS speaks the **first** reply of a session in the wrong language — see the deferred
      item in the session notes above.

**Sources:** [FSRS vs SM-2](https://www.antiagent.io/blog/fsrs-vs-sm-2) ·
[Anki scheduler FAQ](https://faqs.ankiweb.net/what-spaced-repetition-algorithm) ·
[Research-backed app comparison](https://www.sciencebasedlearning.com/blog/best-language-learning-apps-2026) ·
[Praktika review](https://languatalk.com/blog/praktika-review/) ·
[AI speaking apps compared](https://www.talkio.ai/blog/best-ai-language-speaking-practice-apps-in-2026)

---

---

## Executive Summary

This is a well-built, feature-rich local AI platform. The architecture is sound and the separation between researcher, tutor, and composer is clean. The main risks are:

1. **~300 lines of monkey-patches** in `crew.py` that will silently break on any CrewAI/litellm update
2. **Single-user concurrency** — a semaphore serializes all crew runs
3. **Fragmented error handling** — no consistent format across the three agents
4. **Tutor and composer UIs** are incomplete relative to the backend capabilities

The sections below are ordered by priority within each component.

---

## RESEARCHER

### R1 — Monkey-Patch Isolation (HIGH) — ✅ DONE (v2.4.0)

**Actions**:
- [x] Move all monkey-patches into `src/researcher/patches.py`
- [x] Add version guards — implemented as a *range* plus a structural check, not an
      exact-equality assert. The original suggestion here was actively harmful: an exact
      `==` assert turned a routine `uv lock --upgrade` into a hard startup crash. See the
      v2.4.0 session notes.
- [x] Comment block per patch group explaining what it fixes
- [x] `tests/test_patches.py` verifies each patch is still necessary
- [x] **Beyond the original plan**: `_check_executor_reachability()` also catches patches
      that are still *present* but no longer *called* — which is exactly how P3 died silently.

### R2 — Multi-User Crew Isolation (HIGH)
**Files**: `src/researcher/main.py` line 65, `src/researcher/routes/chat.py` lines 155, 173, 208

Authentication is correctly implemented — each request is tied to an authenticated `user_id`. However, the *crew execution* state is stored as a single global on `app.state`:
- `app.state.crew_semaphore` — one Semaphore(1) blocks **all authenticated users**, not just the same user
- `app.state.research_crew` — a single crew instance; user B's request at line 155 overwrites user A's in-flight crew
- `app.state.cancel_event` — a single cancel flag; any user hitting `/chat/cancel` cancels every other user's active run

**Status**: ✅ mostly DONE — one sub-item outstanding.

**Actions**:
- [x] `app.state.research_crews` is a `dict[user_id, ResearchCrew]`
- [x] `app.state.crew_semaphores` is a `dict[user_id, asyncio.Semaphore]`
- [x] `app.state.cancel_events` is a `dict[user_id, threading.Event]`
- [x] Tutor and composer use the same per-user semaphores (see X3)
- [ ] Return HTTP 409 Conflict if the same user already has an active crew run —
      currently the second request simply blocks on the semaphore instead of being rejected

### R3 — Request Timeouts & Circuit Breaker (HIGH)
**Files**: `src/researcher/routes/chat.py`, `src/researcher/routes/ask.py`

Long-running crew kickoffs can hang indefinitely. There is no timeout, no cancellation path surfaced to the client on errors, and no circuit breaker.

**Status**: ⚠️ PARTIAL — the timeout fires, but nothing is actually cancelled.

**Actions**:
- [x] Timeout implemented, configurable via `CREW_TIMEOUT_SEC` and the per-user
      `crew_timeout_sec` LLM param
- [x] On timeout the stream yields an `error` SSE event and stops relaying
- [ ] **Cancel does not cancel.** `/chat/cancel` sets the event and the SSE loop checks it,
      but `_run_crew` never reads it — the worker thread runs to completion, still holding
      the GPU. The user sees the request stop; the machine does not.
- [ ] **The semaphore is released early.** The early-exit `return` leaves the
      `async with semaphore` block while the worker thread is still alive, so a second
      request can start a concurrent crew on a single GPU.
- [ ] Verify both with a test

*Practical note:* `max_iter` (added v2.4.0, tunable from the params panel) is the real
lever on worst-case latency — each iteration is a full LLM call.

### R4 — Consistent Error Responses (MEDIUM)
**Files**: `src/researcher/routes/chat.py`, `routes/ask.py`, `routes/sessions.py`, `routes/models.py`

Some routes return `{"detail": "..."}` (FastAPI default), others return `{"error": "..."}`, others raise bare exceptions.

**Actions**:
- [ ] Define a single `ErrorResponse` Pydantic model: `{"error": str, "code": str, "detail": str | None}`
- [ ] Add a global FastAPI exception handler in `main.py` that converts all unhandled exceptions to this format
- [ ] Update all `HTTPException` raises to use consistent `code` strings (e.g., `"not_found"`, `"rate_limited"`, `"crew_error"`)

### R5 — Structured Logging (MEDIUM)
**Files**: All modules (currently mix of `print()`, `logging.info()`, `logger.debug()`)

**Actions**:
- [ ] Add `structlog` or configure stdlib logging with a JSON formatter in `main.py`
- [ ] Define standard fields: `user_id`, `session_id`, `model`, `duration_ms`, `tool_calls`
- [ ] Replace all bare `print()` calls with proper logger calls
- [ ] Add request/response logging middleware (log method, path, status, duration)

### R6 — VRAM Spike After Image Generation (MEDIUM)
**Files**: `src/researcher/config.py` line 201

After image generation, Ollama is unloaded from VRAM to free space. This causes a ~10s reload penalty on the next message.

**Actions**:
- [ ] Add `KEEP_LLM_LOADED=true` env var option (for users with enough VRAM, e.g., 24GB+)
- [ ] If `KEEP_LLM_LOADED=false` (default), pre-warm Ollama in a background task after image gen completes so the reload happens while the user reads the image, not while they wait for the next response
- [ ] Add a `/memory` endpoint that reports current Ollama VRAM usage

### R7 — Input Sanitization (MEDIUM)
**Files**: `src/researcher/routes/chat.py`, `routes/ask.py`

User prompts are passed directly to the LLM and injected into the system prompt context without escaping.

**Status**: ⚠️ PARTIAL.

**Actions**:
- [ ] Strip leading/trailing control characters and zero-width spaces from user input
- [ ] Enforce a max prompt length with a clear error message
- [x] Attachment text is capped — `MAX_TEXT_CHARS = 50_000` in `ingestion.py`, applied at
      extraction time with a visible truncation marker. Still a character budget rather than
      a token budget, and 50k chars is generous against a 32k-token context.

### R8 — Static File Caching (LOW)
**Files**: `src/researcher/main.py` (StaticFiles mount)

No cache headers, no ETags, no compression. Browser re-downloads all assets on every page load.

**Actions**:
- [ ] Add `Cache-Control: max-age=3600` headers for static assets via a custom `StaticFiles` subclass or middleware
- [ ] Enable gzip compression for HTML/CSS/JS responses (FastAPI middleware or nginx)

---

## TUTOR

### T1 — Situational Dialogs / Roleplay Scenarios (HIGH)
**Files**: `src/researcher/tutor/crew.py`, `src/researcher/static/tutor.html`

- [x] Add `scenario` mode to `TutorCrew` task builder (in `tutor/crew.py`)
- [x] Create `tutor/config/scenarios.yaml` with starter situations
- [x] Add `/tutor/scenarios` endpoint (list) and `/tutor/chat` scenario mode
- [x] Add scenario picker to `tutor.html` UI (🎭 Scenarios tab)
- [x] Reasoning now collapsible instead of overlapping the reply

**Remaining — Scenario quality improvements**:
- [x] Tutor must conduct the entire dialog in the target language — strengthened prompt with NON-NEGOTIABLE language rule and first-person framing
- [x] Make scenarios more pedagogically driven — PEDAGOGY RULES added: always end with a question, correct errors briefly, prompt if student is silent
- [x] Save scenario transcript to `lesson_plans` table — `save_transcript: true` on each real turn saves via `on_complete` callback
- [x] Opening line auto-generated in target language — replaced static English `starter` with LLM-generated first turn (`__start__` trigger)
- [x] Fixed `liveBox` reference bug in `tutor-scenarios.js`

### T2 — Progress Visualization (HIGH) — DONE
**Files**: `src/researcher/routes/tutor.py` (`/tutor/sessions/{id}/stats`), `src/researcher/static/tutor.html`

- [x] Chart.js 4.4 loaded in `tutor.html`
- [x] Stats tab with mastery histogram and quiz score trend charts (`tutor-stats.js`)
- [x] Stats grid shows vocab count, avg mastery, quiz count, avg score

### T3 — Lesson Export (MEDIUM) — DONE
**Files**: `src/researcher/routes/tutor.py`, `src/researcher/static/js/tutor-lessons.js`

- [x] `GET /tutor/lessons/{id}/export?format=md|docx` endpoint
- [x] Markdown export — proper `.md` file from stored content
- [x] DOCX export — line-by-line markdown parsing into python-docx
- [x] Download buttons use `fetch` + blob (auth headers, no JSON download bug)

### T4 — Pronunciation Scoring (MEDIUM) — DONE
**Files**: `src/researcher/routes/speech.py`, `src/researcher/static/js/tutor-quiz.js`

- [x] `pronunciation` quiz type implemented end-to-end
- [x] WER utility in `speech.py` `_pronunciation_score()`
- [x] `/tutor/pronunciation/score` endpoint (Whisper transcribe + WER)
- [x] Quiz UI renders pronunciation questions with Play + Record buttons
- [x] Pre-computed scores bypass LLM grader in quiz submit

### T5 — CEFR Curriculum Structure (MEDIUM) — DONE
**Files**: `src/researcher/tutor/storage.py`, `src/researcher/static/tutor.html`

- [x] `curriculum.yaml` with per-CEFR-level topic sequences
- [x] `topic_index` column in `tutor_sessions`
- [x] `/tutor/curriculum-progress` and `/tutor/sessions/{id}/advance-topic` endpoints
- [x] Curriculum bar in UI: clickable topic name pre-fills lesson generator, Prev/Next navigation
- [x] `/tutor/sessions/{id}/prev-topic` endpoint added

### T6 — SM-2 Edge Cases (LOW)
**Files**: `src/researcher/tutor/storage.py`, `src/researcher/static/js/tutor-vocab.js`

- [x] `SRS_INITIAL_EASE` env var (default 2.5)
- [x] Lapse handling: grade < 3 resets interval to 1, decrements ease (floor 1.3)
- [x] `lapses` column in `vocabulary` table (with migration)
- [x] Words with `lapses >= 3` flagged as "difficult" (⚠ hard badge + red row tint) in vocab UI

---

## COMPOSER

### C1 — Modulation & Key Change Support (HIGH)
**Files**: `src/researcher/composer/musicxml_builder.py`, `src/researcher/composer/knowledge/theory.yaml`

The MusicXML builder handles a single key signature. Real compositions frequently modulate (change key mid-piece). The theory.yaml has no modulation tables.

**Actions**:
- [ ] Add `key_changes: list[{measure: int, fifths: int, mode: str}]` to the composition JSON schema
- [ ] Update `musicxml_builder.py` to emit `<key>` elements mid-score at the correct measure numbers
- [ ] Add modulation section to `theory.yaml`: common modulations per key (relative major/minor, dominant, subdominant), pivot chords
- [ ] Update `agents.yaml` composer rules: "You may modulate by specifying key_changes in the JSON"
- [ ] Add test cases in `test_musicxml_builder.py` for mid-score key changes

### C2 — Dynamics & Articulation in MusicXML (HIGH)
**Files**: `src/researcher/composer/musicxml_builder.py`

The builder has partial dynamics support (mentioned in file comments) but articulation marks (staccato, legato slurs, accents, fermata) are not emitted.

**Actions**:
- [ ] Add `articulation` field to note JSON schema: `"staccato" | "accent" | "tenuto" | "fermata" | null`
- [ ] Add `slur_start` / `slur_end` boolean fields to note JSON (for legato phrasing)
- [ ] Emit `<articulations>` and `<slur>` elements in `musicxml_builder.py`
- [ ] Update `agents.yaml` composer rules with articulation vocabulary
- [ ] Test: verify MuseScore can import and display articulations correctly

### C3 — Multi-Instrument / Orchestration Support (HIGH)
**Files**: `src/researcher/composer/musicxml_builder.py`, `src/researcher/composer/tools.py`

Currently the builder supports piano (grand staff) and single-instrument scores. Orchestration for multiple instruments is not implemented.

**Status**: ✅ largely DONE — this entry was stale.

**Actions**:
- [x] JSON schema has a `parts` array
- [x] `musicxml_builder.py` emits `<part-list>` with per-part `<score-part>` and
      `<midi-instrument>`; also handles grand staff vs single staff and merges split parts
- [ ] Link the `instrument_range` tool to part validation (music21 would do this better)
- [ ] Orchestration templates in `genres.yaml` (string quartet, jazz combo, brass quintet)
- [x] `agents.yaml` documents the parts array

### C4 — Score Playback in Browser (MEDIUM)
**Files**: `src/researcher/static/composer.html`

MusicXML is downloadable but can only be played in external apps (MuseScore, etc.). In-browser playback would close the feedback loop dramatically.

**Actions**:
- [ ] Integrate [OSMD (OpenSheetMusicDisplay)](https://github.com/opensheetmusicdisplay/opensheetmusicdisplay) for browser MusicXML rendering
- [ ] Integrate [midi.js](https://github.com/mudcube/MIDI.js) or Web MIDI API for playback
- [ ] Add "View Score" button that opens rendered notation in a modal
- [ ] Add "Play" button that triggers MIDI playback in browser
- [ ] Keep "Download MusicXML" button for users who want MuseScore

### C5 — Robust JSON Repair Hardening (MEDIUM)
**Files**: `src/researcher/composer/musicxml_fix.py`

There are 19 regex + 8 structural repairs. This file is a patch on top of a patch — the LLM produces broken JSON and it gets repaired downstream. The correct fix is better prompting.

**Status**: ⚠️ PARTIAL — and **likely to be deleted rather than finished**. Switching the
composer's output format to ABC notation removes the reason this layer exists. Do not invest
here before reading the revised COMPOSER section near the top of this file.

**Actions**:
- [ ] Few-shot JSON examples in `agents.yaml`
- [x] Valid JSON already short-circuits — `_extract_json_score` parses as-is first and only
      falls back to repairs on `JSONDecodeError`
- [ ] Log which repairs fire (would identify the most common LLM failures)
- [x] Repair failure returns `None` and logs, rather than crashing
- [x] `tests/test_musicxml_fix.py` exists (419 lines)

### C6 — Harmonization Feedback UI (MEDIUM)
**Files**: `src/researcher/static/composer.html`, `src/researcher/routes/composer.py` (`/composer/harmonize`)

The harmonize endpoint exists and works, but the UI has no way to visually compare the original melody with the harmonized version side by side.

**Actions**:
- [ ] After harmonize response, render both original and harmonized MusicXML snippets in the UI (using OSMD from C4, or as text diff)
- [ ] Highlight changed notes / added chords in a different color
- [ ] Allow user to accept or reject individual harmonic suggestions

### C7 — Ear Training Mode (LOW — v3.0 prep)
**Files**: `src/researcher/composer/crew.py`, `src/researcher/static/composer.html`

The TODO mentions "Music Tutor" as v3.0. Ear training is the natural bridge between composer and tutor.

**Actions**:
- [ ] Add `ear_training` mode to `ComposerCrew`: generate interval/chord identification exercises
- [ ] Use Web Audio API (already present for mic) to play generated tones in browser
- [ ] Store results in `quiz_results` table (reuse tutor quiz infrastructure)
- [ ] This is a research spike — do a 2-day prototype before committing to full implementation

---

## CROSS-CUTTING

### X1 — Shared LLM Initialization Utility (MEDIUM)
**Files**: `crew.py`, `tutor/crew.py`, `composer/crew.py`

All three crew classes repeat the same LLM initialization, capability probing, and planning LLM creation code (~80 lines each).

**Status**: ❌ NOT STARTED — still three near-identical copies. The duplication has since
grown: `executor_kwargs()` and the `max_iter` param had to be added to all three by hand.

**Actions**:
- [ ] Create `src/researcher/llm_factory.py` with `make_llm(model, params) -> LLM` and `make_planning_llm(model) -> LLM`
- [ ] Replace the three duplicated blocks with calls to this factory
- [ ] Move `probe_model_capabilities()` from `config.py` into `llm_factory.py`

### X2 — Central Configuration (MEDIUM)
**Files**: Many (magic numbers scattered throughout)

Context window (32768), predict tokens (4096), image size (512), SM-2 ease (2.5), rate limit window (300s) are all hardcoded.

**Actions**:
- [ ] Add a `src/researcher/settings.py` using Pydantic `BaseSettings`
- [ ] Load all constants from env vars with documented defaults
- [ ] Replace all magic numbers with `settings.X` references
- [ ] Update `.env.example` with all new variables and descriptions

### X3 — Per-User Concurrency Model (MEDIUM)
**Files**: `main.py` (researcher semaphore), tutor routes, composer routes

Tutor and composer have no concurrency protection at all — two tabs can corrupt state. Researcher has a global semaphore.

**Status**: ✅ mostly DONE — this entry was stale. Tutor and composer *do* acquire the
per-user semaphore (`routes/tutor.py:125`, `routes/composer.py:105` with
`async with semaphore`). An earlier audit of mine claimed otherwise; that was wrong.

**Actions**:
- [x] Per-user semaphores live on `app.state.crew_semaphores`, shared by all three components
- [x] Applied to researcher, tutor and composer kickoffs
- [ ] Return 409 Conflict instead of silently queueing behind the semaphore (same open
      sub-item as R2)

### X4 — Database Migrations (LOW)
**Files**: `main.py` (SQLite init), `tutor/storage.py`, `composer/storage.py`

Tables are created with `CREATE TABLE IF NOT EXISTS`. If a column is added to an existing install, the old DB silently has the old schema.

**Actions**:
- [ ] Add [yoyo-migrations](https://ollycope.com/software/yoyo/latest/) or hand-rolled migration runner
- [ ] Write `migrations/001_initial.sql` (extract current CREATE TABLE statements)
- [ ] Any new column addition gets a `migrations/00N_add_X.sql`
- [ ] Run migrations on server startup in `main.py` lifespan

### X5 — Documentation (LOW)
**Files**: `README.md`, `docs/`

The README is good but there is no architecture diagram and no per-component API reference.

**Actions**:
- [ ] Add a Mermaid architecture diagram to README (shows researcher/tutor/composer → CrewAI → Ollama → tools)
- [ ] Auto-generate API docs are available at `/docs` (FastAPI) — add a note in README
- [ ] Add docstrings to the three `crew.py` classes and `musicxml_builder.py` (these are the hardest files to understand cold)

---

## Priority Order (Suggested Sequence)

> **Superseded — kept for reference.** This ordering is from 2026-05-23 and several entries
> are complete. Use the "Status at a glance" table at the top of this file for current state,
> and the recommended order below for what to do next.

### Recommended order now (2026-08-10)

| # | Item | Why now |
|---|------|---------|
| 1 | **C4 — Browser score playback** | Gates all composer work. Until a change can be heard in seconds rather than via MuseScore, musical quality cannot be tuned. |
| 2 | **Composer: ABC notation switch** | Cheapest large win available; also deletes most of C5. See the revised COMPOSER section. |
| 3 | **T-C — Feedback loop, phoneme scoring** | The remaining tutor gap now that T-A and T-B have landed. |
| 4 | **R3 — Make cancel actually cancel** | A "cancelled" run still holds the GPU on a single-GPU machine. |
| 5 | **X1 + X2 — LLM factory, central settings** | Duplication keeps growing; every new param must be added in three places by hand. |
| 6 | T-D — TTS first-reply language bug | Small, user-visible. |
| 7 | C1/C2 — Dynamics, articulation, key changes | Worth doing once you can *hear* the difference (after 1). |
| 8 | Tutor: lemmatisation / proper-noun handling | Would sharpen the i+1 metric; needs a per-language NLP dependency. |

### Original 2026-05-23 ordering (historical)

| # | Item | Component | Effort | Impact |
|---|------|-----------|--------|--------|
| 1 | R2 — Per-user concurrency | Researcher | M | HIGH |
| 2 | R3 — Request timeouts | Researcher | S | HIGH |
| 3 | R1 — Monkey-patch isolation | Researcher | M | HIGH (maintenance) |
| 4 | X1 — Shared LLM factory | All | S | MEDIUM |
| 5 | X2 — Central config/settings | All | S | MEDIUM |
| 6 | T1 — Situational dialogs | Tutor | L | HIGH (UX) |
| 7 | T2 — Progress visualization | Tutor | M | HIGH (UX) |
| 8 | C1 — Modulation support | Composer | M | HIGH (quality) |
| 9 | C2 — Dynamics/articulation | Composer | S | HIGH (quality) |
| 10 | C3 — Multi-instrument | Composer | L | HIGH (feature) |
| 11 | R4 — Consistent error responses | Researcher | S | MEDIUM |
| 12 | T3 — Lesson export | Tutor | S | MEDIUM |
| 13 | C4 — Browser score playback | Composer | L | HIGH (UX) |
| 14 | T4 — Pronunciation scoring | Tutor | M | MEDIUM |
| 15 | T5 — CEFR curriculum | Tutor | M | MEDIUM |
| 16 | C5 — JSON repair hardening | Composer | S | MEDIUM |
| 17 | R5 — Structured logging | Researcher | S | LOW-MEDIUM |
| 18 | R6 — VRAM spike fix | Researcher | S | MEDIUM |
| 19 | X3 — Cross-component locking | All | S | MEDIUM |
| 20 | X4 — DB migrations | All | M | LOW (maintenance) |
| 21 | C6 — Harmonization feedback UI | Composer | M | LOW-MEDIUM |
| 22 | T6 — SM-2 edge cases | Tutor | S | LOW |
| 23 | C7 — Ear training (spike) | Composer | L | FUTURE |
| 24 | R7 — Input sanitization | Researcher | S | LOW (single-user) |
| 25 | R8 — Static file caching | Researcher | S | LOW |
| 26 | X5 — Documentation | All | M | LOW |

**Effort key**: S = < 1 day, M = 1–3 days, L = 3–7 days
