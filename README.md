# AI-Based Cybersecurity Analyst Assistant

A student-level **defensive** cybersecurity application built for a
5th-semester engineering AI (CIA) project. It simulates how a SOC
(Security Operations Center) analyst assistant would take in raw
security logs and help a human analyst triage them faster — by
detecting anomalies, applying rule-based knowledge, classifying
incidents, estimating confidence, retrieving relevant defensive
reference material, generating a plain-English AI summary, proposing a
response plan, and persisting the human analyst's final decision as
feedback. **All seven originally-planned phases are complete, plus an
optional Phase 8 that adds lightweight local retrieval (RAG) to the
Generative AI step.**

> **Scope note:** This project is defensive only. It analyses, detects,
> classifies, and explains suspicious activity from logs. It does **not**
> perform any offensive hacking, exploitation, malware creation, or
> attack execution of any kind.

---

## 1. Project Pipeline

```
Cybersecurity Logs
        ↓
Data Preprocessing
        ↓
AI Anomaly Detection
        ↓
Knowledge / Rule Engine
        ↓
Incident Classification
        ↓
Uncertainty / Confidence
        ↓
RAG Knowledge Retrieval
        ↓
Generative AI Summary
        ↓
Response Planning
        ↓
Human Analyst Review
        ↓
Approve / Reject / Modify
        ↓
Feedback Storage
        ↓
Feedback Insights
        ↓
Continuous Improvement Insights
        ↓
Explainability & Audit Trail
        ↓
Evaluation & Performance Dashboard
```

The project was built **phase by phase**. Each phase was completed and
tested before the next one started. Phases 0-7 are the originally
planned pipeline; Phases 8-11 are later, optional extensions - Phase 8
fills in the RAG hook Phase 5 always left in place for future work,
Phase 9 turns the feedback Phase 7 already collects into offline,
human-reviewed improvement suggestions, Phase 10 adds transparent
explainability and a persistent audit trail over everything Phases 2-9
already decided, and Phase 11 adds a descriptive, measured evaluation
and performance dashboard over everything Phases 2-10 already produced -
none of these later phases changes any decision made by an earlier one.
Phase 12 is a presentation-only pass over the same 11 pages - it
reorganizes, labels, and explains what is already there; it adds no new
pipeline stage and changes no detection/classification/confidence/RAG/
GenAI/response-planning/feedback/continuous-improvement/explainability/
audit/evaluation logic at all. Phase 13 is the final phase: a whole-
project integration audit, a real end-to-end run of the operational
pipeline, a full current-suite test pass, and final packaging - it adds
no pipeline stage, no page, and no algorithm of its own.

| Phase | Description | Status |
|-------|-------------|--------|
| 0 | Project planning, setup, folder structure, synthetic dataset | ✅ Done |
| 1 | Data preprocessing & feature engineering | ✅ Done |
| 2 | AI anomaly detection (Isolation Forest / SOM) | ✅ Done |
| 3 | Knowledge / rule-based engine + incident classification | ✅ Done |
| 4 | Uncertainty & confidence scoring | ✅ Done |
| 5 | Generative AI summary (Ollama, evidence-based fallback) | ✅ Done |
| 6 | Response planning | ✅ Done |
| 7 | Human analyst review + feedback loop | ✅ Done (final planned phase) |
| 8 | RAG / knowledge base integration (optional extension) | ✅ Done |
| 9 | Continuous improvement from analyst feedback (optional extension) | ✅ Done |
| 10 | Explainability & audit trail (optional extension) | ✅ Done |
| 11 | Evaluation & performance dashboard (optional extension) | ✅ Done |
| 12 | Final UI/UX & analyst workflow polish (optional extension) | ✅ Done |
| 13 | Final integration, QA, packaging & demo readiness | ✅ Done (final phase) |

---

## 2. Tech Stack

- **Python** — core language
- **Streamlit** — dashboard / UI
- **Pandas / NumPy** — data handling
- **Scikit-learn** — anomaly detection & classification models, and (Phase
  8) `TfidfVectorizer` + cosine similarity for lightweight local RAG
  retrieval - no separate embeddings model or vector database
- **MiniSom** *(optional)* — Self-Organizing Map for anomaly clustering
- **Rule-based knowledge engine** — hand-written defensive detection rules
- **Ollama** — local Generative AI for analyst-friendly summaries
- **SQLite** (`sqlite3`, built into Python) — persistent storage for
  analyst feedback (Phase 7) and, in a separate database file, the
  Phase 10 audit trail - no external database server needed
- **Plotly** — the bar/heatmap-style charts on the Evaluation &
  Performance dashboard (Phase 11), reusing the same `plotly.express` +
  `_style_chart()` styling already used since Phase 2 - no new charting
  dependency was introduced
- **CSV files** — log storage (no external database needed for this
  academic project)

---

## 3. Folder Structure

```
ai_cyber_analyst/
├── app.py                      # Streamlit entry point (still 11 pages: Overview + Data
│                                # Preprocessing + AI Anomaly Detection + Incident
│                                # Classification + Uncertainty & Confidence +
│                                # Generative AI Analysis + Response Planning +
│                                # Analyst Review + Feedback History + Explainability
│                                # & Audit Trail + Evaluation & Performance - Phase 12
│                                # (see §19) polished navigation/presentation on top of
│                                # this same page list, it did not add a page)
├── config.py                    # Central configuration (paths, constants, thresholds)
├── generate_data.py             # One-time script that creates the synthetic dataset
├── test_preprocessing.py        # Standalone validation script for Phase 1 (see §8)
├── test_anomaly_detection.py    # Standalone validation script for Phase 2 (see §9)
├── test_rule_engine.py          # Standalone validation script for Phase 3 (see §10);
│                                 # also re-runs the Phase 1 and Phase 2 suites
├── test_confidence.py           # Standalone validation script for Phase 4 (see §11);
│                                 # also re-runs the Phase 1, 2, and 3 suites
├── test_genai.py                # Standalone validation script for Phase 5 (see §12);
│                                 # also re-runs the Phase 1, 2, 3, and 4 suites
├── test_response_planner.py     # Standalone validation script for Phase 6 (see §13);
│                                 # also re-runs the Phase 1, 2, 3, 4, and 5 suites
├── test_feedback.py             # Standalone validation script for Phase 7 (see §14);
│                                 # also re-runs the Phase 1-6 suites
├── test_retriever.py            # Standalone validation script for Phase 8 (see §15);
│                                 # also re-runs the Phase 1-7 suites (the FULL
│                                 # regression through Phase 8)
├── test_continuous_improvement.py  # Standalone validation script for Phase 9 (see
│                                     # §16) - run on its own, does NOT cascade into
│                                     # the Phase 1-8 regression (by design - see §16)
├── test_explainability.py       # Standalone validation script for Phase 10's
│                                 # explainability module (see §17) - run on its own
├── test_audit_trail.py          # Standalone validation script for Phase 10's audit
│                                 # trail module (see §17) - run on its own
├── test_evaluation.py           # Standalone validation script for Phase 11's
│                                 # evaluation module (see §18) - run on its own
├── test_ui_polish.py            # Standalone validation script for Phase 12's UI/UX
│                                 # polish pass (see §19) - run on its own; mostly
│                                 # structural (pages still load, new elements render,
│                                 # baseline unaffected), since Phase 12 changed no
│                                 # pipeline logic
├── requirements.txt              # Python dependencies for the whole project
├── README.md                     # This file
├── .streamlit/
│   └── config.toml                # Dashboard theme (dark, professional look)
├── data/
│   ├── synthetic_security_logs.csv   # 350 synthetic log records (never modified)
│   ├── analyst_feedback.db           # SQLite feedback database - created
│   │                                   # automatically on first run (Phase 7);
│   │                                   # not shipped pre-populated
│   └── audit_trail.db                # SQLite audit-trail database - a SEPARATE
│                                       # file from analyst_feedback.db, created
│                                       # automatically on first run (Phase 10);
│                                       # not shipped pre-populated
├── knowledge_base/                    # Phase 8 - local defensive reference
│   ├── kb_001_brute_force_login.json     # documents used by the RAG retriever.
│   ├── kb_002_unusual_network_activity.json   # One small JSON file per entry
│   ├── kb_003_port_scanning.json              # (knowledge_id, title, topic,
│   ├── kb_004_suspicious_data_transfer.json   # incident_types, content,
│   ├── kb_005_suspicious_login_location.json  # source_type). Hand-written,
│   ├── kb_006_off_hours_privilege_activity.json  # defensive-only, never
│   ├── kb_007_multi_device_activity.json         # modified by the running app.
│   ├── kb_008_incident_triage_and_review.json
│   ├── kb_009_defensive_incident_response.json
│   └── kb_010_security_logging_monitoring.json
└── src/
    ├── __init__.py
    ├── data_loader.py            # CSV loading + basic stats helper (Phase 0)
    ├── preprocessor.py           # Validation, cleaning, feature engineering,
    │                             # encoding & scaling (Phase 1)
    ├── anomaly_detector.py       # Isolation Forest training/prediction/evaluation
    │                             # + optional SOM clustering visualization (Phase 2)
    ├── rule_engine.py            # Facts, rules, inference, incident classification
    │                             # & initial severity (Phase 3)
    ├── confidence.py             # Evidence-based confidence scoring, confidence
    │                             # levels, uncertainty explanations & the
    │                             # human-review recommendation flag (Phase 4)
    ├── genai_analyzer.py         # Local Ollama integration, prompt construction,
    │                             # response parsing/validation & a deterministic
    │                             # evidence-based fallback (Phase 5)
    ├── response_planner.py       # Conditional defensive response planning, plan
    │                             # validation - advisory only, never executes any
    │                             # action against a real system (Phase 6)
    ├── feedback.py               # Human analyst review persistence (SQLite):
    │                             # approve/reject/modify decisions, stable
    │                             # incident IDs, descriptive feedback insights
    │                             # (Phase 7)
    ├── retriever.py              # Lightweight local RAG: loads knowledge_base/,
    │                             # builds TF-IDF vectors, ranks entries by cosine
    │                             # similarity to the current incident's evidence -
    │                             # no external service, deterministic (Phase 8)
    ├── continuous_improvement.py # Offline, read-only analysis of Phase 7's stored
    │                             # feedback: incident-type disagreement, repeated
    │                             # plan-step modifications, recurring analyst-note
    │                             # themes, and traceable improvement suggestions -
    │                             # never automatic learning (Phase 9)
    ├── explainability.py         # Structured, non-fabricated explanations built
    │                             # from Phases 2-9's own already-computed results -
    │                             # explains existing decisions, never recomputes
    │                             # them (Phase 10)
    ├── audit_trail.py            # Persistent, append-only SQLite audit trail
    │                             # (data/audit_trail.db, separate from Phase 7's
    │                             # feedback database) recording which pipeline
    │                             # stage ran for which incident - observational
    │                             # only, never a decision (Phase 10)
    └── evaluation.py             # Descriptive, measured evaluation & performance
                                   # metrics built from Phases 2-10's own already-
                                   # computed results and audit history - the ONE
                                   # module allowed to read true_label, and only for
                                   # OFFLINE EVALUATION (Phase 11)
```

Every stage of the project's pipeline has its own module under `src/`,
and `app.py` stays a thin dashboard that calls into them rather than
containing pipeline logic itself. `src/retriever.py`,
`src/continuous_improvement.py`, `src/explainability.py`,
`src/audit_trail.py`, and `src/evaluation.py` are the additions made
after the seven originally-planned phases were complete (Phases 8, 9,
10, and 11 respectively). Phase 12 added no new `src/` module at all -
it is a presentation-only pass entirely inside `app.py` (plus
`test_ui_polish.py` and this README).

---

## 4. The Synthetic Dataset

`data/synthetic_security_logs.csv` contains **350 synthetic log
records** (generated by `generate_data.py`, seeded for
reproducibility) representing realistic SOC monitoring events.

Columns:

| Column | Description |
|---|---|
| `timestamp` | When the event occurred |
| `user` | Synthetic username / service account |
| `source_ip` | Origin IP of the request |
| `destination_ip` | Internal server/host contacted |
| `failed_logins` | Failed login attempts around this event |
| `request_count` | Number of requests in the session |
| `port` | Destination port |
| `data_transferred_mb` | Data transferred, in MB |
| `session_duration` | Session length, in seconds |
| `event_type` | Type of event observed (e.g. `login_success`, `brute_force_attempt`, `network_scan_detected`) |
| `login_location` | City/country the login appeared to come from |
| `device_count` | Distinct devices seen for that user around this event |
| `true_label` | `normal` or `suspicious` — **ground truth used ONLY by `src/evaluation.py` (Phase 11) for offline evaluation metrics**; the detection pipeline (and every other module) never uses this column as an input feature |

About 78% of rows are ordinary daily activity and about 22% are
crafted to look suspicious (brute-force attempts, "impossible travel"
logins far from a user's usual location, large data exfiltration-style
transfers, port scans against unusual ports, off-hours privilege
escalation attempts, and logins from an unusually high number of
devices). Suspicion is expressed as a deviation from each user's own
normal baseline rather than by treating any specific country as
inherently suspicious, to keep the data realistic and unbiased.

To regenerate the dataset (e.g. with a different size or ratio, see
`config.py`):

```bash
python generate_data.py
```

---

## 5. Installation

```bash
# 1. (Recommended) create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate        # on Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt
```

> Note: `scikit-learn` is used starting Phase 1 (`StandardScaler`) and
> Phase 2 (`IsolationForest`) and is required for the Data Preprocessing
> and AI Anomaly Detection pages. `minisom` (Phase 2, **optional**) only
> powers the small "Self-Organizing Map" section on the Anomaly Detection
> page — if it isn't installed, or fails to build on your machine, that
> one section shows a message and skips itself; everything else keeps
> working normally. Phase 3 (the rule engine) and Phase 4 (confidence
> scoring) introduce no new dependencies at all - both are pure Python +
> pandas on top of what Phase 1/2 already use. Phase 5 (Generative AI
> summary) uses the `ollama` **Python package** (already in
> `requirements.txt`) to talk to a **local Ollama server** — that server
> is a separate application, not something `pip install` sets up, so it
> is genuinely optional: if it isn't installed or isn't running, the
> Generative AI Analysis page automatically and silently uses a
> deterministic, evidence-based fallback instead of the AI model (see
> §12). Phase 8's RAG retrieval (see §15) introduces no new dependency
> either — it reuses the same `scikit-learn` install already required by
> Phase 1/2 (`TfidfVectorizer` + cosine similarity). No vector database
> package is used anywhere in this project.

---

## 6. Running the Project

```bash
streamlit run app.py
```

This opens the dashboard in your browser (usually at
`http://localhost:8501`). If the dataset doesn't exist yet, generate
it first:

```bash
python generate_data.py
streamlit run app.py
```

**Optional - enabling the local Generative AI (Phase 5):** the app works
fully without this step. To let the "Generative AI Analysis" page use a
real local model instead of its deterministic fallback, separately
install and run [Ollama](https://ollama.com) and pull the configured
model (`OLLAMA_MODEL_NAME` in `config.py`, currently `qwen3:1.7b`):

```bash
ollama serve             # in its own terminal, if not already running
ollama pull qwen3:1.7b   # or whichever model config.py currently names
```

The Streamlit page's "Ollama" status indicator turns from UNAVAILABLE to
AVAILABLE automatically once both are done — no restart or code change
needed.

**End-to-end demo walkthrough** (everything runs locally; nothing here
ever executes a real cybersecurity action):

1. Open **AI Anomaly Detection** and run the Isolation Forest (or just
   visit any later page - it runs automatically with default settings).
2. Open **Incident Classification** to see a suspicious event's triggered
   rules, incident type, and initial severity.
3. Open **Uncertainty & Confidence** to see the same event's confidence
   score/level and human-review recommendation.
4. Open **Generative AI Analysis**, select the same event, review the
   automatically retrieved **"Retrieved Knowledge (RAG)"** entries, and
   click "Generate AI Case Summary" (optional - works with or without a
   local Ollama server, and with or without any knowledge retrieved).
5. Open **Response Planning** to see the ordered, advisory-only defensive
   plan the system proposes for that event.
6. Open **Analyst Review**, select the same event, review everything
   above in one place, and choose **APPROVE**, **REJECT**, or **MODIFY**
   (optionally editing the plan's steps), add a note, and submit.
7. Open **Feedback History** to see the persisted decision, inspect the
   original vs. any modified plan, and see the Feedback Insights summary.

Every step of this walkthrough is a read/compute/display or a single
database write of a small structured record - nothing is executed
against a real network, account, or machine at any point.

---

## 7. What Phase 0 Currently Does

- Loads `data/synthetic_security_logs.csv`.
- Shows summary metrics: total records, unique users, unique source
  IPs, date range.
- Shows an event-type breakdown chart.
- Lets you filter the raw log table by user and preview it.
- Shows a sidebar checklist of the full pipeline, so the project's
  direction is visible even though only Phase 0–1 are implemented.

It intentionally does **not** yet detect anomalies, classify
incidents, or generate AI summaries.

---

## 8. What Phase 1 Currently Does — Data Preprocessing

Phase 1 adds a **"Data Preprocessing"** page to the sidebar (`app.py`)
and a new module, `src/preprocessor.py`, that turns the raw logs into a
clean, numeric feature matrix ready for the anomaly-detection model
that Phase 2 will build.

**Pipeline implemented:**

```
Raw Logs
    ↓
Validation        (validate_logs)
    ↓
Cleaning          (clean_logs)
    ↓
Feature Engineering  (engineer_features)
    ↓
Encoding / Scaling   (prepare_ml_features)
    ↓
ML-Ready Features
```

`run_preprocessing_pipeline()` in `src/preprocessor.py` chains all four
functions together; the Streamlit page just calls that one function and
displays the result. Every function is reusable on its own by later
phases too.

### Validation

Checks (without ever raising an exception): required columns exist, the
dataset isn't empty, numeric columns actually contain numbers, the
timestamp can be parsed, and categorical columns have usable values.
Problems are split into **errors** (dataset is unusable, e.g. a missing
required column or an empty file) and **warnings** (recoverable issues
that cleaning will fix, e.g. a few bad timestamps or duplicate rows).

### Cleaning

- Removes exact duplicate rows.
- Drops rows whose `timestamp` can't be parsed.
- Coerces the six numerical columns to numbers and drops rows that still
  fail (e.g. text in a numeric column).
- Fills missing/blank categorical values with `"unknown"` instead of
  dropping the row, since a blank category isn't fatal.

A before/after row count and a breakdown of *why* rows were removed is
shown on the dashboard.

### Feature Engineering (derived features)

| Feature | How it's computed | Why |
|---|---|---|
| `hour` | Hour (0–23) from `timestamp` | Basic time signal |
| `day_of_week` | Day name from `timestamp` | Display / context |
| `is_off_hours` | `1` if hour is in `22:00–23:59` or `00:00–05:59`, else `0` | Off-hours activity is more suspicious |
| `request_rate` | `request_count ÷ session_duration` (0 if duration is 0) | Requests per second |
| `failed_login_ratio` | `failed_logins ÷ (failed_logins + request_count)` (0 if both are 0) | Share of attempts that failed |
| `is_external_source_ip` | `1` if `source_ip` doesn't start with `10.` | Simple internal-vs-external signal, without one-hot-encoding raw IPs |
| `location_mismatch` | `1` if `login_location` differs from that **user's own** most common location in the dataset | A label-free stand-in for "impossible travel" |

`is_external_source_ip` and `location_mismatch` are the answer to
"identifiers such as IP addresses or usernames would create excessive
dimensions if one-hot encoded" — instead of encoding the raw IP or
location, we derive one small, meaningful number from each.

### Encoding / Scaling → ML-Ready Features

- **`event_type`** (12 categories, low cardinality) is **one-hot
  encoded** — 12 `event_type_*` columns.
- **`user`, `source_ip`, `destination_ip`, `login_location`** are
  **not** one-hot encoded (too many unique values for a small dataset);
  their useful signal was already captured above.
- **`timestamp`** and **`true_label`** are excluded entirely from the
  ML matrix.
- All 12 remaining numeric/derived columns are scaled together with a
  single **`StandardScaler`** for simplicity (Isolation Forest, planned
  for Phase 2, doesn't need scaling, but a distance-based method like a
  SOM does — scaling once now keeps both options open).

**Exact ML-ready feature columns produced (24 total):**

```
failed_logins, request_count, port, data_transferred_mb, session_duration,
device_count, hour, request_rate, failed_login_ratio, is_off_hours,
is_external_source_ip, location_mismatch,
event_type_brute_force_attempt, event_type_database_query,
event_type_email_activity, event_type_file_access,
event_type_large_data_transfer, event_type_login_failure,
event_type_login_success, event_type_network_scan_detected,
event_type_normal_browsing, event_type_privilege_escalation_attempt,
event_type_unusual_login_location, event_type_vpn_connection
```

### Why `true_label` is excluded

`true_label` was added to the synthetic dataset purely so this project
can later check how well an *unsupervised* anomaly detector (Phase 2)
and the rule engine (Phase 3) actually perform. A real-world SOC log
would never come with a "this is suspicious" column — the whole point
of the AI pipeline is to work that out. `prepare_ml_features()` builds
the ML matrix from an explicit **allow-list** of engineered columns
that simply never includes `true_label` (rather than a "drop-list",
which would be easy to accidentally miss a column with).

### The three data "levels" (shown on the dashboard)

1. **Display/original data** — `cleaned_df`: the raw columns after
   cleaning, human-readable.
2. **Derived features** — `feature_df`: (1) plus the seven engineered
   columns above, still human-readable.
3. **ML-ready features** — `ml_df`: fully numeric, one-hot encoded and
   scaled — what Phase 2 will train on.

### Error handling

`validate_logs()` never raises. `run_preprocessing_pipeline()` catches
the "unusable data" cases (missing columns, empty file, or a dataset
where every row is invalid) and returns a result object whose
`.validation.errors` explains why, instead of throwing an exception.
The Streamlit page additionally wraps the whole pipeline call in a
`try/except` so any unexpected error shows a friendly message and
leaves the rest of the app working, instead of crashing it.

### Testing Phase 1

Run the standalone test script (no new dependencies — plain asserts):

```bash
python test_preprocessing.py
```

It checks: the 350-record dataset loads, timestamp conversion works,
the derived time features are generated correctly, the six required
numerical features are present with numeric dtypes, `true_label` is
excluded from the ML feature matrix, the final ML-ready matrix has zero
NaN values, the full pipeline runs without exceptions, and — using
small in-memory test data (the real CSV is never modified) — that
missing columns, an empty dataset, invalid numeric values, invalid
timestamps, duplicate rows, and blank categorical values are all
handled gracefully instead of crashing.

---

## 9. What Phase 2 Currently Does — AI Anomaly Detection

Phase 2 adds an **"AI Anomaly Detection"** page and a new module,
`src/anomaly_detector.py`, that trains an **unsupervised** model on top
of Phase 1's `ml_df` to flag events that look statistically unusual —
without ever being told which events are actually suspicious.

**Full flow:**

```
Cybersecurity Logs
        ↓
Phase 1 Preprocessing
        ↓
ML-Ready Features
        ↓
Isolation Forest
        ↓
Anomaly Score
        ↓
Normal / Anomalous
        ↓
Optional Post-Hoc Evaluation
```

### Why anomaly detection is needed

A SOC analyst can't manually read every log line. Anomaly detection is
the first automated triage step: instead of a human scanning 350 (or
350,000) rows, the model narrows attention down to the small fraction
of events that look statistically unusual, so the analyst's time goes
to the events most likely to matter.

### Why Isolation Forest

Isolation Forest was chosen (and required by this phase) because it:
- needs **no labels** to train — a realistic fit for real SOC logs,
  where "this was an attack" is rarely known in advance;
- handles the mix of feature types in `ml_df` (counts, ratios, 0/1
  flags, one-hot columns) without assumptions about their distribution;
- scales well to more logs without becoming a bottleneck; and
- is simple enough to explain in a viva: "it isolates points by random
  splits, and outliers get isolated faster."

### How Isolation Forest works (plain language)

Isolation Forest builds many random decision trees. Each tree repeatedly
picks a random feature and a random split value, dividing the data in
two, over and over, until every point sits alone in its own "leaf."

A normal event tends to sit in a dense region of the data — surrounded
by similar events — so it takes **many** random splits before it ends
up isolated. An unusual event (e.g. a huge data transfer, or 15 failed
logins) tends to stand apart from everything else, so it gets isolated
in **very few** splits. The model turns "how many splits did it take,
averaged across all the trees" into the anomaly score: events that are
easy to isolate score as more anomalous.

### Features used

Exactly the 24-column `ml_df` produced by Phase 1 (see §8) — the six
core numeric columns, `hour`, `request_rate`, `failed_login_ratio`,
the three engineered 0/1 flags (`is_off_hours`, `is_external_source_ip`,
`location_mismatch`), and the 12 one-hot `event_type_*` columns. Nothing
else is added, and nothing here is removed — Phase 2 reuses Phase 1's
pipeline as-is rather than duplicating any preprocessing logic.

### How predictions are produced

`run_anomaly_detection()` in `src/anomaly_detector.py`:
1. Fits `sklearn.ensemble.IsolationForest` on `ml_df` only.
2. Calls `.predict()` (→ `-1` = anomaly / `1` = normal) and
   `.decision_function()` (→ the raw anomaly score) on the same data.
3. Re-attaches those outputs to the original, human-readable event
   columns (timestamp, user, source/destination IP, ports, etc.) by row
   index, so the analyst can see *why* an event was flagged, not just
   *that* it was.

### What the anomaly score means

- `anomaly_score_raw` is scikit-learn's `decision_function` output
  directly — **higher means more normal**, and values below roughly zero
  correspond to what `.predict()` calls an anomaly, given the chosen
  contamination.
- `anomaly_score` (the one shown across the UI) is simply
  `-anomaly_score_raw` — flipped so **higher means more anomalous**,
  which reads more naturally when sorting a table of suspicious events.
  This is the only transformation applied; the raw score is always kept
  alongside it in the code for transparency.
- **This is not a probability.** It does not mean "70% chance this is an
  attack." It measures how easily/quickly the model could isolate that
  point relative to everything else — a relative measure of unusualness,
  not a calibrated likelihood. `anomaly_status` (`NORMAL` /
  `ANOMALOUS`) is the simple label derived from the score via the
  model's contamination-based cutoff.

### Why `true_label` is excluded from training (and how it IS used)

`prepare_ml_features()` (Phase 1) builds `ml_df` from an explicit
allow-list of columns that never includes `true_label` — so it is
structurally impossible for it to reach `IsolationForest.fit()`.
`train_isolation_forest()` additionally asserts this and raises an error
if `true_label` is ever present in what's passed to it, as a second,
independent safety net.

`true_label` is used in exactly one place: **after** predictions already
exist, in `evaluate_against_ground_truth()`, purely to score how well
the unsupervised model agrees with the ground truth that was baked into
this synthetic dataset (accuracy / precision / recall / F1 / confusion
matrix). This mirrors how anomaly detection is evaluated in practice —
train blind, then check against known incidents afterward — and the
model was **not** re-tuned by looking at these numbers and adjusting
parameters to chase a better score; `contamination` was set once, up
front, as documented in `config.py`.

### Optional: Self-Organizing Map (SOM)

If the optional `minisom` package is installed, the page offers a small,
opt-in "Self-Organizing Map" section: it trains a small SOM on the same
`ml_df` and plots which grid cell ("neuron") each event lands on, purely
as an exploratory look at how events cluster in feature space. It is
**not** a second anomaly detector, produces no `anomaly_status` of its
own, and never influences the Isolation Forest results above it. If
`minisom` isn't installed (or fails to build on a given machine — it
occasionally does on very new Python/setuptools combinations), the page
shows a plain message and continues normally; every other part of the
dashboard is unaffected.

### Limitations of using synthetic data here

- The synthetic dataset (Phase 0) was generated with fairly distinct
  suspicious patterns (large spikes in failed logins, data volume, port
  numbers, etc.), so Isolation Forest performs unusually well on it —
  real-world logs are noisier and the same model would need more
  tuning and a lower expectation of near-perfect scores.
- `contamination` (the assumed anomaly rate) was chosen with knowledge
  of how this dataset was built; on real logs, the true rate is unknown
  and would typically be estimated from historical incident volume or
  swept over a small range instead of assumed.
- `destination_ip` is always an internal server address in this
  synthetic data, so it currently carries little signal — noted as a
  limitation, not a bug.
- Isolation Forest flags *statistical* outliers, not *confirmed*
  incidents — a real deployment still needs the rule engine (Phase 3)
  and a human analyst (Phase 6/7) in the loop before acting on a flag.

### Testing Phase 2

```bash
python test_anomaly_detection.py
```

Verifies (12 checks, matching the Phase 2 spec): Phase 1 still produces
the expected `ml_df`; `true_label` is absent from the model's input;
Isolation Forest trains successfully; every processed event gets an
`anomaly_prediction`, `anomaly_score`, and `anomaly_status`; result row
counts match the processed event count; predictions keep both the
status and the original event columns together; evaluation metrics are
produced when `true_label` exists (and cleanly skipped when it doesn't);
all three dashboard pages run without exceptions
(`streamlit.testing.v1.AppTest`); invalid parameters (bad
`contamination`, bad `n_estimators`, an empty feature matrix, or a
feature matrix that incorrectly contains `true_label`) all raise a
clean, catchable error instead of crashing; and a simulated missing
`minisom` package doesn't affect Isolation Forest at all.

---

## 10. What Phase 3 Currently Does — Knowledge-Based Rule Engine & Incident Classification

Phase 3 adds an **"Incident Classification"** page and a new module,
`src/rule_engine.py`, that sits on top of Phase 2's predictions and turns
them into an explainable incident category and initial severity - using
explicit, hand-written cybersecurity rules instead of another ML model.
No new dependencies were needed.

**Conceptual flow:**

```
Phase 2 Anomaly Detection
        ↓
Observed Evidence
        ↓
Derived Facts
        ↓
Knowledge / Rules
        ↓
Inference
        ↓
Incident Classification
        ↓
Initial Severity
```

### What a knowledge-based rule engine is

Instead of a model that learns patterns from data, a rule engine encodes
a human expert's knowledge directly as `IF ... THEN ...` statements. Each
rule has a stable ID, a precise condition, a conclusion, and a
plain-English explanation - so every decision the system makes can be
traced back to the exact rule and evidence that produced it. That
traceability is the entire point: it's the difference between "the
model said so" and "RULE-BF-01 fired because failed_logins was 23,
which is ≥ the threshold of 5, on an event Phase 2 already flagged
anomalous."

### How facts are created

`extract_facts()` reads one processed event (its original fields, Phase
1's `is_off_hours`/`location_mismatch`, and Phase 2's `anomaly_status`)
and produces ten booleans - `anomalous_event`, `high_failed_logins`,
`high_request_count`, `unusual_port`, `high_data_transfer`,
`off_hours_activity`, `location_mismatch`, `multiple_devices`,
`privilege_escalation_attempt`, `network_scan_indicator` - each compared
against a named threshold from `config.py` (`FAILED_LOGIN_THRESHOLD`,
`REQUEST_COUNT_THRESHOLD`, `DATA_TRANSFER_THRESHOLD`,
`DEVICE_COUNT_THRESHOLD`, `COMMON_PORTS`). Every field is read
defensively (`row.get(...)` with a safe default), so an event missing an
optional column never crashes the engine - it just can't support the
fact(s) that depend on that column. **`true_label` is never read here.**

### How rules create inferences

Seven rules are evaluated against the facts for **every** event - the
engine never stops at the first match, so all supporting evidence is
kept:

| Rule ID | Name | Condition (exact logical form) | Incident Type |
|---|---|---|---|
| `RULE-BF-01` | Possible Brute Force | `anomalous_event AND high_failed_logins` | Possible Brute Force |
| `RULE-NET-01` | Unusual Network Activity | `anomalous_event AND (high_request_count OR network_scan_indicator)` | Unusual Network Activity |
| `RULE-DATA-01` | Suspicious Data Transfer | `anomalous_event AND high_data_transfer` | Suspicious Data Transfer |
| `RULE-LOC-01` | Suspicious Login Location | `anomalous_event AND location_mismatch` | Suspicious Login |
| `RULE-PRIV-01` | Off-Hours Privilege Activity | `anomalous_event AND off_hours_activity AND privilege_escalation_attempt` | Other Anomaly |
| `RULE-DEV-01` | Unusual Multi-Device Activity | `anomalous_event AND multiple_devices` | Other Anomaly |
| `RULE-SCAN-01` | Port Scanning Indicator | `anomalous_event AND (network_scan_indicator OR (unusual_port AND high_request_count))` | Port Scanning Indicator |

Parentheses above are exact - e.g. `RULE-SCAN-01` fires if EITHER the
event's `event_type` is a scan indicator, OR (the port is unusual AND
the request count is high); it does **not** fire from an unusual port
alone. Every condition is a plain Python boolean expression with
explicit parentheses (`src/rule_engine.py`), so there is no ambiguous
operator precedence to misread.

### How incident classification works

- Not anomalous → **Normal Activity** (no rule is even evaluated to
  reach this - every rule's first clause is `anomalous_event`).
- Anomalous, but no rule matched → **Other Anomaly**.
- Anomalous, one or more rules matched → the matched rule's
  `incident_type`, or, if rules from **different** categories fire on
  the same event, the highest-priority one wins by a fixed, documented
  order (`INCIDENT_TYPE_PRIORITY` in `src/rule_engine.py`):
  Possible Brute Force → Port Scanning Indicator → Suspicious Data
  Transfer → Suspicious Login → Unusual Network Activity → Other Anomaly.
  This order exists so a more specific, technique-named finding (e.g. a
  brute-force attempt that also happens to have a high request count)
  is reported as itself rather than folded into the broader "Unusual
  Network Activity" bucket. Every triggered rule and its evidence is
  still kept and shown to the analyst - only the single *primary* label
  is chosen this way.

### How severity is assigned

Deterministic, rule-based, and explicitly **not a probability or
confidence score** (that's Phase 4):

- **N/A** - the event isn't anomalous; severity doesn't apply.
- **LOW** - anomalous, but no specific rule matched (a weak/generic anomaly).
- **HIGH** - any of: two or more distinct rules matched (multiple
  corroborating indicators); the off-hours privilege-escalation rule
  matched (`RULE-PRIV-01`); or the suspicious-data-transfer rule matched
  **and** `data_transferred_mb` ≥ `SEVERE_DATA_TRANSFER_THRESHOLD`
  ("severe data-transfer behavior").
- **MEDIUM** - exactly one rule matched and none of the HIGH conditions above apply.

### Why this is explainable

Every event's classification can be walked back through a concrete
chain: the exact evidence values → the boolean facts they produced → the
named rule(s) those facts satisfied (with a plain-English explanation
per rule) → the resulting incident type and severity. The **Inference
Details** section on the page renders exactly this chain for any
selected event. Nothing is a black box - there's no model weight or
learned parameter anywhere in this module.

### Why `true_label` is not used

`run_rule_engine()` builds its working data by joining Phase 1's
`feature_df` with only three columns from Phase 2's output
(`anomaly_prediction`, `anomaly_score`, `anomaly_status`) - `true_label`
is deliberately left out of that join. `config.LABEL_COLUMN` is never
imported into `rule_engine.py`, so it is structurally unavailable to any
rule, not just unused by convention. All rule thresholds
(`FAILED_LOGIN_THRESHOLD`, `REQUEST_COUNT_THRESHOLD`,
`DATA_TRANSFER_THRESHOLD`, `DEVICE_COUNT_THRESHOLD`) were chosen by
looking at how the synthetic dataset's normal-vs-suspicious patterns
were *generated* (Phase 0) and general defensive-security reasoning
(e.g. "5+ failed logins is unusual") - never by checking `true_label`
and adjusting a threshold to improve a score.

### Limitations of a rule-based cybersecurity system

- **Fixed thresholds don't adapt.** A real network's notion of "high
  request count" changes over time and by role; these thresholds would
  need periodic review, not a one-time setting.
- **Rules only catch what they're written for.** An attack pattern that
  doesn't resemble brute force, scanning, exfiltration, impossible
  travel, off-hours privilege use, or multi-device access falls through
  to "Other Anomaly" with no specific explanation - the rule base would
  need to grow to cover new patterns.
- **No severity nuance within a bucket.** Two very different "MEDIUM"
  events are both just "MEDIUM" - Phase 4's confidence/uncertainty layer
  is meant to add the missing shading.
- **Priority is a simplification.** When several categories legitimately
  apply, collapsing to one primary label (even while keeping all the
  evidence) is a judgment call, not a mathematical necessity - a real
  SOC workflow might instead track all matched categories in parallel.
- **Still built on Phase 2's synthetic-data performance caveats** (see
  §9) - a rule firing correctly on this dataset doesn't guarantee the
  same threshold values generalize to real traffic.

### Testing Phase 3

```bash
python test_rule_engine.py
```

This single command runs all Phase 3 checks (17, matching the spec) AND
then re-runs `test_preprocessing.py` and `test_anomaly_detection.py` in
full, so one command regression-tests the entire pipeline built so far.
Phase 3 checks include: a normal event → Normal Activity; each of the
seven rules individually triggered by a crafted event (including
verifying the precise AND/OR precedence, e.g. that `RULE-PRIV-01` needs
BOTH off-hours and privilege escalation, and that `RULE-SCAN-01` fires
via either of its two OR-branches but not from an unusual port alone);
multiple rules firing on one event with the priority mechanism correctly
picking the primary category; classification and severity determinism
(20 repeated runs, identical results); `true_label` proven both
structurally absent (not imported) and behaviorally irrelevant (flipping
it changes nothing); missing optional fields and a completely empty
input both handled without crashing; a full-dataset integration check
confirming all 7 rules fire at least once on the real 350-record
dataset; and the Streamlit page rendering and responding to interaction
with no exceptions.

---

## 11. What Phase 4 Currently Does — Uncertainty & Confidence Scoring

Phase 4 adds an **"Uncertainty & Confidence"** page and a new module,
`src/confidence.py`, that sits on top of Phase 3's classifications and
answers a question Phase 3 deliberately leaves open: *how strong and
consistent is the evidence behind the current interpretation?* No new
dependencies were needed - it's pure Python + pandas, exactly like
Phase 3.

**Conceptual flow:**

```
Observed Evidence
        ↓
Anomaly Evidence
        ↓
Rule Evidence
        ↓
Supporting Indicators
        ↓
Evidence Consistency
        ↓
Confidence Score
        ↓
Confidence Level
        ↓
Human Review Recommendation
```

### What "uncertainty" means here, and why it's needed

Phase 3's rule engine is deliberately black-and-white: a rule either
matches or it doesn't, and every anomalous event gets exactly one
incident category and one severity. That's great for explainability, but
it hides a real difference between two events that both end up
"Possible Brute Force, HIGH": one might be backed by a strong anomaly
score and several corroborating indicators, while the other might barely
clear the anomaly threshold with only the bare minimum of evidence for
that one rule. Phase 4 makes that difference visible instead of silently
collapsing it, so an analyst (or a future phase, like Phase 7's review
workflow) knows which classifications deserve more scrutiny.

### How the evidence-based score works

`src/confidence.py` computes one **evidence-based confidence score**, in
`[0, 1]`, per event, as a weighted blend of four transparent components
(no ML model, no learned weights - every number below is a fixed,
documented constant in `config.py`):

```
confidence_score =
      CONFIDENCE_WEIGHT_ANOMALY     × anomaly_evidence        (weight 0.35)
    + CONFIDENCE_WEIGHT_RULES       × rule_evidence            (weight 0.30)
    + CONFIDENCE_WEIGHT_SUPPORTING  × supporting_evidence      (weight 0.15)
    + CONFIDENCE_WEIGHT_CONSISTENCY × consistency_evidence     (weight 0.20)
```

The weights sum to 1.0, and since every component is itself constrained
to `[0, 1]`, the blended score is naturally within `[0, 1]` (a final
`min(max(score, 0), 1)` clip is kept purely as a defensive safety net).
Anomaly and rule evidence get the largest, roughly-equal shares because
they're this project's two primary interpretation signals (the Phase 2
AI model and the Phase 3 knowledge base); supporting indicators get a
smaller share since most of their signal already feeds the rules;
consistency gets a moderate share so agreement/disagreement across the
other three sources meaningfully moves the score without dominating it.

**1. Anomaly evidence** - Phase 2's `anomaly_score` is first
batch-relative min-max normalized to `[0, 1]`
(`normalize_anomaly_scores()`): `(score - batch_min) / (batch_max -
batch_min)`, with every event getting the neutral 0.5 if the batch has no
spread at all. This is a deterministic, documented, bounded rescaling -
**never called a probability** - and it never reads `true_label`. The
normalized value then measures how strongly the anomaly signal supports
the *current* classification: a high normalized score supports an
anomalous/incident interpretation, while a low one supports a normal
one; a normal event with a borderline (high) score gets weaker "this is
normal" evidence than one that's clearly unremarkable.

**2. Rule evidence** - based on how many Phase 3 rules triggered
(`RULE_EVIDENCE_ZERO_RULES` = 0.2, `RULE_EVIDENCE_ONE_RULE` = 0.6,
`RULE_EVIDENCE_MULTIPLE_RULES` = 0.9), with a small
`STRONG_RULE_EVIDENCE_BONUS` (0.1) added if a triggered rule's conclusion
is one of `rule_engine.HIGH_RISK_CONCLUSIONS` (currently: off-hours
privilege activity). Rules never evaluate for Normal Activity, so a
fixed `RULE_EVIDENCE_NORMAL_BASELINE` (0.6) is used there instead of
letting an inapplicable component drag the score down or inflate it.
More rules isn't blindly treated as "definitely true" - it's only one of
four components.

**3. Supporting-indicator evidence** - the fraction of eight observed
indicator facts that are true (high failed logins, high request count,
high data transfer, off-hours activity, location mismatch, multiple
devices, a privilege-escalation event type, a network-scan event type).
Symmetric with anomaly evidence: for an anomalous event, more present
indicators support the incident interpretation; for a normal event,
*fewer* present indicators support the normal interpretation - an
indicator crossing its threshold on an otherwise-normal event is mild
evidence against that classification, not for it.

**4. Evidence consistency** - a simple three-tier check
(`CONSISTENCY_HIGH` = 1.0, `_MEDIUM` = 0.6, `_LOW` = 0.25) of whether the
anomaly signal, the rule engine, and the supporting indicators agree.
For an anomalous event: HIGH if a rule fired **and** 2+ indicators are
present; LOW if **zero** rules fired **and zero** indicators are present
(a "bare" anomaly with no explainable corroboration - this is exactly
the *"anomaly detected, but evidence is insufficient to confidently
assign a specific incident type"* situation called out for Phase 4).
For a normal event: HIGH if zero indicators are present (clean
agreement); LOW if 2+ indicators are present despite the normal
verdict - genuinely conflicting evidence worth a second look.

### Evidence sources used (and what's deliberately excluded)

Only fields already produced by earlier phases are used: Phase 2's
`anomaly_status`/`anomaly_score`; Phase 3's triggered rules, rule
conclusions, and derived facts; and the same observed event fields Phase
3 already turned into facts (failed logins, request count, data
transferred, device count, off-hours flag, location mismatch, event
type). `true_label` is never inspected, never imported, and never a
parameter anywhere in `src/confidence.py` - see the section below.

### Confidence levels

Uses the existing `config.py` thresholds, unchanged from their Phase 0
placeholder definitions:

| Score range | Confidence level |
|---|---|
| `score < 0.5` (`LOW_CONFIDENCE_THRESHOLD`) | **LOW** |
| `0.5 ≤ score < 0.8` (`HIGH_CONFIDENCE_THRESHOLD`) | **MEDIUM** |
| `score ≥ 0.8` | **HIGH** |

### Why confidence is NOT a probability

`confidence_score` is an **explainable, evidence-based score** built from
the four hand-weighted, documented components above - it is *not* a
calibrated statistical probability that an attack occurred, and *not* a
guarantee that Phase 3's `incident_type`/`severity` are correct. The app
and this README always phrase it as "confidence score: 0.85" / "confidence
level: HIGH" with a supporting reason, and deliberately never as "there is
an 85% probability this is a brute-force attack."

### Why `true_label` is excluded

Exactly the same discipline as Phase 2 and Phase 3: `config.LABEL_COLUMN`
is never imported into `src/confidence.py`, so it is structurally
unavailable to any function in the module, not just unused by
convention. The four component weights and every tier cutoff
(`RULE_EVIDENCE_*`, `CONSISTENCY_*`, `LOW_/HIGH_CONFIDENCE_THRESHOLD`)
were chosen from documented reasoning about what should count as
stronger or weaker evidence - never by tuning against `true_label` or
any evaluation metric (accuracy/F1/etc.). `test_confidence.py` proves
this both structurally (the name is absent from the module's namespace,
and never accessed as a dict/Series key anywhere in the code) and
behaviorally (flipping `true_label` on an otherwise-identical event
leaves `confidence_score`, `confidence_level`, and
`human_review_recommended` all unchanged).

### Why severity and confidence are kept separate

Phase 3's `severity` answers *"how serious is this incident according to
the rules that fired?"*. Phase 4's `confidence_score`/`confidence_level`
answer a different question: *"how strong and consistent is the evidence
supporting that interpretation?"*. They are computed independently and
never merged into one number - **a HIGH-severity, LOW-confidence event is
valid and expected.** On the real 350-event dataset, 11 events are
HIGH severity with LOW confidence (e.g. an off-hours privilege-escalation
or large data-transfer rule fired - which is always/likely HIGH severity
- but the underlying anomaly score was only barely over the model's
threshold, so the supporting evidence is thin). `test_confidence.py`
verifies this combination is possible and correctly labeled, not
accidentally prevented by the formula.

### When human review is recommended

`human_review_recommended` is a **recommendation flag only** - Phase 7
will build the actual analyst review workflow; nothing in Phase 4 takes
any automatic action. It is set `True` when any of:

- overall confidence is **LOW**;
- the event is anomalous but **zero rules** currently support a specific
  incident interpretation (a "bare" anomaly - insufficient evidence),
  regardless of what the blended score came out to;
- evidence consistency is the **LOW** tier (conflicting or insufficient
  agreement across the anomaly/rule/indicator sources), surfaced even if
  the blended score didn't happen to land in LOW.

### Limitations of the heuristic confidence model

- **Hand-weighted, not learned.** The four weights and every tier cutoff
  are a documented, reasoned starting point, not calibrated against any
  ground truth - by design, since using `true_label` for that would
  violate the project's core constraint. A different weighting could
  reasonably shift scores at the margins without being "more correct."
- **Batch-relative anomaly normalization.** The same raw `anomaly_score`
  can normalize to a different value depending on which batch of events
  it's scored alongside (see `normalize_anomaly_scores()`). This is
  documented and deterministic *within* a given batch, but it means
  confidence scores from two different runs (e.g. different date ranges)
  aren't directly comparable in absolute terms.
- **No "bare anomaly" (0 rules) events exist in this particular synthetic
  dataset.** Every anomalous event here happens to trip at least one
  Phase 3 rule (the same finding Phase 3's README section noted for rule
  coverage), so the LOW-consistency "insufficient evidence" path is
  exercised by a crafted test case in `test_confidence.py` rather than by
  a real row in `data/synthetic_security_logs.csv`. The code path exists
  and is tested; this dataset just doesn't happen to trigger it naturally.
- **Coarse three-level output.** LOW/MEDIUM/HIGH (and WEAK/MODERATE/STRONG
  for evidence strength) necessarily collapse a continuous score into a
  few buckets for readability - the underlying `confidence_score` and the
  per-component breakdown remain available for anyone who wants finer
  granularity.
- **Still built on Phases 2 and 3's caveats.** A confident-looking score
  is only as good as the anomaly model and rule set underneath it - see
  §9 and §10 for their own documented limitations.

### Testing Phase 4

```bash
python test_confidence.py
```

This single command runs all Phase 4 checks (17, matching the spec) AND
then re-runs `test_rule_engine.py` in full - which itself re-runs
`test_preprocessing.py` and `test_anomaly_detection.py` - so one command
regression-tests the entire pipeline built so far (261 checks total: 44
Phase 1 + 53 Phase 2 + 61 Phase 3 + 103 Phase 4). Phase 4 checks include:
a normal event producing a valid, non-absolute confidence; a bare anomaly
(0 rules) scoring lower than an otherwise-identical 1-rule event and
always recommending review; valid results for one-rule and multi-rule
events; every confidence score staying in `[0, 1]` across many crafted
scenarios and the real 350-event dataset; exact threshold-boundary
classification into LOW/MEDIUM/HIGH; LOW confidence reliably triggering
`human_review_recommended`; a constructed HIGH-severity/LOW-confidence
case proving the two concepts stay independent; `true_label` proven both
structurally absent and behaviorally irrelevant; deterministic repeated
calls; missing optional fields and a completely empty input both handled
without crashing; and the Streamlit page rendering and responding to
interaction with no exceptions.

---

## 12. What Phase 5 Currently Does — Generative AI Analyst Summary

Phase 5 adds a **"Generative AI Analysis"** page and a new module,
`src/genai_analyzer.py`, that turns the STRUCTURED results Phases 2-4
already produced for one selected event into a plain-English, analyst-
friendly case summary and defensive recommendations - using a local
Ollama model when one is available, and a fully deterministic fallback
when it isn't.

**Conceptual flow:**

```
Structured Security Analysis
        ↓
Incident + Severity + Confidence
        ↓
Evidence Context
        ↓
Local Ollama Model
        ↓
AI Case Summary
        ↓
Defensive Recommendations
        ↓
Human Analyst
```

### Why Generative AI is used

Phases 2-4 are precise but terse: a table row with an incident type, a
severity, a confidence score, and a handful of fact names. That's ideal
for building the pipeline and for an analyst who already knows this
project's vocabulary, but it isn't the kind of plain-English narrative a
human analyst writes up in a ticket. Phase 5 doesn't add any new
detection capability - it adds a **communication layer** that explains an
already-complete analysis in ordinary language, the same way a senior
analyst might summarize a junior analyst's findings for a report.

### What information is provided to the model, and why not `true_label`

`build_incident_context()` builds an `IncidentContext` **exclusively**
from Phase 3's `InferenceRecord` and Phase 4's `ConfidenceRecord` -
`incident_type`, `severity`, `confidence_score`/`level`/`evidence_strength`,
`anomaly_status`/`anomaly_score`, the observed evidence fields, derived
facts, triggered rule IDs and explanations, supporting indicators,
confidence reasons, the uncertainty reason, and the human-review flag.
It does **not** receive the raw dataset, other events, or anything the
application hasn't already computed. `IncidentContext` has **no field for
`true_label` at all** - it is structurally impossible to pass it in,
not merely unused by convention, exactly like `src/rule_engine.py` and
`src/confidence.py` before it (neither of which reads `true_label`
either, so it was never available to inherit even if this module wanted
to).

### What the AI generates

Nine structured fields (JSON, `REQUIRED_AI_FIELDS`), rendered as eight UI
sections (CASE SUMMARY, WHAT WAS OBSERVED, WHY SUSPICIOUS, SEVERITY,
CONFIDENCE, SUPPORTING EVIDENCE, RECOMMENDATIONS, HUMAN REVIEW - severity
and classification are combined into one SEVERITY section since they're
closely related): a plain-English summary; only-the-facts observed
evidence; why the anomaly/rule evidence looked suspicious; the incident
classification and severity (restated, never re-decided); an explanation
of the confidence level and its uncertainty; the key supporting
indicators; a list of defensive-only recommendations; and a statement of
whether human review is recommended.

### How Ollama is used

`src/genai_analyzer.py` calls the local `ollama` Python package (already
in `requirements.txt`) against `config.OLLAMA_MODEL_NAME` (currently
`"qwen3:1.7b"`, changeable from the page's Settings expander without
editing any code). Before every call, `get_ollama_status()` checks (1) whether
the Ollama package can reach a local server at all
(`is_ollama_available()`) and (2) whether the configured model is
actually pulled (`is_model_available()`). The model name is never
hard-coded in `app.py` - it always flows from `config.py` (or the
Settings override) into `src/genai_analyzer.py`.

### What happens if Ollama is unavailable

The application never requires Ollama to be installed or running.
`generate_ai_case_summary()` checks status first and, if Ollama isn't
reachable or the model isn't pulled, calls
`generate_fallback_summary()` instead - a fully deterministic function
that builds all nine sections directly from the same `IncidentContext`
(no ML, no LLM). It is never just "AI unavailable": every section is
populated from the real structured evidence, and it is always clearly
labeled `"Deterministic fallback — Generative AI unavailable."` so the
analyst knows which mode produced the text. **In this project's own
sandboxed test/build environment, no Ollama server is running at all**,
so every example below was produced by the fallback path - which is
exactly the situation Phase 5 was required to handle gracefully, not a
gap in the implementation.

Real example (from the actual 350-event dataset, fallback mode):

> **SYSTEM:** Incident Type = Possible Brute Force · Severity = HIGH ·
> Confidence = MEDIUM (0.76)
>
> **CASE SUMMARY:** This event was flagged ANOMALOUS by the anomaly
> detector and classified as 'Possible Brute Force' with HIGH severity,
> based on 2 triggered rule(s) and 6 supporting indicator(s). Confidence
> in this interpretation is MEDIUM (score 0.76).
>
> **WHY SUSPICIOUS:** Triggered rule evidence: RULE-BF-01 (failed_logins
> = 5 >= threshold 5, on an ANOMALOUS event) \| RULE-PRIV-01 (event_type =
> 'privilege_escalation_attempt' occurred off-hours, hour = 2) - a
> combination treated as high-risk regardless of category.
>
> **RECOMMENDATIONS:** Review the relevant logs for this event; verify
> the affected account with its owner; check for a pattern of repeated
> failed logins; verify the privilege change was authorized through
> normal change management; collect related events over the surrounding
> time window.

### How hallucination risk is reduced

Several layers, not just prompt wording:

1. **Structured input only** - the model never sees the raw dataset, so
   it has nothing to hallucinate additional usernames/IPs/events from.
2. **A strict system prompt** telling the model to use only the supplied
   evidence, distinguish observed fact from inference, and never invent
   facts.
3. **Structured JSON output**, validated by `parse_ai_response()` against
   an exact required-field list - a response missing fields, not valid
   JSON, or not an object is rejected rather than partially trusted.
4. **A text-fallback safety net**: if the model returns non-JSON free
   text, that text is kept only as supplementary narrative - every
   FACTUAL field (classification, severity, confidence, human review,
   recommendations) is still generated deterministically from `context`,
   never from the unvalidated text.
5. **A keyword-based offensive-content filter**
   (`_contains_offensive_content()`) that strips individual offending
   recommendations, or discards the entire AI response in favor of the
   deterministic fallback if any section looks like exploitation/attack
   guidance - defense in depth on top of the prompt's own instruction not
   to produce it.
6. **Unconditional authoritative-value prefixing** - see below.

### Why the AI does not override the structured analysis

This is enforced in code, not just requested in the prompt.
`_finalize_result()` always prefixes `incident_classification`,
`confidence_explanation`, and `human_review` with the SYSTEM's actual
`incident_type`/`severity`, `confidence_level`/`score`, and
`human_review_recommended` values, **verbatim**, regardless of what the
model produced - the model's own phrasing is kept only as supplementary
text appended afterward. `test_genai.py`'s
`test_ai_cannot_change_structured_classification` proves this directly:
a deliberately "disagreeing" AI response (claiming Normal Activity/LOW
severity for an event the system scored HIGH severity) still results in
a `GenAIResult` whose classification/confidence/review fields state the
correct HIGH severity and the correct confidence level first. Nothing in
this module ever receives or mutates the underlying `RuleEngineResult` or
`ConfidenceEngineResult` objects - it only reads two already-built
per-event records and returns a new, separate `GenAIResult`.

### System-generated analysis vs. AI-generated explanation

The Generative AI Analysis page always shows a **SYSTEM Analysis**
section first (incident type, severity, confidence score/level, straight
from Phase 3/4) and labels everything below it as AI-generated (or
fallback-generated) explanation. The two are visually and structurally
separate: SYSTEM values are metrics read directly from
`InferenceRecord`/`ConfidenceRecord`; AI values are a `GenAIResult`
built afterward, purely for explanation, cached per (event, model) pair
in `st.session_state` so re-visiting the same event doesn't silently
regenerate different wording.

### RAG - implemented in Phase 8

`IncidentContext.retrieved_knowledge` / `retrieved_knowledge_items` were
originally added as a hook for a future retrieval-augmented phase
(Requirement #12) and were always `[]` here in Phase 5 - this module
still never performs retrieval itself; no embeddings, no vector store,
no document search happens anywhere in `src/genai_analyzer.py`. Phase 8
(§15) later filled in that hook from the outside: `app.py` calls
`src/retriever.py` to retrieve reference material and hands the result
to `build_incident_context()`, and this module only (a) includes it in
the prompt as clearly-labeled background context and (b) copies the
retrieval metadata onto the outgoing `GenAIResult` so the UI can display
it. See §15 for the full retrieval design.

### Limitations of using a local language model

- **Only as good as its structured input.** The AI cannot correct a
  wrong Phase 2/3/4 result - it explains whatever it's given, including
  a wrong classification, if one existed upstream.
- **Free-text fields are inherently less checkable than the numeric/
  categorical fields.** The safeguards above catch missing structure and
  obviously offensive content, but cannot mathematically prove every
  sentence is 100% faithful to the evidence the way a rule condition can
  be proven true/false.
- **Local-model quality varies.** A small local model may produce
  blander or less precise prose than a larger hosted model; this project
  intentionally uses a local model (no external API calls, no data
  leaving the machine) as a defensive-project design choice, trading some
  fluency for that guarantee.
- **No caching across sessions.** Generated summaries are cached only in
  `st.session_state` for the running session, not persisted - re-running
  the app regenerates them (or recomputes the fallback) from scratch.
- **The offensive-content filter is keyword-based**, not semantic - it
  catches obvious cases and is a second layer on top of the prompt, not a
  substitute for it; a sufficiently indirect phrasing could in principle
  slip past both. No offensive content was observed in this project's own
  testing (which exclusively exercised the fallback path, since no local
  Ollama server runs in this environment).

### Testing Phase 5

```bash
python test_genai.py
```

This single command runs all Phase 5 checks (18, matching the spec) AND
then re-runs `test_confidence.py` in full - which itself cascades through
`test_rule_engine.py`, `test_preprocessing.py`, and
`test_anomaly_detection.py` - so one command regression-tests the entire
pipeline built so far (347 checks total: 44 Phase 1 + 53 Phase 2 + 61
Phase 3 + 103 Phase 4 + 86 Phase 5). Phase 5 checks include: a bool-typed,
non-raising Ollama availability check; clean handling of a missing/
unconfigured model; a complete, non-generic deterministic fallback that
visibly reflects the actual supplied evidence (proven by comparing two
different events' fallback text); `true_label` excluded both structurally
(`IncidentContext` has no such field) and behaviorally; prompt text
proven to contain the real evidence and proven to exclude `true_label`
(including a distinctive marker value); well-formed JSON parsed
correctly with the authoritative values enforced; five different kinds of
malformed/garbage/empty/offensive AI responses all handled without
crashing; missing optional fields and a fully empty incident handled
safely; a deliberately "disagreeing" AI response proven unable to change
the system's classification, severity, or confidence in the final
output; a real-dataset integration pass; the Streamlit page rendering,
generating, and displaying a result with no exceptions; and the whole
application proven to work correctly with Ollama genuinely unavailable
(this project's own test/build environment, not a mock).

---

## 13. What Phase 6 Currently Does — Defensive Response Planning

Phase 6 adds a **"Response Planning"** page and a new module,
`src/response_planner.py`, that turns the STRUCTURED results Phases 3-4
already produced (plus, optionally, a cached Phase 5 summary) for one
selected event into an ordered, advisory-only **defensive response
plan**. It is a small, transparent **conditional planner** — a fixed
decision tree over `incident_type` / `severity` / `confidence_level` /
`human_review_recommended` — never a model, and never an executor.

> **The system generates a recommended defensive workflow. It does not
> execute cybersecurity actions against real systems.**

**Conceptual flow:**

```
Incident Classification            (Phase 3)
        ↓
  Severity + Confidence            (Phase 3 / Phase 4)
        ↓
  Conditional Planner              (Phase 6)
        ↓
  Ordered Defensive Steps
        ↓
  Human Review
        ↓
  Recommended Response
```

**Conceptual planning-state progression** (labels describing a
recommended workflow only — never executed against a real environment):

```
DETECTED → TRIAGED → EVIDENCE_COLLECTION → ANALYST_REVIEW →
RESPONSE_RECOMMENDATION → MONITOR_OR_ESCALATE → CLOSED_OR_ESCALATED
```

### Why a conditional planner, not a model

Phase 6 does not train or fit anything. Every plan can be traced back to
the exact conditions that selected it — the same explainability
philosophy as Phase 3's rule engine. This keeps the response-planning
layer fully auditable: an instructor or analyst can read
`planning_reason` and `PLAN_ORDERING_RATIONALE` and know precisely why a
given plan looks the way it does, with no hidden weights or training
data involved.

### What the planner never does

Nothing in `src/response_planner.py` blocks an IP, disables an account,
modifies a firewall, terminates a session, isolates a machine, or runs
any command against a real system. Every step's action is phrased as a
recommendation ("review", "verify", "collect", "recommend ... according
to policy", "escalate", "document") — never an imperative technical
command. `validate_plan()` enforces this with a keyword blocklist
(`config.PROHIBITED_PLAN_ACTION_KEYWORDS`) as defense in depth, checked
even though the step library itself never produces such text — exactly
so a future bug or careless edit is caught immediately instead of
silently shipping.

### The base step library and the six incident-specific plans

`INCIDENT_CORE_STEPS` holds three hand-written investigative steps for
each of the six non-normal incident categories (Possible Brute Force,
Suspicious Login, Suspicious Data Transfer, Port Scanning Indicator,
Unusual Network Activity, Other Anomaly) — each drawn directly from what
that category's Phase 3 rule(s) actually detected. **Normal Activity
receives no incident-response plan at all** (`steps=[]`, priority
`ROUTINE`, `review_required=False`) — routine monitoring continues
instead.

A single shared, conditionally-built **tail sequence** — evidence
collection, human review, a policy-based recommendation, monitor/
escalate, document — is appended after the core steps, so the closing
logic lives in exactly one place (`_build_tail_steps()`) rather than
being duplicated six times over. Steps always follow this same
investigative lifecycle in a fixed order; only whether an optional step
is included, and each step's wording/priority, changes with severity,
confidence, and the human-review flag (see `PLAN_ORDERING_RATIONALE` in
the module for the full "why this order?" explanation).

### Severity-aware planning

`_severity_base_priority()` maps HIGH → HIGH, MEDIUM → STANDARD, LOW →
ROUTINE. The policy-based "Recommend containment or escalation according
to organizational policy" step is included only for HIGH/MEDIUM
severity — LOW severity explicitly prioritizes routine investigation,
monitoring, and documentation over a containment recommendation.

### Confidence-aware planning

A dedicated "Collect additional evidence" step is skipped when
confidence is already HIGH (existing evidence is strong enough), and
included — with a confidence-specific reason — for MEDIUM or LOW
confidence. LOW confidence additionally guarantees a mandatory human-
review step, regardless of the `human_review_recommended` flag.

**Severity and confidence are combined into an overall plan priority
without being merged into one score** (`_overall_priority()`): severity
sets a base tier from `config.PLAN_PRIORITY_LEVELS`
(`ROUTINE`/`STANDARD`/`HIGH`/`URGENT`), and LOW confidence bumps that
base up exactly one tier. Two real examples, generated by this exact
code on the real dataset:

- **HIGH severity + LOW confidence → URGENT priority.** Event #11 in the
  dataset (`Suspicious Data Transfer`, HIGH severity, LOW confidence)
  produces an 8-step URGENT plan that explicitly collects more evidence
  and requires human review before any high-impact response — exactly
  the "urgent analyst attention, no automatic action" behavior required.
- **MEDIUM severity + HIGH confidence → a normal STANDARD plan.** The
  same incident type at MEDIUM severity and HIGH confidence produces a
  STANDARD-priority plan with no extra evidence-collection step (the
  evidence is already strong) but still includes the policy-based
  recommendation step — a completely different, less urgent plan from
  the case above, proving severity and confidence are never collapsed
  into a single number.

### Human review handling

`confidence_record.human_review_recommended` (Phase 4's own flag) is
checked independently of confidence level. Whenever it is `True`, or
confidence is LOW, the plan includes an explicit **"Human analyst review
required before any high-impact response"** step whose `reason` field
names which trigger(s) applied, and which itself has
`requires_human_review=True`. A plan's `review_required` field is `True`
whenever *any* step in the plan requires human review — mirrored in the
"Review Required" metric on the Response Planning page.

### Using Phase 5 output — supplementary only, never authoritative

`generate_response_plan()` accepts an **optional** Phase 5 `GenAIResult`.
When supplied, its `recommendations` are copied verbatim into
`supplementary_ai_context` — a read-only field shown to the analyst — and
are **never** used for anything else. The exact authoritative priority
order this module follows is: (1) `incident_type`, (2) `severity`, (3)
`confidence_level`, (4) `human_review_recommended`, (5) structured
evidence, and only then, purely as supplementary context, (6) Phase 5's
recommendations. `test_genai_supplementary_context_is_advisory_only`
proves this directly: even a deliberately "disagreeing" AI
recommendation (e.g. claiming an incident is actually Normal Activity)
cannot suppress, remove, or alter a single generated step.

### Plan validation — never a crash

`validate_plan()` is a pure self-check that always returns a
`PlanValidationResult` (`is_valid`, `errors`, `warnings`) and never
raises. It checks: Normal Activity never has steps and every real
incident has at least one; step numbers are unique and form a
contiguous `1..N` sequence; no step has an empty action; any
containment-recommendation or mandatory-review step requires human
review; a LOW-confidence incident includes both an evidence-collection
step and a human-review step; no step's action contains a prohibited
offensive/automated-action keyword; and the plan's priority is one of
`config.PLAN_PRIORITY_LEVELS`. A failed validation surfaces as a clear
on-page warning (never an exception) — and, empirically, every plan
generated on the real 350-event dataset already passes validation with
zero errors.

### Streamlit integration

The **Response Planning** page reuses Phase 1-4 results exactly like the
Generative AI Analysis page (never re-running preprocessing, anomaly
detection, the rule engine, or confidence scoring), and reuses a cached
Phase 5 `GenAIResult` for the selected event/model from
`st.session_state["genai_results"]` when one exists — without ever
re-invoking Phase 5. The page shows:

- **Planning Summary** — total events, incidents with a plan, HIGH/MEDIUM
  severity counts, and how many plans require human review (via the new
  batch orchestration function `run_response_planning()`).
- **SYSTEM Analysis** — the authoritative incident type, severity,
  confidence level, and human-review flag for the selected event.
- **Response Plan** — the plan title, priority, review-required flag,
  validation status, planning reason, and the full ordered steps table
  (action / purpose / reason / priority / human review / stage).
- **Plan Flow** — a simple visualization of the 7-stage planning
  progression, marking which stages this specific plan actually
  reaches.
- A persistent **"All actions are advisory and require analyst review"**
  statement, plus the required "does not execute cybersecurity actions
  against real systems" statement shown prominently at the top of the
  page.

### Code quality

All planning logic lives in `src/response_planner.py`; all thresholds
and constants (`PLAN_PRIORITY_LEVELS`, `PROHIBITED_PLAN_ACTION_KEYWORDS`)
live in `config.py`; `app.py`'s `render_response_planning_page()` only
orchestrates calls into `src/` and renders their results. No logic is
duplicated between the six incident-specific templates (the shared tail
is built once); planning is fully deterministic (no randomness anywhere
in the module); and no new third-party dependencies were added.

### Testing Phase 6

```bash
python test_response_planner.py
```

This single command runs all Phase 6 checks (86) AND then re-runs
`test_genai.py` in full — which itself cascades through
`test_confidence.py`, `test_rule_engine.py`, `test_preprocessing.py`,
and `test_anomaly_detection.py` — so one command regression-tests the
entire pipeline built so far (433 checks total: 44 Phase 1 + 53 Phase 2
+ 61 Phase 3 + 103 Phase 4 + 86 Phase 5 + 86 Phase 6). Phase 6 checks
include: Normal Activity correctly producing no plan; each of the six
incident-specific plan templates verified against real Phase 3/4 output;
an unrecognized incident type falling back safely instead of crashing;
severity-aware planning (HIGH/MEDIUM/LOW); confidence-aware planning
(HIGH/MEDIUM/LOW); the two explicit spec examples (HIGH severity + LOW
confidence → URGENT; MEDIUM severity + HIGH confidence → a distinctly
different STANDARD plan) proving severity and confidence are never
merged; the human-review flag adding a mandatory step independently of
confidence, and correctly staying absent when nothing calls for it;
deterministic, contiguous step ordering and byte-for-byte repeatable
plan generation; every plan generated on the real 350-event dataset
passing its own validation; `validate_plan()` correctly flagging
prohibited-action keywords, structural problems (duplicate/non-
contiguous step numbers, empty actions, Normal Activity with steps, an
incident with none), and missing LOW-confidence requirements — all
without ever raising; `true_label` proven excluded both structurally and
behaviorally; Phase 5 output proven to be optional and purely
supplementary, including a deliberately disagreeing AI recommendation
that cannot change the plan; a fully empty/minimal event handled safely;
and the Streamlit page rendering the full plan with no exceptions.

---

## 14. What Phase 7 Currently Does — Human Analyst Review & Feedback Loop

Phase 7 is the **final originally-planned phase** (Phase 8, §15, is a
later optional extension). It adds two new sidebar pages -
**"Analyst Review"** and **"Feedback History"** - and a new module,
`src/feedback.py`, that completes the pipeline: a human analyst reviews
the system's complete analysis and the Phase 6 response plan for one
incident, makes the FINAL decision, and that decision is persisted to a
local SQLite database so it survives an application restart.

**Final conceptual flow:**

```
System Analysis
        ↓
AI Recommendation
        ↓
Defensive Response Plan
        ↓
Human Analyst Review
        ↓
Approve / Reject / Modify
        ↓
Persistent Feedback
        ↓
Feedback Insights
        ↓
Future Improvement
```

### Human-in-the-loop design

Every earlier phase (2-6) is advisory: an anomaly score, a rule-based
classification, a confidence estimate, an AI explanation, a suggested
response plan. None of them takes any action. Phase 7 is where a human
being finally acts on all of that - and the design makes this explicit
rather than implicit: **"The analyst's decision is the final decision for
this review"** is shown directly above the decision control on the
Analyst Review page, and every page in the project (this one included)
repeats the same statement: **"The system provides analysis and
recommendations. The analyst makes the final decision. No cybersecurity
action is executed automatically."**

### The analyst decision workflow: APPROVE / REJECT / MODIFY

For each incident, the Analyst Review page shows the complete structured
analysis before any decision is made:

- **SYSTEM Analysis** - incident type, severity, confidence score/level,
  evidence strength, human-review recommendation (Phases 3-4, read-only).
- **Knowledge-Based Analysis** - observed evidence, derived facts,
  triggered rules and their explanations (Phase 3, read-only).
- **Generative AI Analysis** - the cached case summary, why-suspicious
  explanation, supporting evidence, recommendations, confidence
  explanation, and Phase 4's uncertainty reason, **if** an AI summary was
  already generated on the Generative AI Analysis page (Phase 5, never
  re-generated here - purely reused if present, otherwise the review
  proceeds without it).
- **Response Plan** - the Phase 6 plan title, priority, ordered steps,
  planning reason, and validation status (reused via
  `generate_response_plan()`, never recomputed differently).

The analyst then chooses exactly one of:

- **APPROVE** - accepts the system's analysis/plan as appropriate.
- **REJECT** - does not accept the proposed analysis/plan.
- **MODIFY** - wants to change the plan before accepting it.

...adds free-text **analyst notes**, and submits. `incident_type`,
`severity`, and `confidence_score`/`confidence_level` are never shown as
editable anywhere on this page - they are simply not part of any input
widget, so there is no way to change them from this screen. They are
copied into the saved feedback record exactly as computed by Phases 3-4.

### Modified-plan handling

Choosing MODIFY opens a safe, per-step editor (`_render_plan_editor()`
in `app.py`) built entirely from plain, individually-testable Streamlit
widgets (text inputs, a priority dropdown, checkboxes, a number input for
ordering) rather than a single opaque grid - each control maps to exactly
one thing an analyst is allowed to change: a step's wording (action/
reason), its priority, whether it requires human review, its position
(via an editable "Order #", renumbered automatically), or whether it's
removed entirely. A dedicated "Add a defensive investigation step" form
lets the analyst introduce a brand-new step. A live preview re-renders
the candidate plan and re-validates it after every edit.

**The original, system-generated plan is never overwritten.** Both are
stored in the feedback record: `original_plan` (from Phase 6, untouched)
and `modified_plan` (the analyst's edited version) - the saved
`review_status` of `MODIFIED` makes it explicit that the final plan
differs from what the system proposed.

### Modified-plan validation - reused, not duplicated

Before a MODIFY (or, defensively, an APPROVE) decision can be saved,
`src/feedback.py`'s `validate_feedback()` reconstructs the plan as a real
`ResponsePlanResult` (`dict_to_plan()`) and calls Phase 6's own
`src.response_planner.validate_plan()` on it - **the exact same
function**, never a second implementation of plan-validity rules. This
checks step structure and ordering, that any high-impact step still
requires human review, that a LOW-confidence plan still includes evidence
collection and a review step, and - critically - that **no step contains
a prohibited offensive/automated-action keyword**
(`config.PROHIBITED_PLAN_ACTION_KEYWORDS`). If validation fails, nothing
is saved, and the page clearly lists what must be corrected; the
"Submit Decision" button is additionally disabled while the candidate
plan is invalid, so the failure is visible before the analyst even tries
to submit - two independent layers of the same safety guarantee.

### Review statuses

Every incident with a response plan starts **PENDING** (no feedback
record exists yet for it). Once an analyst submits a decision, it becomes
**APPROVED**, **REJECTED**, or **MODIFIED** - one-to-one with the analyst's
APPROVE/REJECT/MODIFY choice. PENDING is never written to the database;
it is simply the state of any incident that `get_review_status_map()`
doesn't have an entry for. An incident can be reviewed more than once
(e.g. re-reviewed later) - `get_feedback_by_incident()` returns every
past review, and the *most recent* one determines the current status.

### Persistent SQLite feedback

`src/feedback.py` uses Python's built-in `sqlite3` module - no external
database dependency - writing to `config.FEEDBACK_DB_PATH`
(`data/analyst_feedback.db`, a file completely separate from the
checksummed dataset CSV). The database and its `feedback` table are
created automatically the first time any feedback function runs
(`CREATE TABLE IF NOT EXISTS`, called defensively by every public
function, not just an explicit "setup" step) - the application never
fails just because the file doesn't exist yet. Every public function
(`save_feedback`, `load_feedback`, `get_feedback_by_incident`,
`initialize_feedback_database`) catches `sqlite3.Error`/`OSError` and
returns a safe, empty/failure result instead of raising, and a duplicate
`feedback_id` (a primary-key collision) is caught and reported as a clear
error rather than silently overwriting a previous analyst's decision or
crashing. A stored feedback record holds: `feedback_id`, `incident_id`,
`timestamp`, `incident_type`, `severity`, `confidence_level`,
`confidence_score`, `human_review_recommended`, `analyst_decision`,
`review_status`, `original_plan` (JSON), `modified_plan` (JSON, nullable),
`analyst_notes`, `triggered_rule_ids` (JSON), `supporting_indicators`
(JSON), `ai_model_name`, and `ai_summary_source` - enough to answer "what
did the system recommend?", "what did the analyst decide?", "did they
modify it?", and "why?" from the stored row alone. **`true_label` has no
field here at all** - structurally impossible to store, exactly like
every earlier phase's structured result.

### Stable incident IDs

`build_incident_id(event_index)` derives `INC-0000`, `INC-0001`, ... directly
from an event's row index in the (checksum-verified, never-regenerated)
dataset. This is deterministic, not just "stable during a session":
every upstream stage - preprocessing, the seeded Isolation Forest, the
rule engine, confidence scoring - is fully deterministic, so the same row
index produces the same classification on every run, today or after a
full application restart.

### Feedback History and Feedback Insights

The **Feedback History** page loads every stored decision straight from
SQLite (never from `st.session_state`) and lets the analyst inspect any
record's original plan, its analyst-modified plan (if any), and its
notes side by side. A **Feedback Insights** section computes purely
descriptive statistics from the stored data: total reviews, approval/
rejection/modification counts and rates, the most common incident types
reviewed, and simple keyword counts from analyst notes (a small
hand-written stopword list, no NLP dependency).

### Continuous-improvement preparation - and why automatic retraining is NOT implemented

The pipeline's final conceptual step is "Future Improvement," but Phase 7
deliberately stops at **Feedback Insights**, not automatic learning.
Nothing in this project changes the Isolation Forest, the rule engine's
thresholds, the confidence weights, or any prompt based on stored
feedback - `summarize_feedback()` only ever reads and counts; it never
writes back into any earlier phase's configuration or model. This is a
deliberate, honest boundary for an academic project: automatically
retraining or re-tuning a security system from a handful of analyst
decisions - without safeguards like held-out validation, drift detection,
or a review process for the retraining itself - would be a much larger,
higher-risk undertaking than this project's scope, and pretending it was
implemented would overstate what a semester project can responsibly
claim. Feedback Insights exists specifically to make future,
deliberately-designed continuous improvement possible without
prematurely automating it.

### Why the analyst remains the final decision-maker

This mirrors the project's defensive-only philosophy from Phase 0
onward: the system's job is to analyze and recommend, never to act.
Giving an algorithm the authority to approve its own recommendation, or
to automatically execute a response plan, would cross exactly the line
this project has avoided at every phase (no offensive functionality, no
real action execution). A human analyst - accountable, informed by the
complete structured analysis above, and free to reject or modify anything
- is the only thing in this pipeline that ever finalizes a decision.

### Security / advisory limitations

- **Still advisory end-to-end.** Even after a human APPROVE, nothing is
  executed against a real system - "approved" means "the analyst agrees
  this plan is a reasonable set of recommendations," not "the blocking
  rule has been deployed."
- **The MODIFY editor is deliberately simple.** Reordering is done via a
  numeric "Order #" rather than drag-and-drop, and a step's `purpose`/
  `stage` fields are shown read-only rather than editable, to keep the
  interface's surface area small, predictable, and fully covered by
  automated tests - not a limitation of what Phase 6's data model could
  support.
- **One feedback record per submission, not a merge.** Re-reviewing an
  incident adds a new feedback row rather than editing a previous one -
  `get_feedback_by_incident()` shows the full history, and the most
  recent row is authoritative for "current status."
- **Keyword-based note insights, not semantic analysis.** The "common
  analyst note keywords" feature is a simple, transparent word count
  (with a small stopword list) - it can miss synonyms or context, exactly
  like Phase 5's offensive-content filter is keyword-based rather than
  semantic.

### Testing Phase 7

```bash
python test_feedback.py
```

This single command runs all Phase 7 checks (104) AND then re-runs
`test_response_planner.py` in full - which itself cascades through
`test_genai.py`, `test_confidence.py`, `test_rule_engine.py`,
`test_preprocessing.py`, and `test_anomaly_detection.py` - so one command
regression-tests the ENTIRE project (537 checks total: 44 Phase 1 + 53
Phase 2 + 61 Phase 3 + 103 Phase 4 + 86 Phase 5 + 86 Phase 6 + 104 Phase
7). Every database-touching Phase 7 test uses its own temporary SQLite
file (never the real `data/analyst_feedback.db`), so running this suite
never creates or pollutes real analyst feedback. Phase 7 checks include:
database and table auto-creation; APPROVE/REJECT/MODIFY all saving
correctly, with the modified plan proven stored separately from the
untouched original; invalid decisions and other malformed/empty input
handled safely; real file-based persistence proven by reopening the
database with an independent `sqlite3` connection; multiple records and
retrieval by incident ID; all four review statuses; a modified plan
proven to require Phase 6's own `validate_plan()` to pass before it can
be saved; prohibited actions proven impossible to save; `true_label`
proven excluded structurally and behaviorally; incident-ID determinism
proven both directly and via two independent pipeline runs; analyst notes
round-tripping exactly, including multi-line text and punctuation;
feedback summary metrics (counts, rates, keyword counts) verified against
hand-computed expected values; an empty database and simulated database
connection failures both handled without raising; and the real Streamlit
UI exercised end-to-end for APPROVE, REJECT, and MODIFY (including a
MODIFY where the "Submit Decision" button is proven disabled, and a
click on it proven to still save nothing, when the edited plan contains a
prohibited action).

Note: booting the full application (as several regression tests across
every phase do) triggers Phase 7's automatic, empty-table database
creation as a normal side effect of `app.py` starting up - this is the
intended "create the database automatically the first time the
application runs" behavior (Requirement #7), not test pollution; no
regression test ever writes an actual feedback row to that file.

---

## 15. What Phase 8 Currently Does — RAG / Knowledge Base Integration

Phase 8 is an **optional extension** built after the seven originally
planned phases were complete. It fills in the `retrieved_knowledge` hook
that `src/genai_analyzer.IncidentContext` always had (see §12) with a
real, working - but deliberately lightweight - Retrieval-Augmented
Generation (RAG) layer. It changes no other phase's logic: Phases 1-7
behave exactly as before, and this phase adds no new Streamlit page.

### Why RAG was added

The Generative AI summary (Phase 5) explains an incident using only the
structured evidence Phases 2-4 already produced. That is accurate, but a
human analyst often also wants a little general background - "what does
a brute-force pattern usually mean, and what's the standard way to
triage it?" - without leaving the page. RAG supplies exactly that: a
handful of short, curated, defensive reference notes relevant to the
current incident, shown transparently and handed to the AI purely as
background context.

### How retrieval works: TF-IDF + cosine similarity

`src/retriever.py` implements retrieval with nothing more than
`sklearn.feature_extraction.text.TfidfVectorizer` and
`sklearn.metrics.pairwise.cosine_similarity` - the same `scikit-learn`
install Phase 1/2 already require. There is **no vector database, no
embeddings model, and no external service** - `chromadb` has been
removed from `requirements.txt` entirely.

1. **Load** every `*.json` file under `knowledge_base/`, sorted by
   filename for a fully deterministic order (`load_knowledge_base()`).
2. **Build a query** from the current incident's already-computed
   Phase 3/4 results only - `incident_type`, `severity`, derived facts,
   triggered-rule names/conclusions/explanations, supporting indicators,
   and the confidence level/uncertainty reason (`build_retrieval_query()`).
   Never the raw dataframe row, and never `true_label` - neither
   `InferenceRecord` nor `ConfidenceRecord` has such a field to begin
   with.
3. **Vectorize** the query and every knowledge entry's `title + topic +
   content` together with one shared `TfidfVectorizer`, then compute
   cosine similarity between the query vector and every entry vector.
4. **Rank and filter**: keep entries at or above `config.RAG_MIN_SIMILARITY`
   (0.05), sort by similarity score descending (ties broken by
   `knowledge_id` for full determinism), and keep at most
   `config.RAG_TOP_K` (3) entries.
5. **Return** a `RetrievalResult` - always safe to inspect, even when
   nothing matched (`items` is simply `[]`, never `None` or an
   exception).

Because TF-IDF term counts and cosine similarity involve no randomness,
retrieval is **fully deterministic**: the same incident always retrieves
the same entries, in the same order, with the same scores - verified
directly in `test_retriever.py`.

### Knowledge base structure

`knowledge_base/` holds 10 small, hand-written JSON reference documents,
one topic each, covering: brute-force login activity, unusual network
activity, port scanning, suspicious data transfer / possible
exfiltration, suspicious login location / impossible travel, off-hours
privilege activity, multi-device anomalous activity, incident triage and
analyst review, defensive incident response, and security logging /
monitoring. Each file has exactly five fields:

```json
{
  "knowledge_id": "KB-001",
  "title": "Brute-Force Login Activity",
  "topic": "Authentication Attacks",
  "incident_types": ["Possible Brute Force"],
  "content": "…a few sentences of defensive guidance…",
  "source_type": "defensive_reference"
}
```

Every entry is deliberately concise, academic/demo-appropriate, and
strictly defensive - none contains offensive instructions, exploit
steps, credential-attack techniques, or guidance for attacking a real
system. `test_retriever.py` checks every single entry against the same
prohibited-keyword lists used elsewhere in the project
(`config.PROHIBITED_PLAN_ACTION_KEYWORDS` and
`src.genai_analyzer.OFFENSIVE_KEYWORDS`) to prove this, not just assert
it in prose.

### How retrieved knowledge reaches the GenAI module

`app.py`'s Generative AI Analysis page runs retrieval automatically for
the selected incident (no extra button) and shows it under a
**"📚 Retrieved Knowledge (RAG)"** section - knowledge ID, title,
similarity score, and the short content snippet for each match - placed
between **SYSTEM Analysis** and **Generative AI Case Summary**, so the
on-screen flow is: *Incident → Retrieved Knowledge → AI Analysis →
Recommendations*. When the analyst clicks **Generate AI Case Summary**,
that same retrieval result is passed into
`build_incident_context(..., retrieved_knowledge=…,
retrieved_knowledge_items=…)`, which stores it on `IncidentContext` both
as flat text (embedded in the prompt as a `retrieved_knowledge` field)
and as structured metadata. The system prompt explicitly instructs the
model to treat it as background only - never as instructions to follow,
never able to override or invent `incident_type`/`severity`/
`confidence_level` - and `GenAIResult` is extended with
`retrieved_knowledge_ids` / `_titles` / `_scores` / `_snippets` /
`_count` so the page can also show, right under the AI summary, exactly
which sources that particular analysis drew on.

### Fallback behavior

RAG is never a hard dependency. If the knowledge base is missing, empty,
or nothing meets the similarity threshold, retrieval simply returns an
empty result and the page says so - the AI analysis proceeds normally
with an empty `retrieved_knowledge`. Independently, Ollama being
unavailable (as it is in this sandboxed environment) makes
`generate_ai_case_summary()` fall through to the same deterministic,
evidence-based fallback Phase 5 always used - and that fallback now
*also* carries whatever was retrieved onto its `GenAIResult`, so
retrieved knowledge is still shown even when the AI text itself is the
deterministic fallback, exactly as required.

### Safety limitations

RAG only ever adds **reference text to read**, never an action to take.
Nothing about it can execute a cybersecurity action, and none of the
existing safeguards were loosened to add it:

- The knowledge base itself is hand-written and defensive-only (verified
  by `test_retriever.py`), so there is nothing offensive for the model to
  echo.
- The system prompt's existing rule that the model must respect the
  supplied `incident_type`/`severity`/`confidence_level` exactly, plus a
  new explicit rule about retrieved knowledge, means retrieved text can
  never override an authoritative value - proven in
  `test_retriever.py`'s authoritative-value tests, which attach retrieval
  to several different incidents and confirm the system's own
  classification, severity, and confidence are always restated verbatim
  regardless of what was retrieved.
- The existing keyword-based output-safety net
  (`_contains_offensive_content()` / `_enforce_output_safety()`) still
  runs on the AI's final text exactly as in Phase 5, unchanged.
- Retrieval can only ever shrink to nothing (an empty result) - it can
  never inject an incident, a rule, or a fact that Phases 2-4 didn't
  already produce.

### Testing Phase 8

```bash
python test_retriever.py
```

This single command runs all Phase 8 checks (659) and then re-runs
`test_feedback.py` in full - which itself cascades through every earlier
phase's suite - so one command regression-tests the ENTIRE project
(1,196 checks total: 44 Phase 1 + 53 Phase 2 + 61 Phase 3 + 103 Phase 4 +
86 Phase 5 + 86 Phase 6 + 104 Phase 7 + 659 Phase 8). Phase 8 checks
include: the real knowledge base loading successfully with every
required topic present; fully deterministic retrieval (identical IDs,
order, and scores across repeated calls); a relevant query returning the
topically-correct entry first for every incident type; `top_k` and
`min_similarity` both behaving correctly at their limits; empty/blank/
stopword-only queries and a missing/empty/malformed knowledge base all
handled safely with zero items rather than a crash; similarity scores
always validated within `[0, 1]`; `true_label` proven excluded both
structurally (not imported into the module's namespace, not a field on
any Phase 8 dataclass, not a parameter of the query builder) and
behaviorally (flipping it upstream never changes retrieval); every
knowledge-base entry checked against the project's own
offensive/prohibited-keyword lists; retrieved knowledge proven to reach
`IncidentContext` and then the outgoing `GenAIResult`'s new fields
correctly; the deterministic fallback proven to keep working - with
retrieval metadata still attached - when Ollama is unavailable;
authoritative incident type, severity, and confidence proven unchanged
across several incident types even with retrieval attached; no
`chromadb`/heavy dependency introduced; a full pipeline integration pass
over real dataset rows; and the real Streamlit "Generative AI Analysis"
page exercised end-to-end, confirming the "Retrieved Knowledge (RAG)"
section renders and that generating a summary with retrieval attached
raises no exception.

As with Phase 7, booting the full application during regression testing
still triggers the harmless, empty, automatic feedback-database creation
described in §14 - not a Phase 8 concern, just the same documented side
effect appearing again because these tests also boot `app.py`.

---

## 16. What Phase 9 Currently Does — Continuous Improvement from Analyst Feedback

Phase 9 is a later, optional extension built after Phase 8. It does not
touch detection, classification, confidence scoring, RAG, or response
planning at all - it only reads the analyst feedback that Phase 7 already
collects and turns recurring patterns in that feedback into structured,
traceable suggestions for a **human developer to review later**. It adds
no new Streamlit page: everything Phase 9 shows lives at the bottom of
the existing **Feedback History** page, which is otherwise unchanged.

### Why analyst feedback is useful beyond a single incident

Phase 7 already lets an analyst APPROVE, REJECT, or MODIFY the system's
generated response plan for one incident, and stores that decision. Any
single review is just one analyst's judgment on one incident. But once
enough reviews accumulate, patterns across them - the same incident type
keeps getting REJECTed, the same response-plan step keeps getting
reworded, the same word keeps showing up in analyst notes - are a signal
that something about the system's *output*, not any one incident, may be
worth revisiting. Phase 9 exists purely to surface those patterns
clearly, with evidence, so a human can decide what (if anything) to do
about them.

### How feedback patterns are extracted

`src/continuous_improvement.py` is the whole of Phase 9. It reuses Phase
7's own `src.feedback.load_feedback()` and `summarize_feedback()` for the
base counts/rates rather than re-implementing any database or counting
logic, and adds new, purely descriptive analysis on top:

1. **Incident-type breakdown** - for every `incident_type` seen in
   feedback, how many reviews it received and how they split across
   APPROVE/REJECT/MODIFY, plus a **disagreement rate**
   (`(REJECT + MODIFY) / total`) that highlights incident types analysts
   keep pushing back on.
2. **Repeated response-plan step modifications** - every MODIFY record's
   `original_plan` and `modified_plan` (both already stored by Phase 7)
   are compared step-by-step, aligned by `step_number`, to detect wording
   changes, removed steps, and added steps. Diffs are then grouped by
   `(incident_type, original_action)` - the exact, already-existing
   system-generated step text - so a "frequently modified step" always
   names a real template string from `src/response_planner.py`, never an
   invented category.
3. **Recurring analyst-note themes** - analyst notes are tokenized with
   the same word-splitting/stopword logic Phase 7's own note-keyword
   summary already uses, deduplicated per note (so one note repeating a
   word only counts once), and any word appearing in enough *distinct*
   notes is reported as a theme.
4. **Improvement suggestions** - built strictly from the three analyses
   above, in three fixed categories: incident types with repeated
   disagreement, specific response-plan steps analysts keep rewriting,
   and recurring analyst-note themes.

Nothing here is a statistical model or a learned pattern - it is
straightforward, deterministic counting and grouping over data that
already exists.

### The minimum-occurrence rule: never a suggestion from one review

Every pattern above - a disagreement, a modified step, a note theme -
only counts as "repeated"/"frequent" once it recurs across at least
`config.CI_MIN_PATTERN_OCCURRENCES` (2) **distinct** feedback records. A
single one-off REJECT, a single reworded step, or a word used in only one
note never generates a suggestion by itself. This is enforced
structurally in the grouping/aggregation code, not just asserted in
prose, and is checked directly in `test_continuous_improvement.py` by
seeding a genuine one-off pattern alongside a genuine repeated one and
confirming only the repeated one produces a suggestion.

### Traceability: every suggestion names its evidence

Every `ImprovementSuggestion` - and every `IncidentTypeBreakdown`,
`FrequentStepModification`, and `NoteTheme` feeding into it - carries the
exact `incident_id`s and `feedback_id`s it was derived from, for example:

```
Suggested Improvement: "Review the wording of the 'Review the
transferred volume and event details' step for Suspicious Data
Transfer."
Supporting incidents: INC-0007 / INC-0012
Supporting feedback IDs: FB-000007 / FB-000012
```

A suggestion is only ever produced when real, stored feedback backs it -
`test_continuous_improvement.py` cross-checks every suggestion's
supporting IDs against the actual feedback records that exist, to prove
nothing is fabricated.

### Human review: insight, never automatic learning

Phase 9 is intentionally an **offline, human-supervised** reporting step,
not a learning system:

```
Analyst Feedback → Pattern Detection → Improvement Suggestion
        → Human / Developer Review → Manual Future Change
```

`src/continuous_improvement.py` only ever calls
`src.feedback.load_feedback()` (a read). It never calls `save_feedback()`
or any other write path, and it imports nothing from
`src.anomaly_detector`, `src.rule_engine`, `src.confidence`,
`src.retriever`, or `src.response_planner` - there is no import path here
that could change a threshold, a rule, a model, the RAG configuration, or
response-planning logic, and nothing in this module executes a
cybersecurity action of any kind. `test_continuous_improvement.py` proves
this structurally, by asserting those functions are absent from the
module's own namespace, not just by describing the intent in prose.

The Continuous Improvement Insights section on the Feedback History page
carries this notice verbatim:

> **These insights are recommendations for future system improvement.
> They do not automatically modify the system.**

Every existing Phase 7 Feedback History section - Feedback Insights, the
feedback table, and Inspect a Record - is unchanged; Phase 9 only adds a
new section underneath them.

### Why automatic self-tuning/retraining is intentionally not implemented

An AI-assisted defensive tool that silently re-tunes its own thresholds,
rewrites its own rules, or retrains its own model based on a small,
informal stream of analyst feedback would be difficult to audit, easy to
push in an unintended direction with only a handful of reviews, and
outside the scope of a course project. Keeping the human in the loop -
Phase 9 only ever *describes* a pattern and lets a developer decide by
hand whether and how to act on it - is a deliberate design choice, not a
missing feature.

### Testing Phase 9

```bash
python test_continuous_improvement.py
```

This runs all 102 Phase 9 checks directly, without cascading into the
Phase 1-8 regression suite (unlike `test_retriever.py`, which does chain
into the full 1,196-check regression - see §15). Phase 9 checks include:
an empty database and a single feedback record both handled safely; the
exact APPROVE/REJECT/MODIFY counts and rates for a realistic multi-record
scenario; correct incident-type aggregation and disagreement rates;
correct detection of a genuinely repeated plan-step modification *and*
correct exclusion of a one-off modification; correct detection of a
recurring note theme *and* correct exclusion of a one-off note word;
suggestions generated across all three categories with sequential
`CI-001`, `CI-002`, ... IDs; every suggestion's supporting incident/
feedback IDs verified to exist in the real underlying feedback;
deterministic output across repeated calls and across a fresh database
reload; malformed/missing feedback data (a non-dict plan, a missing
`incident_type`, an unexpected decision value) handled without a crash;
a broken/inaccessible database path handled without a crash; a
structural proof that no write function or earlier-phase detection/rule/
confidence/RAG/response-planning function is imported into this module;
a structural and behavioral proof that `true_label`/`config.LABEL_COLUMN`
is never used; the dataset file confirmed byte-for-byte unchanged after
generating a report; and the real Streamlit Feedback History page
exercised end-to-end, confirming the new Continuous Improvement Insights
section renders alongside the unchanged Phase 7 sections with no
exception.

A full project regression (`python test_retriever.py`, 1,196 checks) can
still be run at any time and remains unaffected by Phase 9 - it was
intentionally not re-run repeatedly during Phase 9 development, per the
same "test what changed" practice used throughout this project.

---

## 17. What Phase 10 Currently Does — Explainability & Audit Trail

Phase 10 is a later, optional extension built after Phase 9. It adds no
new detection, classification, confidence, RAG, GenAI, or response-plan
logic at all - every number, label, and piece of retrieved text it shows
was already produced by an earlier phase. Phase 10 only *explains* those
existing results in plain language and *records* what happened, stage by
stage, in a permanent, append-only log. It adds one new Streamlit page,
**Explainability & Audit Trail**, bringing the app to 10 pages total.

### Why explainability is needed

Phases 2-9 make several judgment calls per incident - is this event
anomalous, which rule(s) fired, how severe is it, how confident is the
system, which knowledge-base entries are relevant, what should the GenAI
summary say, what response plan follows. An analyst reviewing an incident
on the existing pages sees the *outputs* of those calls, but not always
*why* the system landed on them. Phase 10 closes that gap: for any
selected incident it answers, in the user's own words, "why anomalous or
normal", "which rules contributed", "how was severity decided", "how was
confidence decided", "which knowledge sources were retrieved", "what was
given to the GenAI module", "what response plan resulted", and "what did
the analyst ultimately decide" - without changing any of those answers.

### Evidence traceability: explain, never recompute

`src/explainability.py` is built around one constraint: it must explain
*existing* decisions, never make new ones. It imports only the dataclass
*types* already produced by earlier phases - `InferenceRecord`,
`ConfidenceRecord`, `RetrievalResult`, `GenAIResult`,
`ResponsePlanResult` - and never imports a single compute function
(`run_inference`, `run_rule_engine`, `score_confidence`,
`retrieve_knowledge_for_incident`, `generate_ai_case_summary`,
`generate_response_plan`, `train_isolation_forest`, and so on are all
structurally absent from the module). Every explanation sentence is built
by interpolating fields that already exist on those records - for
example, `explain_rule_evidence()` reuses each triggered rule's own
`explanation` string verbatim rather than writing a new one, and
`explain_classification()` reuses `InferenceRecord.reason`, which
`src/rule_engine.py` already built in Phase 3. Nothing in this module is
allowed to invent a reason that isn't backed by a real stored value; this
is verified directly in `test_explainability.py`, which replays 60 real
dataset rows through the full pipeline and confirms every rule ID,
indicator, and knowledge ID the module reports exactly matches the real
underlying record.

`src/explainability.py` builds eleven small, focused dataclasses -
covering anomaly detection, rule evidence, classification, severity,
confidence, supporting indicators, uncertainty, RAG retrieval, GenAI
generation, the response plan, and human review - and one bundling
`IncidentExplanation` that the Streamlit page renders directly.

### Distinguishing system facts, RAG context, AI text, and the human decision

A recurring risk in any GenAI-assisted tool is a reader mistaking
retrieved reference material, or the model's own written summary, for an
authoritative system decision. The Explainability page keeps these four
categories visually and structurally separate, with an `st.caption()`
label on every section:

1. **SYSTEM-GENERATED FACT** - the anomaly result, triggered rules,
   classification, severity, confidence score/level, and supporting
   indicators. These come directly from `InferenceRecord` /
   `ConfidenceRecord` and are the same values shown on the earlier Phase
   2-4 pages.
2. **RAG REFERENCE MATERIAL** - the knowledge-base entries retrieved for
   this incident (knowledge ID, title, similarity score). This is
   background context the GenAI module was given, never itself a
   decision.
3. **GENERATIVE AI TEXT** - the model's own written summary/analysis,
   labeled with which model produced it and whether Ollama or the
   deterministic fallback generated it. `GenAIExplanation` deliberately
   sources its `authoritative_incident_type` /
   `authoritative_severity` / `authoritative_confidence_level` fields
   only from `InferenceRecord` / `ConfidenceRecord`, never from the
   GenAI text itself (`GenAIResult` has no such fields to begin with),
   so the written summary can never quietly override the real
   classification.
4. **HUMAN ANALYST DECISION** - the APPROVE/REJECT/MODIFY decision and
   notes recorded on the Analyst Review page. This is the only category
   that represents an actual decision about what to do; everything above
   it is either a system fact or reference material feeding into a
   recommendation.

### Audit event structure and the audit trail database

`src/audit_trail.py` uses its own SQLite database, `data/audit_trail.db`
(`config.AUDIT_DB_PATH`) - a completely separate file from Phase 7's
`data/analyst_feedback.db` (`config.FEEDBACK_DB_PATH`). Phase 7's
feedback schema is untouched. Each row is an `AuditEvent`: `audit_id`,
`incident_id`, `timestamp`, `stage`, `event_type`, plus whichever of
`incident_type`, `severity`, `confidence_level`, `confidence_score`,
`relevant_rule_ids`, `supporting_indicators`, `retrieved_knowledge_ids`,
`ai_model_name`, `human_review_recommended`, `analyst_decision`, and
`descriptive_details` apply to that stage. `stage` is restricted to the
fixed, ordered list in `config.AUDIT_STAGES` - `INCIDENT_SELECTED`,
`ANOMALY_DETECTION`, `RULE_CLASSIFICATION`, `CONFIDENCE`,
`RAG_RETRIEVAL`, `GENAI_ANALYSIS`, `RESPONSE_PLAN`, `ANALYST_REVIEW` -
so a typo can never silently create an untracked stage. `true_label` /
`config.LABEL_COLUMN` is never stored, and no raw dataframe row is ever
copied wholesale into the audit table - only the structured,
already-derived fields above, in keeping with data minimization.

### The audit timeline is append-only

`write_audit_event()` only ever `INSERT`s a fresh row (a new
`audit_id`); there is no update or delete path anywhere in the module.
For an incident's deterministic pipeline stages (selecting the incident,
running anomaly detection, rule classification, confidence, RAG
retrieval, GenAI generation, and the response plan), each stage is
recorded once via a `skip_if_exists=True` existence check, so re-running
the same Streamlit page (which reruns the whole script on every widget
interaction) never produces duplicate rows. `ANALYST_REVIEW` events are
the exception: they are always appended (`skip_if_exists=False`), because
a real analyst can genuinely revisit an incident and record a second,
different decision later, and both should remain visible. A full trace
for one incident therefore reads, in order:

```
INC-0007: ANOMALY_DETECTION → RULE_CLASSIFICATION → CONFIDENCE
          → RAG_RETRIEVAL → GENAI_ANALYSIS → RESPONSE_PLAN → ANALYST_REVIEW
```

`get_events_by_incident()` returns this trace in chronological order;
`summarize_incident_audit_history()` additionally reports which stages
have been recorded (sorted by the canonical `AUDIT_STAGES` order
regardless of the order they were written in) and the most recent
analyst decision, if any. The Explainability & Audit Trail page renders
this as a ✅/⬜ checklist against all eight stages plus the full event
table underneath.

### Why the audit trail never alters a decision

The audit trail is strictly observational. `app.py`'s
`_log_case_processing_audit()` helper is called *after* an incident has
already been run through anomaly detection, the rule engine, confidence
scoring, RAG retrieval, GenAI generation, and the response planner on the
existing Phase 2-8 pages - it only reads the already-computed
`InferenceRecord`/`ConfidenceRecord`/`RetrievalResult`/`GenAIResult`/
`ResponsePlanResult` objects (several of which are cached in
`st.session_state` by those pages specifically so the new Explainability
page can read them too, without ever calling `retrieve_knowledge_for_incident()`
or `generate_response_plan()` a second time) and writes a record of what
those results were. Nothing in `src/audit_trail.py` or
`src/explainability.py` can change a threshold, a rule, a confidence
score, a retrieved document, a GenAI summary, or a response plan, and
neither module executes a cybersecurity action of any kind -
`test_audit_trail.py` proves this structurally, by asserting that
`save_feedback`, `initialize_feedback_database`, and every earlier-phase
compute function are absent from both modules' namespaces.

### Testing Phase 10

```bash
python test_explainability.py
python test_audit_trail.py
```

`test_explainability.py` runs 166 checks covering: normal and anomalous
incident explanations, rule-evidence explanation, classification,
severity, confidence, supporting-indicators, and uncertainty
explanations; RAG explanation with and without a retrieval result;
GenAI explanation with and without a GenAI result, confirming the
authoritative incident type/severity/confidence always come from
`InferenceRecord`/`ConfidenceRecord` and never from the GenAI text;
response-plan explanation with and without a plan; human-review
explanation given `None`, a dict, and a pandas Series; an end-to-end
`build_incident_explanation()` check; a 60-row sweep proving no
unsupported or invented evidence ever appears; a structural and
behavioral proof that `true_label`/`config.LABEL_COLUMN` is never used
and that no compute function from any earlier phase is imported into the
module; and the real Streamlit Explainability page exercised end-to-end
with no exception.

`test_audit_trail.py` runs 92 checks covering: database auto-creation;
single and multiple event creation; retrieval by incident (chronological
order) and by stage; append behavior (two genuinely different analyst
decisions both persist, while a would-be duplicate deterministic-stage
event correctly no-ops); persistence across a fresh database reconnect;
malformed input (empty incident ID, invalid stage, empty event type) and
a broken database path both handled without a crash; an empty database
handled safely; a structural and behavioral proof that `true_label` is
never used and that no earlier-phase decision function or Phase 7 write
function is reachable from this module; deterministic incident IDs
reused from Phase 7's own `build_incident_id()`; `summarize_incident_audit_history()`
returning stages in canonical order regardless of write order and the
correct most-recent analyst decision; and the real Streamlit app exercised
end-to-end, confirming an analyst decision produces exactly one new
`ANALYST_REVIEW` row and that repeated reruns never duplicate a
deterministic stage.

Neither test file cascades into the Phase 1-9 regression suite, per the
same "test what changed" practice used throughout this project. A full
project regression (`python test_retriever.py`, 1,196+ checks) remains
available and unaffected by Phase 10, and was intentionally not re-run
repeatedly during Phase 10 development.

---

## 18. What Phase 11 Currently Does — Evaluation & Performance Dashboard

Phase 11 is a later, optional extension built after Phase 10. It adds no
new detection, classification, confidence, RAG, GenAI, or response-plan
logic, and it does not add a new decision-making step anywhere - it only
*measures* how the system Phases 2-10 already built is behaving, using
either already-computed results or, for a small number of descriptive
counts, a single legitimately-available ground-truth signal read for
that purpose alone. It adds one new Streamlit page, **Evaluation &
Performance**, bringing the app to 11 pages total.

### Offline evaluation, not automatic optimization

Every earlier phase's threshold, weight, or model parameter in this
project was chosen up front from plain domain reasoning (see the
comments in `config.py`) and has never been tuned by looking at
`true_label` or any evaluation metric. Phase 11 does not change that: the
Evaluation & Performance page is strictly descriptive and analytical. It
cannot change the Isolation Forest's parameters, a rule threshold, a
confidence weight, the RAG top-k/similarity cutoff, the GenAI model, a
response-planning rule, or any analyst decision - `src/evaluation.py`
has no write path into any of those modules, the same way
`src/continuous_improvement.py` (Phase 9) and `src/explainability.py` /
`src/audit_trail.py` (Phase 10) do not.

### Measured performance metrics, built from what already exists

`src/evaluation.py` follows the same "reuse, never recompute" discipline
as Phase 10: it does not re-implement accuracy/precision/recall/F1, does
not re-run the rule engine's own logic, and does not re-derive confidence
scores. Instead it:

- **Anomaly detection** - REUSES `src.anomaly_detector.evaluate_against_ground_truth()`,
  the same, already-existing, already-tested post-hoc evaluation function
  used since Phase 2 (and already shown on the AI Anomaly Detection
  page), for accuracy/precision/recall/F1/confusion matrix. Predicted
  normal/anomalous counts come straight off the already-computed
  `AnomalyDetectionResult` and never require `true_label` at all.
- **Incident / classification analysis** - descriptive aggregation
  (`value_counts()`-style grouping) directly over Phase 3's own
  `RuleEngineResult.results_df` - incident-type distribution, severity
  distribution, per-rule firing counts, and how many events triggered
  2+ rules.
- **Confidence & human review** - a direct pass-through of the
  HIGH/MEDIUM/LOW counts and the human-review-recommended count Phase
  4's `ConfidenceEngineResult` already computed.
- **RAG performance** - knowledge-base size from
  `src.retriever.load_knowledge_base()`, plus retrieval statistics
  gathered by calling `retrieve_knowledge_for_incident()` once per
  anomalous event - the exact same function every other page already
  calls one incident at a time, just applied in a simple loop for
  aggregation. A **controlled retrieval success rate** is also reported:
  each knowledge-base entry is hand-tagged in Phase 8 with an
  `incident_types` list, so a top match "succeeds" when the incident's
  own classified type (from Phase 3, never `true_label`) appears in that
  entry's tag list - a legitimately-available check, not an invented one.
- **Generative AI performance** - the configured model name and live
  Ollama/model availability come from `get_ollama_status()` (already
  used on the Generative AI Analysis page). Attempted/successful/
  fallback generation counts and the source breakdown are parsed
  read-only from this project's own audit trail (`GENAI_ANALYSIS`
  events Phase 10 already records) - they reflect GenAI analyses
  actually performed and recorded so far, never an estimate.
- **Analyst feedback** - a direct pass-through of Phase 7's own
  `summarize_feedback()` (the same function powering Feedback History's
  "Feedback Insights" section).
- **Audit activity** - Phase 10's own audit trail, grouped by
  `config.AUDIT_STAGES` for a per-stage event count.

### Do not fabricate metrics

Every metric above is either read directly off an already-computed
result object or derived from real, already-recorded data (the audit
trail, the feedback database). Wherever a metric genuinely cannot be
computed - no `true_label` column, an empty audit trail, no analyst
feedback recorded yet, Ollama not running - the corresponding result
carries an explicit `measured=False` (or an equivalent flag) and a clear
note explaining why, and the page shows that as "unavailable/not
measured" rather than a zero or invented value that could be mistaken
for a real one. `test_evaluation.py` checks this directly: an isolated,
empty audit/feedback database always yields `0`/`False` with a note, and
a dataset stripped of `true_label` always yields `None` accuracy/
precision/recall/F1, never a fabricated number.

### The true_label rule: the first phase allowed to read it, and only here

Every earlier phase - preprocessing, anomaly detection, the rule engine,
confidence scoring, RAG, GenAI, response planning, analyst review,
feedback storage, continuous improvement, explainability, the audit
trail - has never read `true_label` as an input to a decision. Phase 11
formalizes and extends the one narrow exception that has existed since
Phase 2 (`evaluate_against_ground_truth()`, already used on the AI
Anomaly Detection page for post-hoc evaluation) into a dedicated module
and dashboard, under an explicit rule:

```
SYSTEM INPUT / DECISION PIPELINE:  true_label MUST NOT be used.
EVALUATION MODULE (src/evaluation.py):  true_label MAY be read
                                         solely to calculate offline
                                         metrics.
```

`true_label` is read in exactly two places in this codebase, both inside
this narrow allowance: `evaluate_against_ground_truth()` in
`src/anomaly_detector.py` (Phase 2), and `evaluate_dataset_overview()` in
`src/evaluation.py` (Phase 11), which reports the dataset's ground-truth
split for display only. Every metric derived from it is labeled
**"Offline Evaluation"** wherever it is shown on the Evaluation &
Performance page. `test_evaluation.py` proves the boundary both
structurally (`LABEL_COLUMN` is absent from every other module's
namespace) and behaviorally (flipping `true_label` upstream never
changes an anomaly prediction or a rule-engine classification - only the
evaluation metrics that are supposed to read it change).

### Local performance timing, not a production benchmark

Where timing can be measured cheaply, deterministically, and without a
network dependency - training + predicting with Isolation Forest on the
full dataset, and RAG retrieval over every anomalous event - the
Evaluation & Performance page measures it fresh on every page load and
reports it plainly labeled as a local-machine sample. GenAI generation
timing is different: it may call a real, possibly slow, local Ollama
server, so it is deliberately **opt-in only** - a "Measure a sample GenAI
generation time" button runs exactly one real
`generate_ai_case_summary()` call and times it, rather than running
automatically. None of the three timing functions changes any existing
AI behavior; each only wraps an already-existing function call with a
timer, and every result is captioned "not a production benchmark" since
timing depends entirely on the machine running the app.

### Testing Phase 11

```bash
python test_evaluation.py
```

This runs 137 Phase 11 checks directly, without cascading into the
Phase 1-10 regression suite (the same "test what changed" practice used
by every optional extension in this project). Checks include: dataset
overview counts cross-checked by hand; anomaly metrics proven to match
`evaluate_against_ground_truth()` exactly, plus confusion-matrix cell
sums and valid-rate bounds; safe, non-fabricated handling when
`true_label` is absent; incident-type/severity/rule-firing counts
cross-checked by hand against `RuleEngineResult.results_df`; confidence
counts proven to pass through `ConfidenceEngineResult` unchanged; RAG
metrics checked for internal consistency and determinism across repeated
calls; GenAI metrics proven to parse real, written `GENAI_ANALYSIS`
audit events correctly (and to report "unmeasured" safely when none
exist); feedback metrics checked against hand-seeded APPROVE/REJECT
records; audit metrics grouped correctly by stage; a structural and
behavioral proof that `true_label` is confined to this module (and
`src/anomaly_detector.py`'s pre-existing post-hoc function) and never
changes an anomaly prediction or rule classification; empty/missing-data
handling for every section; local timing measurements proven to
complete safely (including on empty input); and the real Streamlit
Evaluation & Performance page exercised end-to-end - all nine sections
render, at least one table renders, and the optional GenAI timing button
works - with no exception.

A full project regression (`python test_retriever.py`, 1,196+ checks)
remains available and unaffected by Phase 11, and was intentionally not
re-run repeatedly during Phase 11 development.

---

## 19. What Phase 12 Currently Does — Final UI/UX & Analyst Workflow Polish

Phase 12 is a **presentation and navigation pass** over the finished
11-page application from Phases 0-11. It changes no detection, rule,
confidence, RAG, GenAI, response-planning, feedback,
continuous-improvement, explainability, audit, or evaluation logic — it
only changes how that logic's already-correct output is organized,
labeled, and explained on screen, so the application reads as one
coherent cybersecurity analyst dashboard instead of eleven separate
technical pages.

**Navigation.** The sidebar keeps the exact same 11-page list and the
same page order (adding, removing, or renaming a page was explicitly
out of scope), but now groups those pages, for reference only, into the
four stages an examiner actually walks through: *Detection &
Classification* (Data Preprocessing → AI Anomaly Detection → Incident
Classification → Uncertainty & Confidence), *AI Analysis & Response*
(Generative AI Analysis → Response Planning), *Human Review & Feedback*
(Analyst Review → Feedback History), and *Explainability & Evaluation*
(Explainability & Audit Trail → Evaluation & Performance). The
previously long, always-visible "Pipeline status" and "Phase completion
status" lists now live inside collapsed expanders, so the sidebar reads
as a short menu rather than a long scroll.

**The Overview page** was substantially expanded from a Phase 0-era
dataset preview into a real system front page: a concise "What this is"
description (purpose, defensive-only nature, AI + rule-based analysis
side by side, RAG grounding, and human analyst oversight), the same
`Logs → Detection → Classification → Confidence → RAG → GenAI →
Response Plan → Analyst Review → Feedback → Improvement` pipeline
every other page already summarizes, a **Current System Status**
section, and a **Recommended Demonstration Flow** guide. Every status
metric is either a real, freshly-computed value or an honest "not yet
computed" placeholder — never a fabricated number:

| Metric | Source | If not yet available |
|--------|--------|-----------------------|
| Dataset Size | `get_basic_stats()` (Phase 0) | always available |
| Anomalous Events | `st.session_state["anomaly_result"]` (Phase 2) | shows **"Not yet run"** until AI Anomaly Detection has been visited this session |
| Knowledge Base Size | `src.retriever.load_knowledge_base()` (Phase 8) | always available |
| Ollama Available / Current AI Model | `src.genai_analyzer.get_ollama_status()` (Phase 5) | always available (reports unavailable honestly) |
| Analyst Feedback Recorded | `src.feedback.summarize_feedback()` (Phase 7) | shows **"Unavailable"** only if the feedback database itself cannot be read |

The Recommended Demonstration Flow expander is a 9-step reading guide
over the real dataset already loaded (Run detection → inspect
classification → inspect confidence → view RAG → generate the AI
summary → view the response plan → analyst review → feedback history →
explainability/evaluation) — it stages nothing and fabricates nothing;
every step just points at a page that already does real work on the
real data.

**Consistent Incident ID.** Every page where an analyst selects one
incident (Incident Classification, Uncertainty & Confidence, Generative
AI Analysis, Response Planning, Analyst Review, Explainability & Audit
Trail) now shows that incident's stable `INC-0000`-style ID (Phase 7's
own `build_incident_id()`, never recomputed differently) alongside its
type, severity, and confidence, so the same incident is recognizable by
the same identifier across every page — directly satisfying the "use
consistent incident identifiers" requirement.

**Analyst Review workflow.** The page now opens with a collapsed,
numbered "How to use this page" guide matching the actual on-screen
order: Select Incident → Review System Analysis → Review RAG Knowledge
→ Review AI Analysis → Review Response Plan → Choose APPROVE / REJECT /
MODIFY → Add Analyst Notes → Submit Decision → see the confirmation →
see previous review history. A **Retrieved Knowledge (RAG)** section was
added between Knowledge-Based Analysis and Generative AI Analysis —
step 3 of the required workflow was previously visible only on the
Generative AI Analysis page. It calls the exact same, already-existing,
deterministic `retrieve_knowledge_for_incident()` every other page
already calls (never a new retrieval path), and its result is now also
passed into the Phase 10 audit-logging helper so a review performed
without first visiting the Generative AI Analysis page still records a
`RAG_RETRIEVAL` audit event.

**Feedback History vs. Continuous Improvement Insights.** The page now
opens each half with an explicit top-level heading — `FEEDBACK HISTORY`
and `CONTINUOUS IMPROVEMENT INSIGHTS` — so the raw, per-decision record
(Phase 7) and the derived, pattern-level insight (Phase 9) read as two
clearly separate sections rather than one long scroll. The required
disclaimer ("These insights are recommendations for future system
improvement. They do not automatically modify the system.") is
unchanged, word for word.

**Explainability & Audit Trail.** The page now opens its per-incident
explanation with one umbrella heading, `WHY WAS THIS INCIDENT FLAGGED?`,
captioned with the 8-step evidence chain (Anomaly Detection → Rule
Evidence → Severity → Confidence → RAG Context → GenAI Output →
Response Plan → Human Decision) before walking through exactly those
eight sections, followed by the (unchanged) Audit Timeline. The
SYSTEM-GENERATED FACT / RAG REFERENCE MATERIAL / GENERATIVE AI TEXT /
HUMAN ANALYST DECISION captions on each section are unchanged from
Phase 10.

**Evaluation & Performance.** Unchanged in substance — Phase 12 verified
that all eight A-H sections plus Local Performance Timing still render
with the same consistent metric-card style, and that every
ground-truth-derived metric is still labeled **"Offline Evaluation."**

**Error / empty states.** Reviewed against the Phase 12 checklist (empty
dataset, no anomaly selected, no feedback yet, no audit events yet,
Ollama unavailable, AI fallback used, no retrieved knowledge, missing
optional information) — every one of these was already a clear,
non-technical `st.info`/`st.warning` message from Phases 0-11, so no
message text needed to change. Genuine, unexpected exceptions (a
preprocessing failure, a bad model parameter, and so on) still surface
via `st.exception()` inside their existing `try/except` blocks, exactly
as before — Phase 12 does not hide errors that need developer
attention, it only ensures ordinary "nothing here yet" situations never
look like one.

**Safety.** Phase 12 added zero new capability of any kind — no command
execution, no account blocking, no firewall modification, no host
isolation, no exploit generation. It is a pure UI/UX and copy pass on
top of an application whose safety properties were already established
in Phases 0-11.

### Testing Phase 12

`test_ui_polish.py` (125 checks) is deliberately structural rather than
numeric, since Phase 12 changed no calculation: it re-confirms all 11
pages still load with no exception, that the specific Phase 12
additions actually render (Overview's status section and demo guide,
the consistent Incident ID strip, the Analyst Review workflow guide and
its new RAG step, the Feedback History / Continuous Improvement
separation, the Explainability "why was this flagged" framing), that
empty/error states stay exception-free, that the dataset checksum is
unchanged, that Qwen3:1.7B + `think=False` + `ollama==0.6.2` remain
exactly as configured, and that Phase 9/10/11's own report-building
functions still return real, correct results when called directly
(a structural regression guard, independent of the UI). Run standalone:

```bash
python test_ui_polish.py
```

A full project regression (`python test_retriever.py`, 1,196+ checks)
remains available and unaffected by Phase 12, and was intentionally not
re-run repeatedly during Phase 12 development.

---

## 20. What Phase 13 Currently Does — Final Integration, QA, Packaging & Demo Readiness

Phase 13 is the final development phase. It adds no pipeline stage, no
Streamlit page, and no algorithm - it is a whole-project audit, a real
end-to-end run of the operational pipeline, a full current-suite test
pass, targeted documentation/dependency cleanup, and final packaging.

**What was audited.** Every module under `src/` was confirmed present,
importable, and syntactically clean (`py_compile` + a fresh import of
all 14 modules); the 11-page `PAGES` list in `app.py` was confirmed
unchanged from Phase 12; the verified baseline
(`OLLAMA_MODEL_NAME = "qwen3:1.7b"`, `think=False` in
`src/genai_analyzer.py`, `ollama==0.6.2` in `requirements.txt`) was
re-confirmed unchanged; and `true_label`/`config.LABEL_COLUMN` was
re-confirmed confined to `src/evaluation.py` (Offline Evaluation) and
`src/anomaly_detector.py`'s optional post-hoc function, with
`src/preprocessor.py` importing it only to explicitly exclude it from
the model's feature matrix.

**What was actually run, not just claimed.** The complete current
on-disk test suite (`test_preprocessing.py` through `test_ui_polish.py`,
13 files) was run once, end to end, on the current project. This
surfaced one genuine regression: Phase 12 promoted the Feedback
History page's "Continuous Improvement Insights" heading from an
`st.subheader()` to a top-level `st.markdown("## ...")` heading (see
§19), but `test_continuous_improvement.py`'s own Phase 9 Streamlit
check still looked for the old `st.subheader` text and failed. The
underlying Phase 9 logic was never wrong - only the test's assumption
about which Streamlit widget carries the heading - so the fix was to
the test, not to `app.py`: it now checks `at.markdown` for the current
heading text, exactly like `test_ui_polish.py` already does for the
same page. After the fix, every one of the 13 test files passes
against the current on-disk project (see §13-§19 above, and the
project's own Cowork history, for the exact per-file check counts;
`test_continuous_improvement.py` passes 102/102).

A separate, real end-to-end script (not part of the shipped test suite)
exercised the full operational pipeline on the real 350-record dataset
with isolated temporary databases (never touching `data/*.db`): loading
logs, preprocessing, Isolation Forest anomaly detection, rule-based
classification, confidence scoring, real TF-IDF RAG retrieval against
the real knowledge base, a real call attempt to
`generate_ai_case_summary()` (this sandbox has no reachable local Ollama
server, so the call correctly and automatically used the application's
own documented deterministic fallback - the real Qwen3:1.7B code path
itself was exercised and left completely untouched), response-plan
generation, three independent analyst decisions (APPROVE, REJECT, and
MODIFY - the modified plan was verified, byte-for-byte after a database
round-trip, to differ from and never overwrite the original stored
plan), feedback persistence, a continuous-improvement report, an
explainability report, audit-trail event logging and retrieval, and a
Phase 11 evaluation report. All 19 steps completed with no exceptions.

**What was fixed.** `test_continuous_improvement.py` (see above); two
stale `README.md` cross-references left over from Phase 12
(`test_ui_polish.py`'s and `app.py`'s own folder-structure comments
both said "see §20" when the actual Phase 12 section is §19); a stale
`test_audit_trail.py` check-count in this README (said 91, the file
actually runs 92); and three leftover `llama3` mentions in this
README's setup/GenAI sections that predated the Qwen3:1.7B switch and
would have told a reader to `ollama pull` the wrong model.
`requirements.txt` was reviewed against every third-party import in the
project and against the installed package versions - no changes were
needed; it was already accurate.

**What was intentionally left alone.** No pipeline, page, or algorithm
logic changed. No new `src/` module, no new page, no new dependency.
The dataset and its checksum are unchanged. Phase 8-12 behavior is
unchanged.

```bash
python test_preprocessing.py
python test_anomaly_detection.py
python test_rule_engine.py
python test_confidence.py
python test_genai.py
python test_response_planner.py
python test_feedback.py
python test_retriever.py          # full Phase 1-8 regression, 1,196 checks
python test_continuous_improvement.py
python test_audit_trail.py
python test_explainability.py
python test_evaluation.py
python test_ui_polish.py
```

---

## 21. Academic Context

This is a defensive-only, educational project for a 5th-semester
Computer/AI engineering course (CIA project). It is built entirely
from scratch for this coursework — no existing repository, product, or
codebase was cloned, copied, or modified to create it.
