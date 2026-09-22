# Generative AI-Based Cybersecurity Process Reengineering System

**AI-Based Cybersecurity Analyst Assistant** | Artificial Intelligence CIA-1

A defensive, human-in-the-loop assistant that helps an analyst interpret synthetic security logs, investigate anomalies, consult defensive knowledge, and review AI-assisted findings before making the final decision.

## 1. Overview

Security analysts must turn large volumes of log activity into consistent, explainable case decisions. This project demonstrates a local AI workflow for that task. It combines data preprocessing, unsupervised anomaly detection, explainable rules, severity and confidence assessment, retrieval-augmented generation (RAG), local generative AI, advisory response planning, and analyst feedback.

The application is designed as an academic prototype of an analyst workstation. It does not replace the analyst: generated summaries and next steps remain advisory, and the analyst must approve, modify, or reject each case.

## 2. Problem Statement

Manual log review can be slow and inconsistent. An analyst needs to know whether activity is unusual, what incident type it may represent, how serious it is, how reliable the evidence is, why it was flagged, which defensive guidance applies, and what should be reviewed next. The project addresses this workflow while keeping the reasoning visible and the final decision with a human.

## 3. Proposed Solution

The system processes each case through the following workflow:

**Security Logs** → **Data Preprocessing** → **Isolation Forest Anomaly Detection** → **Rule-Based Classification** → **Severity + Confidence** → **RAG Knowledge Retrieval** → **Qwen3:1.7B Generative AI** → **AI Case Summary** → **Recommended Next Steps** → **Human Analyst Review** → **APPROVE / MODIFY / REJECT** → **Feedback Storage** → **Continuous Improvement**

The detection and classification stages establish the structured case result. RAG supplies relevant defensive context, while Qwen3:1.7B produces an analyst-friendly summary. A separate advisory planner produces recommended next steps. The analyst reviews all of this evidence and records the final decision.

## 4. Key Features

| Implemented feature | Purpose |
|---|---|
| Security log processing | Loads and presents security-log records for analysis. |
| Data preprocessing | Cleans data and engineers model-ready features. |
| Isolation Forest anomaly detection | Detects unusual activity without operational labels. |
| Rule-based incident classification | Assigns explainable incident types. |
| Severity assessment | Determines case severity from structured evidence. |
| Confidence/uncertainty | Communicates evidence strength and review needs. |
| RAG knowledge retrieval | Finds relevant defensive guidance locally. |
| Qwen3:1.7B local GenAI | Generates analyst-friendly case summaries. |
| AI Case Summary | Explains the detected case in readable language. |
| Recommended Next Steps | Produces separate advisory investigation steps. |
| APPROVE / MODIFY / REJECT | Records the analyst's final decision. |
| Feedback storage | Persists decisions and analyst notes in SQLite. |
| Continuous improvement analysis | Summarizes feedback patterns for review. |
| Explainability | Shows evidence and reasoning behind results. |
| Audit trail | Records pipeline and review events. |
| Evaluation/performance | Reports offline metrics on labeled synthetic data. |

## 5. System Architecture

![System Architecture](docs/architecture.png)

The Streamlit interface connects the processing pipeline to two user-facing areas. Supporting components include TF-IDF and cosine similarity for retrieval, Ollama for local model execution, SQLite for feedback and audit records, and dedicated explainability and evaluation modules.

## 6. How the Application Works

The current interface has two main areas: **Case Analysis** and **Review & Insights**.

In **Case Analysis**, the analyst selects a case and views its incident type, severity, confidence, and review status. The page presents the **AI Case Summary**, **Why Was This Flagged?**, **Knowledge Base Used by AI**, and **Recommended Next Steps**. The analyst then chooses **APPROVE**, **MODIFY**, or **REJECT**, with optional notes.

**Review & Insights** provides case history, analyst decisions, feedback, continuous-improvement analysis, explainability, audit trail records, and evaluation/performance information.

## 7. Application Screenshots

**1. Case Analysis — Main Interface**

![Case Analysis — Main Interface](screenshots/01_case-analysis-main.png)

**2. AI Case Summary — Real Qwen3:1.7B Output**

![AI Case Summary — Real Qwen3:1.7B Output](screenshots/02_ai-case-summary.png)

**3. Why Was This Flagged? + Knowledge Base Used by AI**

![Why Was This Flagged? + Knowledge Base Used by AI](screenshots/03_why-flagged-rag.png)

**4. Recommended Next Steps + Analyst Decision**

![Recommended Next Steps + Analyst Decision](screenshots/04_recommended-next-steps.png)

**5. Review & Insights — Case History**

![Review & Insights — Case History](screenshots/05_review-insights.png)

**6. Review & Insights — Audit Trail**

![Review & Insights — Audit Trail](screenshots/06_audit-trail.png)

## 8. Sample Output

The screenshots demonstrate case **INC-0011**.

**Detection & Classification**

- Incident type: **Suspicious Data Transfer**
- Severity: **High**
- Confidence: **45% (Low)**
- Event evidence: 45 requests on port 3306, 3665.54 MB transferred, at 10:00 AM, from New York, US

**AI Case Summary**

Qwen3:1.7B describes the activity as a suspicious transfer that is anomalous because of high data volume and session duration. It notes that the activity may indicate unauthorized data movement and recommends further investigation.

**Retrieved RAG Knowledge**

The retrieved sources shown in the application include *How to Handle Large Data Transfers* and *Unusual Login Location Guide*.

**Recommended Next Steps**

The separate planner produces a **High-Priority Suspicious Data Transfer Response Plan** with **URGENT** priority and **Human Review Required**. The displayed steps include reviewing transfer details, identifying the user or session, examining related activity, collecting evidence, and escalating according to policy.

**Human Analyst Decision**

The demonstrated analyst decision is **APPROVE**.

## 9. AI and RAG

Qwen3:1.7B runs locally through **Ollama** and generates analyst-friendly case summaries from structured detection results. The model does not decide incident type, severity, or confidence.

The local RAG knowledge base contains **10 JSON documents**. The retriever uses **TF-IDF** and **cosine similarity** to rank relevant defensive references. Retrieved knowledge is supplied as context for analysis; there is no live internet search. The AI Case Summary and Recommended Next Steps remain separate outputs.

## 10. Dataset

The project uses **350 synthetic cybersecurity security-log records** containing normal and suspicious activity. Examples include brute-force activity, unusual login locations, large data transfers, port scanning, off-hours privilege activity, and multi-device activity.

The `true_label` column is ground truth used **only for offline evaluation**. It is not supplied to the operational detection pipeline.

## 11. Evaluation

These are offline evaluation results on the synthetic dataset, not production cybersecurity performance:

| Metric | Result |
|---|---:|
| Accuracy | 98.00% |
| Precision | 100.00% |
| Recall | 90.91% |
| F1 Score | 95.24% |

## 12. Technology Stack

| Technology | Use in the project |
|---|---|
| Python 3.12.x | Application and pipeline implementation |
| Streamlit | Interactive analyst interface |
| Pandas and NumPy | Data loading, transformation, and feature engineering |
| Scikit-learn | Standardization, Isolation Forest, TF-IDF, cosine similarity, and metrics |
| Plotly | Review and evaluation visualizations |
| Ollama | Local runtime for Qwen3:1.7B |
| Qwen3:1.7B | Local generative model for case summaries |
| SQLite | Feedback and audit-trail persistence |
| MiniSom | Optional clustering visualization supported by the application |

## 13. Installation and Running

Requirements: **Python 3.12.x**, **Ollama**, and **Qwen3:1.7B**.

```powershell
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

```powershell
ollama pull qwen3:1.7b
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501). Ollama and Qwen3:1.7B run locally. Model weights are not included in GitHub; the `ollama pull` command downloads them to the local Ollama installation.

## 14. Project Structure

```text
.
├── app.py
├── config.py
├── generate_data.py
├── requirements.txt
├── data/
├── knowledge_base/
├── screenshots/
├── docs/
├── src/
└── README.md
```

`app.py` is the Streamlit entry point. `src/` contains the processing, detection, classification, confidence, retrieval, GenAI, planning, feedback, explainability, audit, and evaluation modules. `data/` contains the synthetic logs, `knowledge_base/` contains the ten JSON references, and `screenshots/` and `docs/` contain project visuals.

## 15. Safety and Scope

This project is **defensive-only, academic, advisory, and human-in-the-loop**. It does not automatically block accounts, modify firewalls, isolate hosts, exploit systems, execute attacks, or perform offensive actions. Recommendations are presented for analyst review, and the analyst makes the final decision.

## 16. Limitations

The dataset is synthetic, so its patterns and evaluation results do not represent real-world SOC performance. The application depends on a local Ollama installation for the demonstrated Qwen3:1.7B output. The project remains an academic prototype, and its offline evaluation is limited to the available synthetic ground truth.

## 17. Reference Paper

Sharbaf, Mehrdad S.  
“Reengineering Cybersecurity Processes with Generative AI: From Automation to Strategic Alignment.”  
5th IEEE International Conference on AI in Cybersecurity (ICAIC), 2026.  
DOI: 10.1109/ICAIC67076.2026.11395839

## 18. Student Details

| Field | Details |
|---|---|
| Student Name | Gabriel James |
| Roll Number | 5024128 |
| Department | Information Technology |
| Institute | Fr. Conceicao Rodrigues Institute of Technology (FCRIT), Vashi, Navi Mumbai |
| Academic Year | 2026–27 |
| Subject | Artificial Intelligence |
| CIA / Assignment | CIA-1 |

## 19. Demo Walkthrough

1. Launch the application and open **Case Analysis**.
2. Select a suspicious case and show its incident type, severity, confidence, and review status.
3. Display the real Qwen3:1.7B **AI Case Summary**.
4. Show **Why Was This Flagged?** and the retrieved RAG sources.
5. Display **Recommended Next Steps** and choose **APPROVE**, **MODIFY**, or **REJECT**.
6. Open **Review & Insights** to review history, feedback, explainability, the audit trail, and evaluation/performance.

## 20. Author

**Gabriel James**  
Roll No.: **5024128**
