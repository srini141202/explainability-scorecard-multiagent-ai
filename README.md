# Explainability Scorecard Framework for Multi-Agent AI Workflows in Healthcare Insurance Claim Processing

![Python](https://img.shields.io/badge/Python-3.13-blue) ![LangGraph](https://img.shields.io/badge/LangGraph-Multi--Agent-green) ![License](https://img.shields.io/badge/License-MIT-yellow) ![NCI](https://img.shields.io/badge/NCI-MSc%20AI-navy)

> MSc Artificial Intelligence Research Practicum — National College of Ireland, August 2026  
> **Author:** Srinivasan Dillikumar (x24285951) | National College of Ireland

---

## Overview

This research proposes and demonstrates the **Explainability Scorecard Framework** — the first quantitative, automated tool for measuring explainability across multi-agent AI workflows in regulated domains.

Existing tools like LIME and SHAP explain single models but cannot trace accountability across a pipeline of collaborating agents. When a four-agent AI system processes a healthcare insurance claim and gets it wrong, no existing tool can say which agent caused the problem. This framework solves that.

The scorecard is demonstrated on a **four-agent LangGraph pipeline** that processes healthcare insurance claims against 947 real CMS Medicare LCD policies. It achieved a **composite explainability score of 9.1/10** — a 350% improvement over a single-agent baseline.

---

## Architecture

```
Patient Claim Input
        │
        ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Agent 1       │────▶│   Agent 2        │────▶│   Agent 3       │────▶│   Agent 4       │
│   Retriever     │     │   Interpreter    │     │   Decision      │     │   Explanation   │
│                 │     │                  │     │                 │     │                 │
│ Semantic search │     │ Checks each      │     │ APPROVE /       │     │ Plain English   │
│ 947 CMS LCD     │     │ criterion vs     │     │ DENY /          │     │ letter for      │
│ policies (k=1)  │     │ patient data     │     │ REQUIRES PRIOR  │     │ patient citing  │
│                 │     │                  │     │ AUTH + reason   │     │ policy by name  │
└─────────────────┘     └─────────────────┘     └─────────────────┘     └─────────────────┘
        │                       │                       │                       │
        └───────────────────────┴───────────────────────┴───────────────────────┘
                                          │
                              ClaimState Shared Memory
                                          │
                                          ▼
                              JSON Log (timestamped, per claim)
                                          │
                                          ▼
                              Scoring Engine → D1–D5 Scores → Composite Score
```

---

## The Explainability Scorecard

Five dimensions, each mapped to an EU AI Act article:

| Dimension | Score | EU AI Act | What it measures |
|---|---|---|---|
| D1 Agent Transparency | 9.62/10 | Art. 13 | Input, reasoning, output logged per agent + RAGAS faithfulness |
| D2 Communication Clarity | 6.74/10 | Art. 13 | Flesch Reading Ease on patient explanation |
| D3 Outcome Attribution | 10.00/10 | Art. 12 | Decision traceable to exact agent + exact policy |
| D4 Human Oversight | 9.13/10 | Art. 14 | Explanation contains policy, decision, coverage, claim reference |
| D5 Temporal Accountability | 10.00/10 | Art. 12 | All agents timestamped in correct sequence |
| **Composite** | **9.10/10** | | Equal-weight average of all five |

---

## Evaluation Results

| Test | Result | Conclusion |
|---|---|---|
| Test 1 — Reliability | Composite CV = 0.59% | RELIABLE (threshold < 10%) |
| Test 2 — Discriminant Validity | D4 HITL 9.17 vs Auto 9.01 (+0.16) | Detects real workflow differences |
| Test 3 — Baseline Comparison | 9.10 vs 2.02 (+7.08 points, +350%) | Structure creates explainability |
| Test 4 — Accuracy Correlation | Pearson r = −0.028 | Explainability ≠ Accuracy (independent) |

---

## Technology Stack

| Component | Technology |
|---|---|
| Multi-agent orchestration | LangGraph (LangChain) |
| LLM | Llama 3.3 70B via Groq API |
| Sentence embeddings | all-MiniLM-L6-v2 (HuggingFace) |
| Vector database | Chroma |
| Readability scoring | textstat (Flesch Reading Ease) |
| Language | Python 3.13 |

---

## Project Structure

```
Thesis/
├── Datasets/
│   ├── Patient_Data_Synthea/          # 112 synthetic patients (Synthea)
│   └── Policy_CMS_LCD/                # 947 CMS Medicare LCD policies
├── Authored_Claims/
│   └── claims_generated.json          # 120 synthetic claims (seed=42)
├── Implementation/
│   ├── main.py                        # Run everything end-to-end
│   ├── langgraph_workflow.py          # Four-agent pipeline
│   ├── batch_runner.py               # Process all 120 claims
│   ├── scoring_engine.py             # Compute D1–D5 scores
│   ├── vector_store.py               # Build/load Chroma vector store
│   ├── data_loader.py                # Load CSV datasets
│   ├── data_cleaner.py               # Clean CMS LCD policies
│   ├── claim_generator.py            # Generate synthetic claims
│   ├── reliability_test.py           # Evaluation Test 1
│   ├── discriminant_validity.py      # Evaluation Test 2
│   ├── baseline_comparison.py        # Evaluation Test 3
│   ├── accuracy_correlation.py       # Evaluation Test 4
│   └── visualization.py             # Generate charts
└── Evaluation/
    └── results/                       # All evaluation JSON results
```

---

## Quick Start

### Prerequisites
- Python 3.13
- Groq API key (free at [console.groq.com](https://console.groq.com))

### Installation

```bash
# Clone the repository
git clone https://github.com/srini141202/explainability-scorecard-multiagent-ai.git
cd explainability-scorecard-multiagent-ai

# Install dependencies
pip install langchain langchain-groq langchain-community langchain-core langgraph groq sentence-transformers chromadb pandas numpy matplotlib textstat python-dotenv

# Configure API key
echo "GROQ_API_KEY=your_groq_api_key_here" > Implementation/.env
```

### Run

```bash
cd Implementation

# Run the full pipeline on all 120 claims
python main.py

# Or run a single claim demonstration
python langgraph_workflow.py
```

### Run Evaluation Tests

```bash
python reliability_test.py          # Test 1 — Reliability
python discriminant_validity.py     # Test 2 — Discriminant Validity
python baseline_comparison.py       # Test 3 — Baseline Comparison
python accuracy_correlation.py      # Test 4 — Accuracy Correlation
python visualization.py             # Generate evaluation charts
```

---

## Data

- **Patient data:** 112 synthetic patients from [Synthea](https://synthea.mitre.org/) — no real patient data, fully GDPR compliant
- **Policy data:** 947 CMS Medicare LCD policies from [CMS public database](https://www.cms.gov/medicare-coverage-database/) — public government documents
- **Claims:** 120 synthetic claims generated with `random.seed(42)` for full reproducibility
- **No ethics approval required** — all data is synthetic or publicly available

---

## Key Findings

1. **Structure creates explainability** — the four-agent system scored 9.1/10 vs 2.0/10 for a single-agent black-box doing the same task. The difference came from structure, not a smarter model.

2. **Explainability ≠ Accuracy** — Pearson r = −0.028 (effectively zero). The scorecard measures *how* a decision was made, not *whether* it was correct. The EU AI Act demands both.

3. **Reliable and automated** — composite CV of 0.59% across 10 repeated runs. No human raters. Any examiner running the scripts gets identical results because of `random.seed(42)`.

---

## Citation

If you use this work, please cite:

```
Dillikumar, S. (2026). Explainability Scorecard Framework for Multi-Agent AI Workflows 
in Healthcare Insurance Claim Processing. MSc Research Practicum, 
National College of Ireland.
```

---


*MSc Artificial Intelligence · School of Computing · National College of Ireland · August 2026*
