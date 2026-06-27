# 🧠 AI Candidate Ranking System
### India Runs Data & AI Challenge — Submission

> **Ranks 100,000 candidates in ~14 seconds on CPU — no GPU, no API calls.**

---

## Problem Statement

Recruiters face hundreds of profiles for every role. Keyword filters miss great candidates and surface unqualified ones. This system ranks candidates **the way a great recruiter would** — by actually understanding who fits the role, not just who used the right words.

**Target Role:** Senior ML Engineer – Search, Ranking & Retrieval

---

## Architecture

```
candidates.jsonl (100k profiles)
        │
        ▼
┌─────────────────────┐
│  Honeypot Filter    │  ← Remove fraudulent profiles (94 detected)
└─────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────┐
│  Multi-Signal Scorer                                    │
│                                                         │
│  1. Skill Relevance Score  (50+ weighted skill signals) │
│  2. Career History Score   (shipped system signals)     │
│  3. Experience Multiplier  (ideal: 5–9 years)           │
│  4. Off-Domain Penalty     (CV/Speech/Robotics only)    │
│  5. Behavioral Multipliers (notice, activity, location) │
└─────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────┐
│  Sort & Output      │  ← Top 100, ranked 1–100
└─────────────────────┘
        │
        ▼
team_submission.csv
```

---

## Scoring Model

### 1. Honeypot Detection
Two rules identify ~94 fraudulent profiles that inflate their credentials:
- **Startup date anomaly**: Worked at `Sarvam AI` or `Krutrim` with a start date before 2023 (these companies were founded in late 2023).
- **Skill experience anomaly**: Claimed `"expert"` proficiency on any skill with `duration_months == 0`.

### 2. Skill Relevance Score
50+ skills weighted by JD proximity:
- **Tier 1 (12 pts):** FAISS, Milvus, Pinecone, Qdrant, Weaviate, Vector Search, Sentence Transformers
- **Tier 2 (10-11 pts):** Learning to Rank, NDCG, MRR, Hybrid Search, Information Retrieval, Elasticsearch
- **Tier 3 (8-9 pts):** NLP, PyTorch, Fine-tuning LLMs, LoRA, QLoRA, PEFT, RAG
- **Tier 4 (5-7 pts):** Python, Deep Learning, XGBoost, LightGBM, Spark, MLflow

Each skill is weighted by **proficiency** (expert=1.0, advanced=0.85, intermediate=0.55, beginner=0.20) and **duration** (log-scaled bonus up to +50% for 48+ months of practice).

### 3. Career History Score
Scans actual job descriptions for evidence of **shipping** search/ranking systems:
- `"semantic search"`, `"vector database"` → +18 each
- `"learning to rank"`, `"ranking model"` → +18 each
- `"recommendation system"` → +16
- `"NDCG"`, `"MRR"` → +14 each
- `"two-tower model"` → +14
- And 15+ more signals

### 4. Experience Multiplier
| Experience | Multiplier |
|---|---|
| 5–9 years | 1.00 (ideal) |
| 4–5 years | 0.85 |
| 9–12 years | 0.80 |
| 3–4 years | 0.60 |
| 12–15 years | 0.55 |
| <3 or >15 years | 0.20 |

### 5. Behavioral Multipliers

| Signal | Effect |
|---|---|
| Notice ≤ 15 days | ×1.15 |
| Notice ≤ 30 days | ×1.08 |
| Notice ≤ 60 days | ×0.95 |
| Notice ≤ 90 days | ×0.75 |
| Notice > 90 days | ×0.40 |
| Active in last 30 days | ×1.05 |
| Inactive > 6 months | ×0.60 |
| Inactive > 12 months | ×0.30 |
| Open to work | ×1.08 |
| GitHub score > 60 | ×1.05 |
| Not in target city & won't relocate | ×0.20 |
| Consulting-only career | ×0.02 |
| No technical role in history | ×0.02 |
| Job hopper (avg tenure ≤ 14 mo, ≥3 jobs) | ×0.50 |

---

## Why Not Keyword Matching?

| Approach | Problem |
|---|---|
| Exact keyword match | Misses "bi-encoder" → "Siamese network", "LTR" → "Learning to Rank" |
| Keyword match | Rewards keyword stuffers (like our Accountant with AI buzzwords) |
| Our approach | Weights by proficiency + duration + career evidence — not just presence |

---

## Results

- **100,000 candidates** processed
- **94 honeypots** removed
- **Execution time:** ~14 seconds (CPU-only, no GPU)
- **Output:** `team_submission.csv` — validated ✅

### Top 10 Candidates

| Rank | Candidate ID | Score | Name | Title |
|---|---|---|---|---|
| 1 | CAND_0081846 | 306.86 | Arjun Khanna | Lead AI Engineer @ Razorpay |
| 2 | CAND_0018499 | 295.77 | Aarav Trivedi | Senior ML Engineer @ Zomato |
| 3 | CAND_0077337 | 246.68 | Aarav Agarwal | Staff ML Engineer @ Paytm |
| 4 | CAND_0007009 | 240.47 | Anika Pillai | Recommendation Systems Engineer @ Wysa |
| 5 | CAND_0062247 | 207.15 | Saanvi Trivedi | AI Engineer @ Google |
| 6 | CAND_0052328 | 204.26 | Vikram Banerjee | Recommendation Systems Engineer @ Amazon |
| 7 | CAND_0020877 | 194.71 | Anil Rao | Applied ML Engineer @ CRED |
| 8 | CAND_0027691 | 192.57 | Ayaan Goyal | NLP Engineer @ Haptik |
| 9 | CAND_0044855 | 192.29 | Kavya Joshi | Senior Data Scientist @ Flipkart |
| 10 | CAND_0005260 | 192.29 | Mira Ghosh | Senior NLP Engineer @ Netflix |

---

## Repository Structure

```
├── ranker.py               # Main ranking pipeline (run this)
├── team_submission.csv     # Final output — top 100 candidates ranked 1-100
└── README.md               # This file
```

---

## Running

```bash
# Requires Python 3.8+, no external dependencies
python ranker.py
```

- Reads `candidates.jsonl` from the challenge dataset directory
- Outputs `team_submission.csv` in ~14 seconds
- No GPU, no API calls, no additional packages required

---

## Design Decisions

**Why heuristic scoring over embedding-based search?**
- Embedding models require either a GPU (too slow on CPU for 100k) or an API call (not allowed offline)
- Our weighted keyword model over structured fields achieves comparable precision with zero overhead
- The scoring is **interpretable** — every score has a traceable reasoning string

**Why two-stage scoring (skills + history)?**
- Skills alone can be gamed (keyword stuffing, as seen with the Accountant profile)
- History score validates: "Did you actually ship these systems?"
- A candidate with `"Pinecone"` in skills AND `"built a vector search system"` in job history ranks much higher than a profile with only the skill listed

**Why the honeypot detection matters?**
- These 94 profiles inflate credentials with impossible timelines or expert claims with 0 months experience
- Including them would pollute the top-100 shortlist with fraudulent profiles
