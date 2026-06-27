"""
AI Candidate Ranking System
============================
Processes 100,000 candidate profiles and ranks them against the
"Senior ML Engineer – Search & Ranking" JD using a multi-signal
scoring model (skill relevance, career trajectory, behavioral signals).

Output: team_submission.csv  (top 100 candidates, ranked 1-100)
"""

import csv
import json
import sys
import time
from datetime import datetime

# ──────────────────────────────────────────────
# Constants / Reference Data
# ──────────────────────────────────────────────

REFERENCE_DATE = datetime(2026, 6, 27)
JSONL_PATH = (
    r"C:\Users\9\Desktop\Ai Ranking System"
    r"\[PUB] India_runs_data_and_ai_challenge"
    r"\India_runs_data_and_ai_challenge\candidates.jsonl"
)
OUTPUT_CSV = r"C:\Users\9\Desktop\Ai Ranking System\team_submission.csv"
TOP_N = 100

# Honeypot company names (exact match)
HONEYPOT_COMPANIES = {"Sarvam AI", "Krutrim"}
HONEYPOT_FOUNDING_YEAR = 2023   # any start_date year < 2023 at these firms is a flag

# Technical title keywords – a candidate MUST have had at least one technical role
TECH_TITLE_KEYWORDS = [
    "engineer", "developer", "programmer", "architect", "scientist",
    "tech lead", "technical staff", "mts", "analyst", "researcher",
    "ai specialist", "data", "ml", "nlp", "ai", "machine learning",
    "search", "ranking", "platform", "infrastructure",
]

# Pure consulting body-shops (if ALL roles are here -> downgrade)
CONSULTING_FIRMS = [
    "tcs", "tata consultancy", "infosys", "wipro", "accenture",
    "cognizant", "capgemini", "hcl", "tech mahindra", "l&t infotech",
    "mphasis", "genpact", "hexaware", "mindtree",
]

# JD-target locations
TARGET_LOCATIONS = [
    "pune", "noida", "bangalore", "bengaluru", "delhi", "new delhi",
    "mumbai", "hyderabad", "gurgaon", "gurugram", "chennai",
]

# ── Skill weights (higher = closer to JD requirements) ────────────────
SKILL_WEIGHTS = {
    # Core retrieval / vector search
    "faiss": 12, "milvus": 12, "pinecone": 12, "qdrant": 12, "weaviate": 12,
    "vector search": 12, "vector database": 12, "ann search": 12,
    "hybrid search": 11, "dense retrieval": 11, "sparse retrieval": 11,
    "bm25": 10, "elasticsearch": 10, "opensearch": 10, "solr": 8,
    # Embeddings / transformers
    "sentence transformers": 12, "embeddings": 11, "bi-encoder": 11,
    "cross-encoder": 11, "siamese network": 10,
    "hugging face": 10, "transformers": 10,
    # Ranking / evaluation
    "learning to rank": 13, "ltr": 13,
    "ndcg": 12, "mrr": 12, "map@": 12, "precision@": 11,
    "ranknet": 11, "lambdamart": 11, "listwise": 10,
    "re-ranking": 11, "reranking": 11,
    # RAG / LLM
    "rag": 10, "retrieval augmented": 10,
    "fine-tuning llms": 9, "fine-tune": 8,
    "lora": 9, "qlora": 9, "peft": 9, "instruction tuning": 8,
    # ML / modelling
    "pytorch": 9, "tensorflow": 7,
    "xgboost": 7, "lightgbm": 7, "catboost": 7,
    "scikit-learn": 6, "sklearn": 6,
    "deep learning": 8, "neural network": 7,
    # NLP
    "nlp": 9, "named entity": 7, "text classification": 7,
    "question answering": 8, "information retrieval": 10,
    # Recommendation
    "recommendation system": 11, "collaborative filtering": 9,
    "matrix factorization": 8, "two-tower": 10, "recall": 8,
    # Engineering
    "python": 5, "spark": 6, "kafka": 6, "airflow": 5,
    "kubernetes": 5, "docker": 4, "rest api": 4, "fastapi": 5,
    "mlflow": 6, "kubeflow": 6, "ab testing": 6, "a/b testing": 6,
    # Weak signals (small boost)
    "machine learning": 5, "data science": 4,
}

# Keywords in job history that strongly signal shipping search/ranking systems
HIST_SIGNALS = {
    "semantic search": 18, "vector search": 18, "vector database": 18,
    "learning to rank": 18, "ranking model": 18, "re-rank": 15, "rerank": 15,
    "recommendation system": 16, "recommender": 16,
    "retrieval augmented": 15, "rag": 12,
    "ndcg": 14, "mrr": 14, "map@": 12,
    "faiss": 12, "milvus": 12, "pinecone": 12, "qdrant": 12, "weaviate": 12,
    "embedding model": 12, "bi-encoder": 12, "cross-encoder": 12,
    "information retrieval": 12, "two-tower": 14,
    "hybrid search": 12, "dense retrieval": 12,
    "query understanding": 10, "query expansion": 10,
    "search relevance": 12,
}

# Skills that belong ONLY to unrelated domains -> penalise if no overlap with target
OFF_DOMAIN_SKILLS = [
    "computer vision", "image classification", "object detection", "gan",
    "speech recognition", "text to speech", "tts", "robotics",
    "autonomous driving", "lidar", "slam",
]

# ──────────────────────────────────────────────
# Utility helpers
# ──────────────────────────────────────────────

def parse_date(s):
    if not s:
        return None
    try:
        return datetime.strptime(s, "%Y-%m-%d")
    except ValueError:
        return None


def days_since(d):
    if d is None:
        return 9999
    return (REFERENCE_DATE - d).days


def has_tech_title(history):
    for job in history:
        title = job.get("title", "").lower()
        if any(kw in title for kw in TECH_TITLE_KEYWORDS):
            return True
    return False


def is_consulting_only(history):
    if not history:
        return False
    for job in history:
        comp = job.get("company", "").lower()
        if not any(f in comp for f in CONSULTING_FIRMS):
            return False
    return True


def avg_tenure(history):
    if not history:
        return 36.0
    total = sum(j.get("duration_months", 0) for j in history)
    return total / len(history)


# ──────────────────────────────────────────────
# Honeypot detection
# ──────────────────────────────────────────────

def is_honeypot(cand):
    # Trap 1: worked at Sarvam AI / Krutrim before they existed
    for job in cand.get("career_history", []):
        if job.get("company") in HONEYPOT_COMPANIES:
            start = job.get("start_date", "")
            if start and int(start.split("-")[0]) < HONEYPOT_FOUNDING_YEAR:
                return True

    # Trap 2: skill claimed as "expert" with 0 months experience
    for s in cand.get("skills", []):
        if s.get("proficiency") == "expert" and s.get("duration_months", -1) == 0:
            return True

    return False


# ──────────────────────────────────────────────
# Scoring
# ──────────────────────────────────────────────

PROFICIENCY_MULT = {
    "expert": 1.0,
    "advanced": 0.85,
    "intermediate": 0.55,
    "beginner": 0.20,
}


def score_skills(skills):
    """Return (skill_score, list_of_matched_skill_names)."""
    total = 0.0
    matched = []
    seen = set()

    for s in skills:
        raw_name = s.get("name", "")
        name_lc = raw_name.lower()
        prof = s.get("proficiency", "beginner")
        pm = PROFICIENCY_MULT.get(prof, 0.2)
        dur = s.get("duration_months", 0) or 0

        # Duration bonus: logarithmic – caps at +50% for 48+ months
        dur_boost = 1.0 + min(0.5, dur / 48)

        for kw, weight in SKILL_WEIGHTS.items():
            if kw in name_lc and kw not in seen:
                seen.add(kw)
                contrib = weight * pm * dur_boost
                total += contrib
                matched.append(raw_name)
                break

    return total, matched


def score_history(history):
    """Score based on actual shipped systems described in job history."""
    combined = " ".join(
        (j.get("title", "") + " " + j.get("description", "")).lower()
        for j in history
    )
    total = 0.0
    signals_found = []
    seen = set()
    for kw, weight in HIST_SIGNALS.items():
        if kw in combined and kw not in seen:
            seen.add(kw)
            total += weight
            signals_found.append(kw)
    return total, signals_found


def experience_score(yoe):
    """Ideal: 5-9 years for a Senior ML role."""
    if 5.0 <= yoe <= 9.0:
        return 1.00
    elif 4.0 <= yoe < 5.0:
        return 0.85
    elif 9.0 < yoe <= 12.0:
        return 0.80
    elif 3.0 <= yoe < 4.0:
        return 0.60
    elif 12.0 < yoe <= 15.0:
        return 0.55
    else:
        return 0.20


def compute_score(cand):
    """Returns (final_score, reasoning_string)."""
    profile = cand.get("profile", {})
    yoe = profile.get("years_of_experience", 0) or 0
    location = (profile.get("location", "") or "").lower()
    history = cand.get("career_history", []) or []
    skills = cand.get("skills", []) or []
    signals = cand.get("redrob_signals", {}) or {}

    # Skill scoring
    skill_score, skill_matches = score_skills(skills)

    # History scoring
    hist_score, hist_signals_found = score_history(history)

    # Experience multiplier
    exp_mult = experience_score(yoe)

    # Off-domain penalty
    skill_names_lc = [s.get("name", "").lower() for s in skills]
    has_off_domain = any(
        any(od in sn for od in OFF_DOMAIN_SKILLS)
        for sn in skill_names_lc
    )
    has_target_skills = skill_score > 20
    off_domain_penalty = 0.4 if (has_off_domain and not has_target_skills) else 1.0

    # Core score
    core = exp_mult * (10 + skill_score + hist_score) * off_domain_penalty

    # Behavioural multipliers
    mult = 1.0
    reasons = []

    # 1. Notice period
    np_days = int(signals.get("notice_period_days", 90) or 90)
    if np_days <= 15:
        mult *= 1.15
        reasons.append(f"notice={np_days}d")
    elif np_days <= 30:
        mult *= 1.08
        reasons.append(f"notice={np_days}d")
    elif np_days <= 60:
        mult *= 0.95
        reasons.append(f"notice={np_days}d")
    elif np_days <= 90:
        mult *= 0.75
        reasons.append(f"notice={np_days}d")
    else:
        mult *= 0.40
        reasons.append(f"notice={np_days}d(long)")

    # 2. Recency of activity
    stale_days = days_since(parse_date(signals.get("last_active_date")))
    if stale_days > 365:
        mult *= 0.30
        reasons.append("inactive>1yr")
    elif stale_days > 180:
        mult *= 0.60
        reasons.append("inactive>6mo")
    elif stale_days <= 30:
        mult *= 1.05
        reasons.append("very_active")

    # 3. Recruiter responsiveness
    rrr = float(signals.get("recruiter_response_rate") or 0.0)
    mult *= (0.5 + 0.5 * rrr)

    # 4. Open to work
    if signals.get("open_to_work_flag"):
        mult *= 1.08
        reasons.append("open_to_work")

    # 5. Location / relocation
    in_target_loc = any(tl in location for tl in TARGET_LOCATIONS)
    willing_relocate = signals.get("willing_to_relocate", False)
    if not in_target_loc and not willing_relocate:
        mult *= 0.20
        reasons.append("location_mismatch")
    elif in_target_loc:
        reasons.append(f"loc_ok:{location.split(',')[0].strip()}")

    # 6. Consulting-only career
    if is_consulting_only(history):
        mult *= 0.02
        reasons.append("consulting_only")

    # 7. Non-technical career history
    if not has_tech_title(history):
        mult *= 0.02
        reasons.append("no_tech_history")

    # 8. Job-hopper
    at = avg_tenure(history)
    if at <= 14 and len(history) >= 3:
        mult *= 0.50
        reasons.append(f"avg_tenure={at:.0f}mo")

    # 9. GitHub activity
    gh_score = signals.get("github_activity_score", -1)
    if isinstance(gh_score, (int, float)) and gh_score > 60:
        mult *= 1.05
        reasons.append(f"gh={gh_score}")

    # Final score
    final = core * mult

    # Build reasoning string
    top_skills = ", ".join(skill_matches[:5]) if skill_matches else "none"
    top_hist = ", ".join(hist_signals_found[:3]) if hist_signals_found else "none"
    reasoning = (
        f"yoe={yoe:.1f}|exp_mult={exp_mult:.2f}|"
        f"skill_score={skill_score:.1f}[{top_skills}]|"
        f"hist_score={hist_score:.1f}[{top_hist}]|"
        f"mult={mult:.3f}[{';'.join(reasons)}]|"
        f"final={final:.2f}"
    )

    return final, reasoning


# ──────────────────────────────────────────────
# Main pipeline
# ──────────────────────────────────────────────

def main():
    t0 = time.time()
    print("Loading and scoring candidates ...", flush=True)

    results = []   # (score, cid, cand, reasoning)
    total_read = 0
    honeypots_removed = 0

    with open(JSONL_PATH, "r", encoding="utf-8") as fh:
        for raw in fh:
            if not raw.strip():
                continue
            cand = json.loads(raw)
            total_read += 1

            if total_read % 10000 == 0:
                print(f"  ... {total_read:,} read ({time.time()-t0:.1f}s)", flush=True)

            if is_honeypot(cand):
                honeypots_removed += 1
                continue

            score, reasoning = compute_score(cand)
            results.append((score, cand["candidate_id"], cand, reasoning))

    elapsed_load = time.time() - t0
    print(
        f"  Read {total_read:,} candidates, removed {honeypots_removed} honeypots "
        f"in {elapsed_load:.1f}s",
        flush=True,
    )

    # Sort: descending score, tie-break ascending candidate_id
    results.sort(key=lambda x: (-x[0], x[1]))

    top = results[:TOP_N]

    # Write CSV
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as cf:
        writer = csv.writer(cf)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for rank, (score, cid, cand, reasoning) in enumerate(top, 1):
            writer.writerow([cid, rank, f"{score:.4f}", reasoning])

    elapsed_total = time.time() - t0
    print(f"\nDone in {elapsed_total:.1f}s -- wrote {OUTPUT_CSV}")
    print(f"\nTop 10 preview:")
    print(f"{'Rank':>4}  {'CandID':<16}  {'Score':>8}  {'Name':<22}  {'Title'}")
    print("-" * 100)
    for rank, (score, cid, cand, reasoning) in enumerate(top[:10], 1):
        p = cand["profile"]
        print(
            f"{rank:>4}  {cid:<16}  {score:>8.2f}  "
            f"{p['anonymized_name']:<22}  "
            f"{p['current_title']} @ {p['current_company']}"
        )


if __name__ == "__main__":
    main()
