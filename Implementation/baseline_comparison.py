import os
import json
import time
import datetime
import textstat
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from vector_store import load_vector_store

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model="llama-3.3-70b-versatile"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
THESIS_DIR = os.path.dirname(BASE_DIR)
LOGS_DIR = os.path.join(BASE_DIR, "baseline_logs")
os.makedirs(LOGS_DIR, exist_ok=True)

# Test claims for comparison
TEST_CLAIMS = [
    {"claim_id": "B001", "age": 45, "gender": "Female", "diagnosis": "Routine breast cancer screening", "procedure": "Mammography screening", "expected_decision": "APPROVE"},
    {"claim_id": "B002", "age": 25, "gender": "Female", "diagnosis": "Routine breast cancer screening", "procedure": "Mammography screening", "expected_decision": "DENY"},
    {"claim_id": "B003", "age": 55, "gender": "Male", "diagnosis": "Colorectal cancer screening", "procedure": "Colonoscopy screening", "expected_decision": "APPROVE"},
    {"claim_id": "B004", "age": 35, "gender": "Male", "diagnosis": "Colorectal cancer screening", "procedure": "Colonoscopy screening", "expected_decision": "DENY"},
    {"claim_id": "B005", "age": 50, "gender": "Female", "diagnosis": "Type 2 diabetes mellitus", "procedure": "Diabetes self-management training", "expected_decision": "APPROVE"},
    {"claim_id": "B006", "age": 45, "gender": "Male", "diagnosis": "Chronic heart failure", "procedure": "Cardiac rehabilitation", "expected_decision": "REQUIRES_PRIOR_AUTH"},
    {"claim_id": "B007", "age": 65, "gender": "Female", "diagnosis": "Osteoporosis screening", "procedure": "Bone density scan", "expected_decision": "APPROVE"},
    {"claim_id": "B008", "age": 40, "gender": "Male", "diagnosis": "Lower back pain", "procedure": "Physical therapy", "expected_decision": "REQUIRES_PRIOR_AUTH"},
    {"claim_id": "B009", "age": 70, "gender": "Female", "diagnosis": "Seasonal flu prevention", "procedure": "Influenza vaccination", "expected_decision": "APPROVE"},
    {"claim_id": "B010", "age": 50, "gender": "Female", "diagnosis": "Depression and anxiety disorder", "procedure": "Mental health counselling", "expected_decision": "REQUIRES_PRIOR_AUTH"},
]

# BASELINE 1: Single Agent No XAI
def run_baseline1(claim):
    prompt = f"""You are a healthcare insurance system.
Process this claim and give a decision.
Patient age: {claim['age']}, Gender: {claim['gender']}
Diagnosis: {claim['diagnosis']}, Procedure: {claim['procedure']}
Give APPROVE, DENY, or REQUIRES_PRIOR_AUTH with a brief reason."""

    response = llm.invoke(prompt)
    time.sleep(8)
    decision = response.content

    log_data = {
        "claim_id": claim["claim_id"],
        "claim": claim,
        "system": "Baseline1_SingleAgent_NoXAI",
        "decision": decision,
        "agent_logs": [],
        "policy_used": "",
        "criteria_check": "",
        "explanation": decision
    }

    log_path = os.path.join(LOGS_DIR, f"B1_{claim['claim_id']}.json")
    with open(log_path, "w") as f:
        json.dump(log_data, f, indent=2)

    return log_data

# BASELINE 2: Single Agent Basic Logging
def run_baseline2(claim):
    vs = load_vector_store()
    query = f"{claim['procedure']} {claim['diagnosis']}"
    results = vs.similarity_search(query, k=1)
    policy = results[0].page_content if results else ""
    policy_title = results[0].metadata["title"] if results else "Unknown"

    prompt = f"""You are a healthcare insurance system.
Policy: {policy[:500]}
Patient age: {claim['age']}, Gender: {claim['gender']}
Diagnosis: {claim['diagnosis']}, Procedure: {claim['procedure']}
Decision (APPROVE/DENY/REQUIRES_PRIOR_AUTH) and explanation:"""

    response = llm.invoke(prompt)
    time.sleep(8)
    decision = response.content

    log_entry = {
        "agent": "SingleAgent",
        "timestamp": datetime.datetime.now().isoformat(),
        "input": query,
        "output": decision[:100],
        "reasoning": ""
    }

    log_data = {
        "claim_id": claim["claim_id"],
        "claim": claim,
        "system": "Baseline2_SingleAgent_BasicLogging",
        "decision": decision,
        "agent_logs": [log_entry],
        "policy_used": policy_title,
        "criteria_check": "",
        "explanation": decision
    }

    log_path = os.path.join(LOGS_DIR, f"B2_{claim['claim_id']}.json")
    with open(log_path, "w") as f:
        json.dump(log_data, f, indent=2)

    return log_data

# SCORING
def score_log(log_data):
    agent_logs = log_data.get("agent_logs", [])
    total = max(len(agent_logs), 1)
    explanation = log_data.get("explanation", "")
    policy_used = log_data.get("policy_used", "")
    decision = log_data.get("decision", "")
    criteria_check = log_data.get("criteria_check", "")

    # D1 Agent Transparency
    i = sum(1 for a in agent_logs if a.get("input","").strip())
    r = sum(1 for a in agent_logs if a.get("reasoning","").strip())
    o = sum(1 for a in agent_logs if a.get("output","").strip())
    ragas = 0.85 if policy_used and any(
        w in explanation.lower() for w in ["policy","coverage","medicare","criteria"]
    ) else 0.5
    d1 = round((i/total)*2.5 + (r/total)*2.5 + (o/total)*2.5 + ragas*2.5, 2)

    # D2 Communication Clarity
    all_out = " ".join(a.get("output","") for a in agent_logs)
    flesch = max(0, min(100, textstat.flesch_reading_ease(all_out))) if all_out.strip() else 50
    handoff = (sum(1 for a in agent_logs if a.get("output","").strip()) / total) * 3.3
    info = 3.4 if len(agent_logs) >= 2 else 0
    d2 = round(min(handoff + (flesch/100)*3.3 + info, 10), 2)

    # D3 Outcome Attribution
    a3 = 3.3 if any(a.get("agent") == "Agent3_Decision" for a in agent_logs) else 0
    ev = 3.3 if policy_used and policy_used != "Unknown" else 0
    hal = 3.4 if any(d in decision.upper() for d in ["APPROVE","DENY","REQUIRES_PRIOR_AUTH"]) else 0
    d3 = round(min(a3 + ev + hal, 10), 2)

    # D4 Human Oversight
    ex = 3.3 if explanation.strip() else 0
    kw = sum(1 for k in ["policy","decision","coverage","claim"] if k in explanation.lower())
    comp = (kw/4)*3.3
    cite = 3.4 if policy_used and any(
        w in explanation.lower()
        for w in policy_used.lower().split()[:3] if len(w) > 3
    ) else 0
    d4 = round(min(ex + comp + cite, 10), 2)

    # D5 Temporal Accountability
    ts = (sum(1 for a in agent_logs if a.get("timestamp","").strip()) / total) * 2.5
    seq = 2.5 if total == 4 else 0
    idn = (sum(1 for a in agent_logs if a.get("agent","").strip()) / total) * 2.5
    expected = ["Agent1_Retriever","Agent2_Interpreter","Agent3_Decision","Agent4_Explanation"]
    trc = 2.5 if [a.get("agent","") for a in agent_logs] == expected else 0
    d5 = round(min(ts + seq + idn + trc, 10), 2)

    composite = round((d1+d2+d3+d4+d5)/5, 2)
    return d1, d2, d3, d4, d5, composite

# RUN COMPARISON
def run_comparison():
    print("="*70)
    print("BASELINE COMPARISON — XAI SCORECARD FRAMEWORK")
    print("="*70)

    b1_scores = []
    b2_scores = []

    print("\nRunning Baseline 1 — Single Agent No XAI...")
    for claim in TEST_CLAIMS:
        log = run_baseline1(claim)
        d1,d2,d3,d4,d5,comp = score_log(log)
        b1_scores.append((d1,d2,d3,d4,d5,comp))
        print(f"  {claim['claim_id']} | Composite: {comp}")

    print("\nRunning Baseline 2 — Single Agent Basic Logging...")
    for claim in TEST_CLAIMS:
        log = run_baseline2(claim)
        d1,d2,d3,d4,d5,comp = score_log(log)
        b2_scores.append((d1,d2,d3,d4,d5,comp))
        print(f"  {claim['claim_id']} | Composite: {comp}")

    # Your system scores from existing results
    your_scores = [9.62, 6.74, 10.0, 9.13, 10.0, 9.1]

    # Averages
    def avg(scores, idx):
        return round(sum(s[idx] for s in scores) / len(scores), 2)

    b1_avg = [avg(b1_scores,i) for i in range(6)]
    b2_avg = [avg(b2_scores,i) for i in range(6)]

    print("\n" + "="*70)
    print("COMPARISON RESULTS")
    print("="*70)
    print(f"\n{'Dimension':<30} {'Baseline1':>12} {'Baseline2':>12} {'Your System':>12}")
    print("-"*70)
    dims = ["D1 Agent Transparency","D2 Communication Clarity",
            "D3 Outcome Attribution","D4 Human Oversight",
            "D5 Temporal Accountability","COMPOSITE SCORE"]
    for i, dim in enumerate(dims):
        print(f"{dim:<30} {b1_avg[i]:>12} {b2_avg[i]:>12} {your_scores[i]:>12}")

    print("\n" + "="*70)
    print("IMPROVEMENT OVER BASELINE 1")
    print(f"  Composite: {b1_avg[5]} → {your_scores[5]} (+{round(your_scores[5]-b1_avg[5],2)} points)")
    print("IMPROVEMENT OVER BASELINE 2")
    print(f"  Composite: {b2_avg[5]} → {your_scores[5]} (+{round(your_scores[5]-b2_avg[5],2)} points)")

    # Save comparison results
    output = {
        "baseline1_avg": dict(zip(dims, b1_avg)),
        "baseline2_avg": dict(zip(dims, b2_avg)),
        "your_system": dict(zip(dims, your_scores))
    }
    out_path = os.path.join(THESIS_DIR, "Evaluation", "results", "comparison_results.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {out_path}")
    print("\nDone!")

if __name__ == "__main__":
    run_comparison()