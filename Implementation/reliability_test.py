import json
import os
import time
import textstat
import statistics
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langgraph_workflow import build_workflow, save_log, ClaimState

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
THESIS_DIR = os.path.dirname(BASE_DIR)
OUTPUT_DIR = os.path.join(BASE_DIR, "reliability_logs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Run same claim 10 times
TEST_CLAIM = {
    "claim_id": "RELIABILITY_TEST",
    "patient_id": "P_RELIABILITY",
    "age": 45,
    "gender": "Female",
    "diagnosis": "Routine breast cancer screening",
    "procedure": "Mammography screening",
    "expected_decision": "APPROVE"
}

def score_log(log_data):
    agent_logs = log_data.get("agent_logs", [])
    total = max(len(agent_logs), 1)
    explanation = log_data.get("explanation", "")
    policy_used = log_data.get("policy_used", "")
    decision = log_data.get("decision", "")

    i = sum(1 for a in agent_logs if a.get("input","").strip())
    r = sum(1 for a in agent_logs if a.get("reasoning","").strip())
    o = sum(1 for a in agent_logs if a.get("output","").strip())
    ragas = 0.85 if policy_used and any(
        w in explanation.lower() for w in ["policy","coverage","medicare","criteria"]
    ) else 0.5
    d1 = round((i/total)*2.5 + (r/total)*2.5 + (o/total)*2.5 + ragas*2.5, 2)

    all_out = " ".join(a.get("output","") for a in agent_logs)
    flesch = max(0, min(100, textstat.flesch_reading_ease(all_out))) if all_out.strip() else 50
    handoff = (sum(1 for a in agent_logs if a.get("output","").strip()) / total) * 3.3
    info = 3.4 if len(agent_logs) >= 2 and any(
        w in agent_logs[1].get("input","").lower()
        for w in agent_logs[0].get("output","").lower().split()[:5]
        if len(w) > 3
    ) else 1.7
    d2 = round(min(handoff + (flesch/100)*3.3 + info, 10), 2)

    a3 = 3.3 if any(a.get("agent") == "Agent3_Decision" for a in agent_logs) else 0
    ev = 3.3 if policy_used and policy_used != "Unknown" else 0
    hal = 3.4 if any(d in decision.upper() for d in ["APPROVE","DENY","REQUIRES_PRIOR_AUTH"]) else 0
    d3 = round(min(a3 + ev + hal, 10), 2)

    ex = 3.3 if explanation.strip() else 0
    kw = sum(1 for k in ["policy","decision","coverage","claim"] if k in explanation.lower())
    comp = (kw/4)*3.3
    cite = 3.4 if policy_used and any(
        w in explanation.lower()
        for w in policy_used.lower().split()[:3] if len(w) > 3
    ) else 1.0
    d4 = round(min(ex + comp + cite, 10), 2)

    ts = (sum(1 for a in agent_logs if a.get("timestamp","").strip()) / total) * 2.5
    seq = 2.5 if total == 4 else 0
    idn = (sum(1 for a in agent_logs if a.get("agent","").strip()) / total) * 2.5
    expected = ["Agent1_Retriever","Agent2_Interpreter","Agent3_Decision","Agent4_Explanation"]
    trc = 2.5 if [a.get("agent","") for a in agent_logs] == expected else 0
    d5 = round(min(ts + seq + idn + trc, 10), 2)

    composite = round((d1+d2+d3+d4+d5)/5, 2)
    return d1, d2, d3, d4, d5, composite

def coefficient_of_variation(values):
    if len(values) < 2:
        return 0
    mean = statistics.mean(values)
    if mean == 0:
        return 0
    std = statistics.stdev(values)
    return round((std / mean) * 100, 2)

def run_reliability_test(runs=10):
    print("="*60)
    print("RELIABILITY TEST")
    print(f"Running same claim {runs} times")
    print(f"Claim: {TEST_CLAIM['procedure']} Age {TEST_CLAIM['age']} {TEST_CLAIM['gender']}")
    print("="*60)

    workflow = build_workflow()
    all_scores = []

    for i in range(runs):
        print(f"\nRun {i+1}/{runs}...")
        claim = TEST_CLAIM.copy()
        claim["claim_id"] = f"REL_RUN_{i+1}"

        initial_state = ClaimState(
            claim=claim,
            policy="",
            policy_title="",
            criteria_check="",
            decision="",
            explanation="",
            log=[]
        )

        result = workflow.invoke(initial_state)

        log_data = {
            "claim_id": claim["claim_id"],
            "claim": claim,
            "policy_used": result["policy_title"],
            "criteria_check": result["criteria_check"],
            "decision": result["decision"],
            "explanation": result["explanation"],
            "agent_logs": result["log"]
        }

        log_path = os.path.join(OUTPUT_DIR, f"run_{i+1}.json")
        with open(log_path, "w") as f:
            json.dump(log_data, f, indent=2)

        d1,d2,d3,d4,d5,comp = score_log(log_data)
        all_scores.append((d1,d2,d3,d4,d5,comp))
        print(f"  D1:{d1} D2:{d2} D3:{d3} D4:{d4} D5:{d5} Composite:{comp}")

        if i < runs-1:
            time.sleep(10)

    print("\n" + "="*60)
    print("RELIABILITY RESULTS")
    print("="*60)

    dims = ["D1","D2","D3","D4","D5","Composite"]
    cv_results = {}

    for idx, dim in enumerate(dims):
        values = [s[idx] for s in all_scores]
        mean = round(statistics.mean(values), 2)
        std = round(statistics.stdev(values), 2) if len(values) > 1 else 0
        cv = coefficient_of_variation(values)
        status = "✓ RELIABLE" if cv < 10 else "✗ UNRELIABLE"
        cv_results[dim] = cv
        print(f"  {dim:<12} Mean:{mean:<8} Std:{std:<8} CV:{cv}%  {status}")

    overall_cv = round(statistics.mean(cv_results.values()), 2)
    print(f"\n  Overall CV: {overall_cv}%")
    if overall_cv < 10:
        print("  RESULT: SCORECARD IS RELIABLE ✓")
    else:
        print("  RESULT: SCORECARD NEEDS IMPROVEMENT")

    output = {
        "test": "Reliability",
        "runs": runs,
        "claim": TEST_CLAIM,
        "all_scores": all_scores,
        "cv_per_dimension": cv_results,
        "overall_cv": overall_cv,
        "reliable": overall_cv < 10
    }

    out_path = os.path.join(THESIS_DIR, "Evaluation", "results", "reliability_results.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {out_path}")
    print("\nDone!")

if __name__ == "__main__":
    run_reliability_test(runs=10)