import json
import os
import textstat
import datetime

# Load JSON Log
def load_log(log_path):
    with open(log_path, "r") as f:
        return json.load(f)

# Dimension 1: Agent Transparency
def score_dimension1(log_data):
    agent_logs = log_data.get("agent_logs", [])
    total_agents = len(agent_logs)

    if total_agents == 0:
        return 0

    input_logged = sum(1 for a in agent_logs if a.get("input", "").strip())
    reasoning_logged = sum(1 for a in agent_logs if a.get("reasoning", "").strip())
    output_logged = sum(1 for a in agent_logs if a.get("output", "").strip())

    input_score = (input_logged / total_agents) * 2.5
    reasoning_score = (reasoning_logged / total_agents) * 2.5
    output_score = (output_logged / total_agents) * 2.5

    # RAGAS faithfulness approximation
    # Check if policy was actually used in the decision
    policy_used = log_data.get("policy_used", "")
    decision = log_data.get("decision", "")
    explanation = log_data.get("explanation", "")

    if policy_used and policy_used != "Unknown":
        if any(word in explanation.lower() for word in ["policy", "coverage", "medicare", "criteria"]):
            ragas_score = 0.85
        else:
            ragas_score = 0.5
    else:
        ragas_score = 0.2

    ragas_points = ragas_score * 2.5

    total = input_score + reasoning_score + output_score + ragas_points
    return round(total, 2)

# Dimension 2: Inter-Agent Communication Clarity
def score_dimension2(log_data):
    agent_logs = log_data.get("agent_logs", [])
    total_agents = len(agent_logs)

    if total_agents == 0:
        return 0

    # Check all handoffs logged
    handoffs_logged = sum(1 for a in agent_logs if a.get("output", "").strip())
    handoff_score = (handoffs_logged / total_agents) * 3.3

    # Readability of agent outputs using textstat
    all_outputs = " ".join(
        a.get("output", "") for a in agent_logs if a.get("output", "")
    )
    if all_outputs.strip():
        flesch_score = textstat.flesch_reading_ease(all_outputs)
        flesch_score = max(0, min(100, flesch_score))
        readability_points = (flesch_score / 100) * 3.3
    else:
        readability_points = 0

    # Information preservation check
    # Check Agent 1 output appears in Agent 2 input
    info_preserved = 0
    if len(agent_logs) >= 2:
        agent1_output = agent_logs[0].get("output", "").lower()
        agent2_input = agent_logs[1].get("input", "").lower()
        if agent1_output and any(
            word in agent2_input
            for word in agent1_output.split()[:5]
            if len(word) > 3
        ):
            info_preserved = 3.4
        else:
            info_preserved = 1.7

    total = handoff_score + readability_points + info_preserved
    return round(min(total, 10), 2)

# Dimension 3: Outcome Attribution
def score_dimension3(log_data):
    agent_logs = log_data.get("agent_logs", [])
    decision = log_data.get("decision", "")
    explanation = log_data.get("explanation", "")
    policy_used = log_data.get("policy_used", "")

    # Decision links to specific agent
    decision_agent_exists = any(
        a.get("agent") == "Agent3_Decision" for a in agent_logs
    )
    agent_attribution = 3.3 if decision_agent_exists else 0

    # Decision links to specific evidence
    evidence_linked = 0
    if policy_used and policy_used != "Unknown":
        if policy_used.lower()[:10] in explanation.lower():
            evidence_linked = 3.3
        else:
            evidence_linked = 1.5

    # Hallucination check
    # Check if decision is one of the valid options
    valid_decisions = ["APPROVE", "DENY", "REQUIRES_PRIOR_AUTH"]
    decision_clean = decision.upper()
    hallucination_free = any(d in decision_clean for d in valid_decisions)
    hallucination_score = 3.4 if hallucination_free else 0

    total = agent_attribution + evidence_linked + hallucination_score
    return round(min(total, 10), 2)

# Dimension 4: Human Oversight Integration
def score_dimension4(log_data):
    explanation = log_data.get("explanation", "")
    policy_used = log_data.get("policy_used", "")
    decision = log_data.get("decision", "")

    # Explanation exists
    explanation_exists = 3.3 if explanation.strip() else 0

    # Required fields present in explanation
    required_keywords = ["policy", "decision", "coverage", "claim"]
    keywords_found = sum(
        1 for k in required_keywords
        if k.lower() in explanation.lower()
    )
    completeness_score = (keywords_found / len(required_keywords)) * 3.3

    # Policy cited in explanation
    policy_cited = 0
    if policy_used and policy_used != "Unknown":
        policy_words = policy_used.lower().split()[:3]
        if any(word in explanation.lower() for word in policy_words if len(word) > 3):
            policy_cited = 3.4
        else:
            policy_cited = 1.0

    total = explanation_exists + completeness_score + policy_cited
    return round(min(total, 10), 2)

# Dimension 5: Temporal Accountability
def score_dimension5(log_data):
    agent_logs = log_data.get("agent_logs", [])
    total_agents = len(agent_logs)

    if total_agents == 0:
        return 0

    # Every step timestamped
    timestamped = sum(1 for a in agent_logs if a.get("timestamp", "").strip())
    timestamp_score = (timestamped / total_agents) * 2.5

    # Sequence reconstructable
    sequence_ok = 2.5 if total_agents == 4 else 0

    # Each agent action time identifiable
    identifiable = sum(1 for a in agent_logs if a.get("agent", "").strip())
    identifiable_score = (identifiable / total_agents) * 2.5

    # Error propagation traceable
    # Check all 4 agents are present in correct order
    expected_order = [
        "Agent1_Retriever",
        "Agent2_Interpreter",
        "Agent3_Decision",
        "Agent4_Explanation"
    ]
    actual_order = [a.get("agent", "") for a in agent_logs]
    traceable = 2.5 if actual_order == expected_order else 0

    total = timestamp_score + sequence_ok + identifiable_score + traceable
    return round(min(total, 10), 2)

# Composite Score
def compute_composite(d1, d2, d3, d4, d5):
    return round((d1 + d2 + d3 + d4 + d5) / 5, 2)

# Score One Claim
def score_claim(log_path):
    log_data = load_log(log_path)

    d1 = score_dimension1(log_data)
    d2 = score_dimension2(log_data)
    d3 = score_dimension3(log_data)
    d4 = score_dimension4(log_data)
    d5 = score_dimension5(log_data)
    composite = compute_composite(d1, d2, d3, d4, d5)

    result = {
        "claim_id": log_data.get("claim_id", "unknown"),
        "expected_decision": log_data.get("claim", {}).get("expected_decision", ""),
        "actual_decision": log_data.get("decision", ""),
        "D1_Agent_Transparency": d1,
        "D2_Communication_Clarity": d2,
        "D3_Outcome_Attribution": d3,
        "D4_Human_Oversight": d4,
        "D5_Temporal_Accountability": d5,
        "Composite_Score": composite
    }

    return result

# Score All Claims
def score_all_claims():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    logs_dir = os.path.join(BASE_DIR, "logs")
    results = []

    log_files = [f for f in os.listdir(logs_dir) 
             if f.endswith(".json") 
             and not f.startswith("REL_")
             and not f.startswith("DV")
             and "DEMO" not in f
             and "RELIABILITY" not in f]
    print(f"Found {len(log_files)} log files to score")

    for log_file in sorted(log_files):
        log_path = os.path.join(logs_dir, log_file)
        result = score_claim(log_path)
        results.append(result)
        print(f"{result['claim_id']} | Composite: {result['Composite_Score']} | "
              f"D1:{result['D1_Agent_Transparency']} "
              f"D2:{result['D2_Communication_Clarity']} "
              f"D3:{result['D3_Outcome_Attribution']} "
              f"D4:{result['D4_Human_Oversight']} "
              f"D5:{result['D5_Temporal_Accountability']}")

    return results

# Run
if __name__ == "__main__":
    print("="*60)
    print("SCORING ENGINE")
    print("="*60)
    
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    THESIS_DIR = os.path.dirname(BASE_DIR)
    logs_dir = os.path.join(BASE_DIR, "logs")

    if not os.path.exists(logs_dir) or len(os.listdir(logs_dir)) == 0:
        print("No log files found.")
        print("Run langgraph_workflow.py first to generate a log.")
    
    else:
        results = score_all_claims()
        
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        
        avg_composite = sum(r["Composite_Score"] for r in results) / len(results)
        avg_d1 = sum(r["D1_Agent_Transparency"] for r in results) / len(results)
        avg_d2 = sum(r["D2_Communication_Clarity"] for r in results) / len(results)
        avg_d3 = sum(r["D3_Outcome_Attribution"] for r in results) / len(results)
        avg_d4 = sum(r["D4_Human_Oversight"] for r in results) / len(results)
        avg_d5 = sum(r["D5_Temporal_Accountability"] for r in results) / len(results)
        
        print(f"Claims scored: {len(results)}")
        print(f"Average D1 Agent Transparency:        {round(avg_d1, 2)}/10")
        print(f"Average D2 Communication Clarity:     {round(avg_d2, 2)}/10")
        print(f"Average D3 Outcome Attribution:       {round(avg_d3, 2)}/10") 
        print(f"Average D4 Human Oversight:           {round(avg_d4, 2)}/10")
        print(f"Average D5 Temporal Accountability:   {round(avg_d5, 2)}/10")
        print(f"Average Composite Score:              {round(avg_composite, 2)}/10")

    # Save results
    output_path = os.path.join(THESIS_DIR, "Evaluation", "results", "scoring_results.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")
    print("\nDone!")