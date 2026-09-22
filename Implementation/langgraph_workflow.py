import os
import json
import datetime
from typing import TypedDict
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from vector_store import load_vector_store

# LLM Setup
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model="llama-3.3-70b-versatile"
)

# State Definition
class ClaimState(TypedDict):
    claim: dict
    policy: str
    policy_title: str
    criteria_check: str
    decision: str
    explanation: str
    log: list

# Agent 1: Retriever
def agent_retriever(state: ClaimState) -> ClaimState:
    print("\n[Agent 1 - Retriever] Searching for relevant policy...")
    
    claim = state["claim"]
    query = f"{claim['procedure']} coverage {claim['diagnosis']} patient age {claim['age']}"
    
    vs = load_vector_store()
    results = vs.similarity_search(query, k=1)
    
    policy_text = results[0].page_content if results else "No policy found"
    policy_title = results[0].metadata["title"] if results else "Unknown"
    
    log_entry = {
        "agent": "Agent1_Retriever",
        "timestamp": datetime.datetime.now().isoformat(),
        "input": query,
        "output": policy_title,
        "reasoning": f"Searched vector store for: {query}"
    }
    
    print(f"[Agent 1] Found policy: {policy_title}")
    
    return {
        **state,
        "policy": policy_text,
        "policy_title": policy_title,
        "log": state["log"] + [log_entry]
    }

# Agent 2: Policy Interpreter
def agent_interpreter(state: ClaimState) -> ClaimState:
    print("\n[Agent 2 - Policy Interpreter] Checking criteria...")
    
    claim = state["claim"]
    policy = state["policy"]
    
    prompt = f"""You are a Medicare policy interpreter.

PATIENT CLAIM:
- Patient Age: {claim['age']}
- Gender: {claim['gender']}
- Diagnosis: {claim['diagnosis']}
- Procedure Requested: {claim['procedure']}

POLICY RULES:
{policy}

Check each criterion in the policy against this patient.
For each criterion write: CRITERION: [what it requires] | PATIENT: [patient value] | MET: YES or NO

End with OVERALL: ELIGIBLE or NOT ELIGIBLE"""

    response = llm.invoke(prompt)
    criteria_check = response.content
    
    log_entry = {
        "agent": "Agent2_Interpreter",
        "timestamp": datetime.datetime.now().isoformat(),
        "input": f"Policy: {state['policy_title']}, Patient age: {claim['age']}, Procedure: {claim['procedure']}",
        "output": criteria_check,
        "reasoning": "Checked each policy criterion against patient data"
    }
    
    print(f"[Agent 2] Criteria check complete")
    
    return {
        **state,
        "criteria_check": criteria_check,
        "log": state["log"] + [log_entry]
    }

# Agent 3: Decision Agent
def agent_decision(state: ClaimState) -> ClaimState:
    print("\n[Agent 3 - Decision Agent] Making decision...")
    
    prompt = f"""You are a Medicare claims decision agent.

Based on this criteria check:
{state['criteria_check']}

Give ONE decision only. Choose exactly one:
- APPROVE
- DENY  
- REQUIRES_PRIOR_AUTH

Then give one sentence reason.

Format:
DECISION: [your decision]
REASON: [one sentence]"""

    response = llm.invoke(prompt)
    decision = response.content
    
    log_entry = {
        "agent": "Agent3_Decision",
        "timestamp": datetime.datetime.now().isoformat(),
        "input": state["criteria_check"][:200],
        "output": decision,
        "reasoning": "Made final decision based on criteria check from Agent 2"
    }
    
    print(f"[Agent 3] Decision made")
    
    return {
        **state,
        "decision": decision,
        "log": state["log"] + [log_entry]
    }

# Agent 4: Explanation Agent
def agent_explanation(state: ClaimState) -> ClaimState:
    print("\n[Agent 4 - Explanation Agent] Writing explanation...")
    
    claim = state["claim"]
    
    prompt = f"""You are a patient communication specialist.

Write a clear explanation for the patient about their insurance claim decision.

Claim: {claim['procedure']} for {claim['diagnosis']}
Policy Used: {state['policy_title']}
Decision: {state['decision']}

Write 3-4 sentences. Use simple language. Cite the policy name.
Do not use medical jargon."""

    response = llm.invoke(prompt)
    explanation = response.content
    
    log_entry = {
        "agent": "Agent4_Explanation",
        "timestamp": datetime.datetime.now().isoformat(),
        "input": state["decision"],
        "output": explanation,
        "reasoning": "Translated decision into patient-friendly language citing policy"
    }
    
    print(f"[Agent 4] Explanation written")
    
    return {
        **state,
        "explanation": explanation,
        "log": state["log"] + [log_entry]
    }

# Build the Graph
def build_workflow():
    graph = StateGraph(ClaimState)
    
    graph.add_node("retriever", agent_retriever)
    graph.add_node("interpreter", agent_interpreter)
    graph.add_node("decision", agent_decision)
    graph.add_node("explanation", agent_explanation)
    
    graph.set_entry_point("retriever")
    graph.add_edge("retriever", "interpreter")
    graph.add_edge("interpreter", "decision")
    graph.add_edge("decision", "explanation")
    graph.add_edge("explanation", END)
    
    return graph.compile()

# Save JSON Log
def save_log(state: ClaimState, claim_id: str):
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    log_data = {
        "claim_id": claim_id,
        "claim": state["claim"],
        "policy_used": state["policy_title"],
        "criteria_check": state["criteria_check"],
        "decision": state["decision"],
        "explanation": state["explanation"],
        "agent_logs": state["log"]
    }
    
    log_path = os.path.join(log_dir, f"{claim_id}.json")
    with open(log_path, "w") as f:
        json.dump(log_data, f, indent=2)
    
    print(f"\nLog saved: {log_path}")
    return log_path

# Run One Claim
if __name__ == "__main__":
    test_claim = {
        "claim_id": "CLAIM_001",
        "patient_id": "P001",
        "age": 45,
        "gender": "Female",
        "diagnosis": "Routine breast cancer screening",
        "procedure": "Mammography screening",
        "expected_decision": "APPROVE"
    }
    
    print("="*60)
    print("HEALTHCARE INSURANCE CLAIM PROCESSING SYSTEM")
    print("="*60)
    print(f"Processing claim: {test_claim['claim_id']}")
    print(f"Patient: {test_claim['age']}yr {test_claim['gender']}")
    print(f"Procedure: {test_claim['procedure']}")
    print(f"Diagnosis: {test_claim['diagnosis']}")
    
    workflow = build_workflow()
    
    initial_state = ClaimState(
        claim=test_claim,
        policy="",
        policy_title="",
        criteria_check="",
        decision="",
        explanation="",
        log=[]
    )
    
    result = workflow.invoke(initial_state)
    
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    print(f"\nPolicy Used: {result['policy_title']}")
    print(f"\nDecision:\n{result['decision']}")
    print(f"\nExplanation:\n{result['explanation']}")
    print(f"\nTotal agents logged: {len(result['log'])}")
    
    save_log(result, test_claim["claim_id"])
    print("\nDone!")