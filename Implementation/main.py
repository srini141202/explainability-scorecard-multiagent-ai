import os
import json
import time
from dotenv import load_dotenv
load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
THESIS_DIR = os.path.dirname(BASE_DIR)
LOGS_DIR = os.path.join(BASE_DIR, "logs")
CLAIMS_FILE = os.path.join(THESIS_DIR, "Authored_Claims", "claims_generated.json")
VECTORSTORE_PATH = os.path.join(BASE_DIR, "vectorstore_db")
RESULTS_DIR = os.path.join(THESIS_DIR, "Evaluation", "results")

os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

def divider(title):
    print(f"\n{'='*65}")
    print(f"  {title}")
    print(f"{'='*65}")

print("="*65)
print("  XAI MULTI-AGENT HEALTHCARE INSURANCE CLAIM PROCESSING")
print("  MSc AI Thesis — Srinivasan Dillikumar — NCI 2026")
print("="*65)

# STEP 1: DATA LOADING 
divider("STEP 1 — DATA LOADING (data_loader.py)")
from data_loader import (load_patients, load_conditions,
                          load_procedures, load_lcd_policies)
patients   = load_patients()
conditions = load_conditions()
procedures = load_procedures()
policies   = load_lcd_policies()
print(f"\n✓ Patients: {len(patients)} rows")
print(f"✓ Conditions: {len(conditions)} rows")
print(f"✓ Procedures: {len(procedures)} rows")
print(f"✓ LCD Policies: {len(policies)} rows")

# STEP 2: VECTOR STORE
divider("STEP 2 — VECTOR STORE (vector_store.py)")
if os.path.exists(VECTORSTORE_PATH) and os.listdir(VECTORSTORE_PATH):
    print("Vector store already exists — loading from disk...")
    from vector_store import load_vector_store
    vs = load_vector_store()
    print("✓ Vector store loaded")
    print("✓ 947 CMS Medicare policies ready for semantic search")
else:
    print("Building vector store for first time — please wait 2-3 mins...")
    from vector_store import build_vector_store
    vs = build_vector_store()
    print("✓ Vector store built and saved to disk")

# STEP 3: CLAIM GENERATION
divider("STEP 3 — CLAIM GENERATION (claim_generator.py)")
if os.path.exists(CLAIMS_FILE):
    with open(CLAIMS_FILE) as f:
        claims = json.load(f)
    print(f"Claims already exist — loaded {len(claims)} claims")
    approve = sum(1 for c in claims if c["expected_decision"] == "APPROVE")
    deny    = sum(1 for c in claims if c["expected_decision"] == "DENY")
    auth    = sum(1 for c in claims if c["expected_decision"] == "REQUIRES_PRIOR_AUTH")
    print(f"  APPROVE: {approve} | DENY: {deny} | REQUIRES_PRIOR_AUTH: {auth}")
    print(f"  Templates: 15 procedure types | 8 medical specialties")
    print(f"  Reproducibility: random.seed(42)")
else:
    print("Generating 120 claims from Synthea patient data...")
    from claim_generator import generate_claims, save_claims
    claims = generate_claims(target=120)
    save_claims(claims)
    print(f"✓ {len(claims)} claims generated and saved")

# STEP 4: FOUR AGENT PIPELINE
divider("STEP 4 — FOUR AGENT LANGGRAPH PIPELINE (langgraph_workflow.py)")
print("  Agent 1 — Retriever    : searches 947 policies semantically")
print("  Agent 2 — Interpreter  : checks each policy criterion")
print("  Agent 3 — Decision     : APPROVE / DENY / REQUIRES_PRIOR_AUTH")
print("  Agent 4 — Explanation  : plain English output for patient")
print("  ClaimState             : shared memory across all agents")
print("  JSON Log               : saved per claim with timestamps")

from langgraph_workflow import build_workflow, save_log, ClaimState

# STEP 4b: BATCH RUNNER
divider("STEP 4b — BATCH PROCESSING (batch_runner.py)")
existing_logs = set(
    f.replace(".json", "") for f in os.listdir(LOGS_DIR)
    if f.endswith(".json")
)
unprocessed = [c for c in claims if c["claim_id"] not in existing_logs]

if not unprocessed:
    processed_count = len([f for f in os.listdir(LOGS_DIR) if f.endswith(".json")])
    print(f"All claims already processed through pipeline")
    print(f"✓ {processed_count} JSON log files exist in logs folder")
else:
    print(f"Processing {len(unprocessed)} unprocessed claims...")
    print("Each claim passes through all 4 agents automatically")
    workflow = build_workflow()
    success = 0
    failed  = 0
    for i, claim in enumerate(unprocessed):
        try:
            print(f"  [{i+1}/{len(unprocessed)}] {claim['claim_id']} — "
                  f"{claim['procedure']} — Age {claim['age']} {claim['gender']}")
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
            save_log(result, claim["claim_id"])
            success += 1
            time.sleep(5)
        except Exception as e:
            print(f"  ✗ Failed: {str(e)[:60]}")
            failed += 1
            time.sleep(5)
    print(f"\n✓ Batch complete — Success: {success} | Failed: {failed}")

# STEP 5: SCORING ENGINE
divider("STEP 5 — EXPLAINABILITY SCORING (scoring_engine.py)")
print("Reading JSON logs and computing 5 dimension scores automatically")
print("No human raters — fully automated evaluation")
from scoring_engine import score_all_claims
results = score_all_claims()

avg_composite = 0
avg_d1 = avg_d2 = avg_d3 = avg_d4 = avg_d5 = 0

if results:
    avg_composite = round(sum(r["Composite_Score"] for r in results) / len(results), 2)
    avg_d1 = round(sum(r["D1_Agent_Transparency"] for r in results) / len(results), 2)
    avg_d2 = round(sum(r["D2_Communication_Clarity"] for r in results) / len(results), 2)
    avg_d3 = round(sum(r["D3_Outcome_Attribution"] for r in results) / len(results), 2)
    avg_d4 = round(sum(r["D4_Human_Oversight"] for r in results) / len(results), 2)
    avg_d5 = round(sum(r["D5_Temporal_Accountability"] for r in results) / len(results), 2)

    print(f"\n  D1 Agent Transparency:        {avg_d1}/10")
    print(f"  D2 Communication Clarity:     {avg_d2}/10")
    print(f"  D3 Outcome Attribution:       {avg_d3}/10")
    print(f"  D4 Human Oversight:           {avg_d4}/10")
    print(f"  D5 Temporal Accountability:   {avg_d5}/10")
    print(f"\n  ✓ Composite Score: {avg_composite}/10 across {len(results)} claims")

# STEP 6: FINAL SUMMARY
divider("STEP 6 — FINAL SUMMARY")
print(f"""
  ┌───────────────────────────────────────────────────────────────┐
  │  IMPLEMENTATION COMPLETE                                      │
  │                                                               │
  │  Data Sources                                                 │
  │    Synthea patients  : {len(patients)} synthetic records      │
  │    CMS LCD policies  : {len(policies)} Medicare rules         │
  │                                                               │
  │  Claim Generation                                             │
  │    Total claims      : {len(claims)} generated automatically  │
  │    Procedure types   : 15 across 8 specialties                │
  │                                                               │
  │  Four Agent Pipeline                                          │
  │    Agent 1 Retriever    — semantic policy search              │
  │    Agent 2 Interpreter  — criteria checking                   │
  │    Agent 3 Decision     — APPROVE/DENY/AUTH                   │
  │    Agent 4 Explanation  — plain English output                │
  │                                                               │
  │  Explainability Scorecard                                     │
  │    Claims scored     : {len(results) if results else 0}       │
  │    D1 Transparency   : {avg_d1}/10                            │
  │    D2 Clarity        : {avg_d2}/10                            │
  │    D3 Attribution    : {avg_d3}/10                            │
  │    D4 Oversight      : {avg_d4}/10                            │
  │    D5 Temporal       : {avg_d5}/10                            │
  │    Composite Score   : {avg_composite}/10                     │
  │                                                               │
  │  Run evaluation tests separately                              │
  │    python reliability_test.py                                 │
  │    python discriminant_validity.py                            │
  │    python baseline_comparison.py                              │
  │    python accuracy_correlation.py                             │
  └───────────────────────────────────────────────────────────────┘
""")
print("="*65)
print("  System ready. All steps complete.")
print("="*65)