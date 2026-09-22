import json
import os
import time
from langgraph_workflow import build_workflow, save_log, ClaimState

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
THESIS_DIR = os.path.dirname(BASE_DIR)
CLAIMS_FILE = os.path.join(THESIS_DIR, "Authored_Claims", "claims_generated.json")
LOGS_DIR = os.path.join(BASE_DIR, "logs")

def load_claims():
    with open(CLAIMS_FILE, "r") as f:
        return json.load(f)

def already_processed(claim_id):
    log_path = os.path.join(LOGS_DIR, f"{claim_id}.json")
    return os.path.exists(log_path)

def run_batch():
    claims = load_claims()
    total = len(claims)
    print(f"Total claims to process: {total}")
    print(f"Logs will be saved to: {LOGS_DIR}")
    print("="*50)

    workflow = build_workflow()
    success = 0
    skipped = 0
    failed = 0

    for i, claim in enumerate(claims):
        claim_id = claim["claim_id"]

        if already_processed(claim_id):
            print(f"[{i+1}/{total}] SKIP {claim_id} — already processed")
            skipped += 1
            continue

        try:
            print(f"[{i+1}/{total}] Processing {claim_id} — {claim['procedure']} — Age {claim['age']} {claim['gender']}")

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
            save_log(result, claim_id)
            success += 1
            print(f"         ✓ Done — Decision: {result['decision'][:30]}")
            time.sleep(10)

        except Exception as e:
            print(f"         ✗ Failed — {str(e)[:60]}")
            failed += 1
            time.sleep(10)

    print("\n" + "="*50)
    print(f"BATCH COMPLETE")
    print(f"Success:  {success}")
    print(f"Skipped:  {skipped}")
    print(f"Failed:   {failed}")
    print(f"Total:    {total}")

if __name__ == "__main__":
    run_batch()