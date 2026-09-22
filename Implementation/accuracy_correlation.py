import json
import os
import statistics

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
THESIS_DIR = os.path.dirname(BASE_DIR)
LOGS_DIR = os.path.join(BASE_DIR, "logs")
SCORES_FILE = os.path.join(THESIS_DIR, "Evaluation", "results", "scoring_results.json")

def load_scoring_results():
    with open(SCORES_FILE, "r") as f:
        return json.load(f)

def extract_decision(decision_text):
    decision_text = decision_text.upper()
    if "APPROVE" in decision_text:
        return "APPROVE"
    elif "DENY" in decision_text:
        return "DENY"
    elif "REQUIRES_PRIOR_AUTH" in decision_text:
        return "REQUIRES_PRIOR_AUTH"
    return "UNKNOWN"

def load_claim_decisions():
    decisions = {}
    for fname in os.listdir(LOGS_DIR):
        if not fname.endswith(".json"):
            continue
        if "REL_RUN" in fname or "DEMO" in fname:
            continue
        path = os.path.join(LOGS_DIR, fname)
        try:
            with open(path) as f:
                log = json.load(f)
            claim_id = log.get("claim_id", "")
            expected = log.get("claim", {}).get("expected_decision", "")
            actual = extract_decision(log.get("decision", ""))
            decisions[claim_id] = {
                "expected": expected,
                "actual": actual,
                "correct": expected == actual
            }
        except:
            continue
    return decisions

def pearson_correlation(x, y):
    n = len(x)
    if n < 2:
        return 0
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    numerator = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n))
    denom_x = sum((xi - mean_x)**2 for xi in x)
    denom_y = sum((yi - mean_y)**2 for yi in y)
    if denom_x == 0 or denom_y == 0:
        return 0
    return round(numerator / ((denom_x * denom_y) ** 0.5), 4)

def run_accuracy_correlation():
    print("="*60)
    print("ACCURACY CORRELATION TEST")
    print("Pearson correlation between composite score and accuracy")
    print("="*60)

    scores = load_scoring_results()
    decisions = load_claim_decisions()

    composite_scores = []
    accuracy_values = []
    correct = 0
    total = 0
    unknown = 0

    print(f"\nClaim ID       Expected         Actual           Match  Composite")
    print("-"*75)

    for result in scores:
        claim_id = result.get("claim_id", "")
        if "DEMO" in claim_id or "REL" in claim_id:
            continue

        composite = result.get("Composite_Score", 0)
        decision_info = decisions.get(claim_id, {})
        expected = decision_info.get("expected", "UNKNOWN")
        actual = decision_info.get("actual", "UNKNOWN")
        is_correct = decision_info.get("correct", False)

        if actual == "UNKNOWN":
            unknown += 1
            continue

        accuracy = 1 if is_correct else 0
        composite_scores.append(composite)
        accuracy_values.append(accuracy)

        total += 1
        if is_correct:
            correct += 1

        match = "✓" if is_correct else "✗"
        print(f"{claim_id:<15}{expected:<17}{actual:<17}{match:<7}{composite}")

    accuracy_rate = round((correct / total) * 100, 2) if total > 0 else 0
    correlation = pearson_correlation(composite_scores, accuracy_values)

    print("\n" + "="*60)
    print("ACCURACY CORRELATION RESULTS")
    print("="*60)
    print(f"\nTotal claims evaluated: {total}")
    print(f"Correct decisions:      {correct}")
    print(f"Incorrect decisions:    {total - correct}")
    print(f"Accuracy rate:          {accuracy_rate}%")
    print(f"\nPearson correlation (composite vs accuracy): {correlation}")

    if correlation >= 0.8:
        print(f"RESULT: STRONG POSITIVE CORRELATION ✓")
        print(f"CONCLUSION: Higher explainability correlates with")
        print(f"higher decision accuracy — hypothesis supported ✓")
    elif correlation >= 0.5:
        print(f"RESULT: MODERATE POSITIVE CORRELATION")
        print(f"CONCLUSION: Some relationship between explainability")
        print(f"and accuracy — partially supported")
    else:
        print(f"RESULT: WEAK CORRELATION")
        print(f"CONCLUSION: Limited relationship found")

    output = {
        "test": "Accuracy Correlation",
        "total_claims": total,
        "correct_decisions": correct,
        "accuracy_rate": accuracy_rate,
        "pearson_correlation": correlation,
        "composite_scores": composite_scores,
        "accuracy_values": accuracy_values,
        "interpretation": "Strong" if correlation >= 0.8 else "Moderate" if correlation >= 0.5 else "Weak"
    }

    out_path = os.path.join(THESIS_DIR, "Evaluation", "results", "accuracy_correlation_results.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {out_path}")
    print("\nDone!")

if __name__ == "__main__":
    run_accuracy_correlation()