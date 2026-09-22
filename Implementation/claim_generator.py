import os
import json
import random
import pandas as pd

random.seed(42)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
THESIS_DIR = os.path.dirname(BASE_DIR)
BASE = os.path.join(THESIS_DIR, "Datasets")
SYNTHEA_PATH = os.path.join(BASE, "Patient_Data_Synthea", "synthea_csv")
CLAIMS_OUTPUT = os.path.join(THESIS_DIR, "Authored_Claims", "claims_generated.json")

# Load Data
def load_data():
    patients = pd.read_csv(os.path.join(SYNTHEA_PATH, "patients.csv"))
    conditions = pd.read_csv(os.path.join(SYNTHEA_PATH, "conditions.csv"))
    procedures = pd.read_csv(os.path.join(SYNTHEA_PATH, "procedures.csv"))
    print(f"Patients: {len(patients)}, Conditions: {len(conditions)}, Procedures: {len(procedures)}")
    return patients, conditions, procedures

# Calculate Age
def calculate_age(birthdate_str):
    try:
        birth_year = int(str(birthdate_str)[:4])
        return 2026 - birth_year
    except:
        return None

# Claim Templates
CLAIM_TEMPLATES = [
    {
        "procedure": "Mammography screening",
        "diagnosis": "Routine breast cancer screening",
        "gender_required": "F",
        "min_age": 40,
        "max_age": 100,
        "expected_decision": "APPROVE"
    },
    {
        "procedure": "Mammography screening",
        "diagnosis": "Routine breast cancer screening",
        "gender_required": "F",
        "min_age": 20,
        "max_age": 39,
        "expected_decision": "DENY"
    },
    {
        "procedure": "Colonoscopy screening",
        "diagnosis": "Colorectal cancer screening",
        "gender_required": "ANY",
        "min_age": 50,
        "max_age": 100,
        "expected_decision": "APPROVE"
    },
    {
        "procedure": "Colonoscopy screening",
        "diagnosis": "Colorectal cancer screening",
        "gender_required": "ANY",
        "min_age": 18,
        "max_age": 49,
        "expected_decision": "DENY"
    },
    {
        "procedure": "Diabetes self-management training",
        "diagnosis": "Type 2 diabetes mellitus",
        "gender_required": "ANY",
        "min_age": 18,
        "max_age": 100,
        "expected_decision": "APPROVE"
    },
    {
        "procedure": "Cardiac rehabilitation",
        "diagnosis": "Chronic heart failure",
        "gender_required": "ANY",
        "min_age": 18,
        "max_age": 100,
        "expected_decision": "REQUIRES_PRIOR_AUTH"
    },
    {
        "procedure": "Physical therapy",
        "diagnosis": "Lower back pain",
        "gender_required": "ANY",
        "min_age": 18,
        "max_age": 100,
        "expected_decision": "REQUIRES_PRIOR_AUTH"
    },
    {
        "procedure": "MRI brain scan",
        "diagnosis": "Severe headache investigation",
        "gender_required": "ANY",
        "min_age": 18,
        "max_age": 100,
        "expected_decision": "REQUIRES_PRIOR_AUTH"
    },
    {
        "procedure": "Hip replacement surgery",
        "diagnosis": "Severe osteoarthritis",
        "gender_required": "ANY",
        "min_age": 50,
        "max_age": 100,
        "expected_decision": "APPROVE"
    },
    {
        "procedure": "Hip replacement surgery",
        "diagnosis": "Mild joint discomfort",
        "gender_required": "ANY",
        "min_age": 18,
        "max_age": 49,
        "expected_decision": "DENY"
    },
    {
        "procedure": "Hip replacement surgery",
        "diagnosis": "Mild joint discomfort",
        "gender_required": "ANY",
        "min_age": 18,
        "max_age": 49,
        "expected_decision": "DENY"
    },
    {
        "procedure": "Diabetic retinopathy eye examination",
        "diagnosis": "Type 2 diabetes mellitus with eye complications",
        "gender_required": "ANY",
        "min_age": 18,
        "max_age": 100,
        "expected_decision": "APPROVE"
    },
    {
        "procedure": "Influenza vaccination",
        "diagnosis": "Seasonal flu prevention",
        "gender_required": "ANY",
        "min_age": 65,
        "max_age": 100,
        "expected_decision": "APPROVE"
    },
    {
        "procedure": "Bone density scan",
        "diagnosis": "Osteoporosis screening",
        "gender_required": "F",
        "min_age": 65,
        "max_age": 100,
        "expected_decision": "APPROVE"
    },
    {
        "procedure": "Mental health counselling",
        "diagnosis": "Depression and anxiety disorder",
        "gender_required": "ANY",
        "min_age": 18,
        "max_age": 100,
        "expected_decision": "REQUIRES_PRIOR_AUTH"
    },
    {
        "procedure": "Kidney dialysis",
        "diagnosis": "End stage renal disease",
        "gender_required": "ANY",
        "min_age": 18,
        "max_age": 100,
        "expected_decision": "APPROVE"
    }
]

# Generate Claims
def generate_claims(target=100):
    patients, conditions, procedures = load_data()
    claims = []
    claim_counter = 1

    patient_list = patients.to_dict('records')
    random.shuffle(patient_list)

    for patient in patient_list:
        if len(claims) >= target:
            break

        age = calculate_age(patient.get("BIRTHDATE", ""))
        gender = str(patient.get("GENDER", "")).strip().upper()
        patient_id = str(patient.get("Id", ""))

        if age is None:
            continue

        for template in CLAIM_TEMPLATES:
            if len(claims) >= target:
                break

            gender_ok = (
                template["gender_required"] == "ANY" or
                template["gender_required"] == gender
            )
            age_ok = template["min_age"] <= age <= template["max_age"]

            if gender_ok and age_ok:
                claim = {
                    "claim_id": f"CLAIM_{claim_counter:03d}",
                    "patient_id": patient_id,
                    "age": age,
                    "gender": "Female" if gender == "F" else "Male",
                    "diagnosis": template["diagnosis"],
                    "procedure": template["procedure"],
                    "expected_decision": template["expected_decision"]
                }
                claims.append(claim)
                claim_counter += 1

    print(f"\nTotal claims generated: {len(claims)}")

    approve_count = sum(1 for c in claims if c["expected_decision"] == "APPROVE")
    deny_count = sum(1 for c in claims if c["expected_decision"] == "DENY")
    auth_count = sum(1 for c in claims if c["expected_decision"] == "REQUIRES_PRIOR_AUTH")

    print(f"APPROVE: {approve_count}")
    print(f"DENY: {deny_count}")
    print(f"REQUIRES_PRIOR_AUTH: {auth_count}")

    return claims

# Save Claims
def save_claims(claims):
    os.makedirs(os.path.dirname(CLAIMS_OUTPUT), exist_ok=True)
    with open(CLAIMS_OUTPUT, "w") as f:
        json.dump(claims, f, indent=2)
    print(f"\nClaims saved to: {CLAIMS_OUTPUT}")

# Preview
def preview_claims(claims, n=5):
    print(f"\nFirst {n} claims:")
    for c in claims[:n]:
        print(f"  {c['claim_id']} | Age:{c['age']} | {c['gender']} | {c['procedure']} | {c['expected_decision']}")

# Run
if __name__ == "__main__":
    print("Generating claims from Synthea patient data...")
    claims = generate_claims(target=120)
    preview_claims(claims)
    save_claims(claims)
    print("\nClaim generation complete!")