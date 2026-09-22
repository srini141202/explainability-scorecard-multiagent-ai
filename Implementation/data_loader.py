import pandas as pd
import os

#Path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
THESIS_DIR = os.path.dirname(BASE_DIR)
BASE = os.path.join(THESIS_DIR, "Datasets")

SYNTHEA_PATH = os.path.join(BASE, "Patient_Data_Synthea", "synthea_csv")
LCD_PATH     = os.path.join(BASE, "Policy_CMS_LCD", "current_lcd_csv")

# Load patient data
def load_patients():
    path = os.path.join(SYNTHEA_PATH, "patients.csv")
    df = pd.read_csv(path)
    print(f"Patients loaded: {len(df)} rows")
    return df

def load_conditions():
    path = os.path.join(SYNTHEA_PATH, "conditions.csv")
    df = pd.read_csv(path)
    print(f"Conditions loaded: {len(df)} rows")
    return df

def load_procedures():
    path = os.path.join(SYNTHEA_PATH, "procedures.csv")
    df = pd.read_csv(path)
    print(f"Procedures loaded: {len(df)} rows")
    return df

# Load policy data
def load_lcd_policies():
    path = os.path.join(LCD_PATH, "lcd_cleaned.csv")
    df = pd.read_csv(path, encoding="utf-8", on_bad_lines="skip")
    print(f"LCD policies loaded: {len(df)} rows")
    return df

#  Test
if __name__ == "__main__":
    patients   = load_patients()
    conditions = load_conditions()
    procedures = load_procedures()
    policies   = load_lcd_policies()

    print("\nPatient columns:",   list(patients.columns))
    print("Condition columns:",  list(conditions.columns))
    print("Procedure columns:",  list(procedures.columns))
    print("Policy columns:",     list(policies.columns))