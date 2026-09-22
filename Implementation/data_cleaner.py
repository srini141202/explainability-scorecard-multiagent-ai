import os
import re
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
THESIS_DIR = os.path.dirname(BASE_DIR)
LCD_PATH = os.path.join(THESIS_DIR, "Datasets", "Policy_CMS_LCD", "current_lcd_csv")

INPUT_FILE = os.path.join(LCD_PATH, "lcd.csv")
OUTPUT_FILE = os.path.join(LCD_PATH, "lcd_cleaned.csv")

def clean_text(text):
    if pd.isna(text) or str(text).strip() == '':
        return ''
    text = str(text)
    text = re.sub(r'<[^>]+>', ' ', text)
    entities = {
        '&rsquo;': "'", '&lsquo;': "'",
        '&ldquo;': '"', '&rdquo;': '"',
        '&amp;': 'and', '&nbsp;': ' ',
        '&lt;': '<', '&gt;': '>',
        '&#39;': "'", '&mdash;': '-',
        '&ndash;': '-', '&bull;': '•',
        '&hellip;': '...', '&copy;': '(c)',
        '&reg;': '(R)', '&trade;': '(TM)',
        '&eacute;': 'e', '&ouml;': 'o',
        '&uuml;': 'u', '&auml;': 'a',
        '&aacute;': 'a', '&iacute;': 'i',
        '&oacute;': 'o', '&uacute;': 'u',
    }
    for entity, replacement in entities.items():
        text = text.replace(entity, replacement)
    text = re.sub(r'&#\d+;', '', text)
    text = re.sub(r'&[a-zA-Z]+;', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text

def is_useful_indication(text):
    if not text or len(text) < 50:
        return False
    lower = text.lower().strip()
    skip_phrases = [
        "covered indications",
        "see policy",
        "refer to policy",
        "not covered",
        "coverage policy",
        "coverage rule",
        "this is a noncoverage policy",
    ]
    for phrase in skip_phrases:
        if lower == phrase or (lower.startswith(phrase) and len(text) < 150):
            return False
    return True

def clean_lcd():
    print("="*60)
    print("LCD POLICY DATA CLEANER")
    print("Reading from raw lcd.csv — saving to lcd_cleaned.csv")
    print("="*60)

    print(f"\nLoading raw file...")
    df = pd.read_csv(INPUT_FILE, encoding="latin-1", on_bad_lines="skip")
    print(f"Total rows in raw file: {len(df)}")

    html_before = df['indication'].dropna().apply(
        lambda x: bool(re.search(r'<[^>]+>', str(x)))
    ).sum()
    entities_before = df['indication'].dropna().str.contains(
        '&rsquo;|&amp;|&nbsp;|&lt;|&gt;', na=False
    ).sum()

    print(f"\nBEFORE CLEANING:")
    print(f"  Contains HTML tags:      {html_before} rows")
    print(f"  Contains HTML entities:  {entities_before} rows")

    sample = df[df['indication'].notna()].iloc[1]
    print(f"\nSAMPLE BEFORE (Policy: {str(sample['title'])[:50]}):")
    print(f"  {str(sample['indication'])[:250]}")

    print(f"\nCleaning indication column...")
    df['indication'] = df['indication'].apply(clean_text)

    print("Filtering rows with meaningful indication text...")
    df_clean = df[df['indication'].apply(is_useful_indication)].copy()

    html_after = df_clean['indication'].apply(
        lambda x: bool(re.search(r'<[^>]+>', str(x)))
    ).sum()
    entities_after = df_clean['indication'].str.contains(
        '&rsquo;|&amp;|&nbsp;', na=False
    ).sum()

    print(f"\nAFTER CLEANING:")
    print(f"  Rows with valid indication: {len(df_clean)}")
    print(f"  Rows removed:               {len(df) - len(df_clean)}")
    print(f"  HTML tags remaining:        {html_after}")
    print(f"  HTML entities remaining:    {entities_after}")

    clean_sample = df_clean[df_clean['title'] == sample['title']]
    if len(clean_sample) > 0:
        print(f"\nSAMPLE AFTER (same policy):")
        print(f"  {str(clean_sample.iloc[0]['indication'])[:250]}")

    print(f"\nSAMPLE CLEAN POLICIES:")
    for i, row in df_clean.head(3).iterrows():
        print(f"\n  Title: {row['title']}")
        print(f"  Text:  {str(row['indication'])[:200]}...")

    df_clean.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")
    print(f"\nSaved: {OUTPUT_FILE}")
    print(f"Rows: {len(df_clean)} clean policies ready for vector store")
    print("\n" + "="*60)
    print("CLEANING COMPLETE")
    print("="*60)
    print("\nNext step: rebuild vector store with clean data")
    print("Run: python vector_store.py")

    return df_clean

if __name__ == "__main__":
    clean_lcd()