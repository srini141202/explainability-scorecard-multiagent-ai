import os
import pandas as pd
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
THESIS_DIR = os.path.dirname(BASE_DIR)
BASE = os.path.join(THESIS_DIR, "Datasets")
LCD_PATH = os.path.join(BASE, "Policy_CMS_LCD", "current_lcd_csv")
VECTORSTORE_PATH = os.path.join(BASE_DIR, "vectorstore_db")

def build_vector_store():
    print("Loading LCD policies...")
    df = pd.read_csv(os.path.join(LCD_PATH, "lcd_cleaned.csv"), encoding="utf-8", on_bad_lines="skip")
    print(f"Policies loaded: {len(df)} rows")

    print("Converting policies to documents...")
    documents = []
    for _, row in df.iterrows():
        title = str(row.get("title", ""))
        indication = str(row.get("indication", ""))
        lcd_id = str(row.get("lcd_id", ""))
        if indication and indication != "nan":
            content = f"Policy: {title}\nCoverage Rules: {indication}"
            doc = Document(
                page_content=content,
                metadata={"lcd_id": lcd_id, "title": title}
            )
            documents.append(doc)

    print(f"Documents prepared: {len(documents)}")
    print("Building vector store - please wait 2-3 minutes...")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=VECTORSTORE_PATH
    )

    print("Vector store built successfully!")
    return vectorstore

def load_vector_store():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectorstore = Chroma(
        persist_directory=VECTORSTORE_PATH,
        embedding_function=embeddings
    )
    return vectorstore

if __name__ == "__main__":
    print("Starting vector store build...")
    vs = build_vector_store()
    print("\nTesting search...")
    results = vs.similarity_search("mammography screening coverage women over 40", k=3)
    for i, r in enumerate(results):
        print(f"\nResult {i+1}: {r.metadata['title']}")
        print(r.page_content[:200])