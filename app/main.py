# app/main.py
import argparse
import os
import sys

# pastikan src bisa diimpor
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.preprocess import preprocess_text
from src.boolean_ir import build_inverted_index, boolean_retrieve
from src.vsm_ir import build_vsm, build_bm25, score_bm25
from src.eval import precision, recall, f1_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd


def load_documents(data_path="data/processed"):
    docs = {}
    for fname in os.listdir(data_path):
        if fname.endswith(".txt"):
            with open(os.path.join(data_path, fname), encoding="utf-8") as f:
                docs[fname] = f.read()
    return docs


def run_boolean(query, docs):
    print("\n=== BOOLEAN RETRIEVAL ===")
    index = build_inverted_index(docs)
    result = boolean_retrieve(query, index)
    print(f"Query: {query}")
    print(f"Hasil dokumen: {result if result else 'Tidak ada dokumen yang cocok.'}")
    return result


def run_vsm(query, docs, k=5):
    print("\n=== VECTOR SPACE MODEL (TF-IDF) ===")
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(list(docs.values()))
    terms = vectorizer.get_feature_names_out()

    q_tokens = preprocess_text(query)
    q_text = ' '.join(q_tokens)
    q_vec = vectorizer.transform([q_text])
    sims = cosine_similarity(q_vec, tfidf_matrix)[0]

    sorted_idx = sims.argsort()[::-1][:k]
    print(f"\nQuery: {query}")
    print("Top-k dokumen:")
    for i in sorted_idx:
        docname = list(docs.keys())[i]
        print(f"{docname:15} | Skor: {sims[i]:.4f}")
    return sorted_idx


def run_bm25(query, docs, k=5):
    print("\n=== BM25 MODEL ===")
    index = build_bm25(docs)
    results = score_bm25(query, index, topk=k)
    print(f"Query: {query}")
    for doc, sc in results:
        print(f"{doc:15} | Skor: {sc:.4f}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Mini Search Engine STKI — Bintang Rifky Ananta")
    parser.add_argument("--model", choices=["boolean", "vsm", "bm25"], default="vsm", help="Model IR yang digunakan")
    parser.add_argument("--query", required=True, help="Kata kunci pencarian")
    parser.add_argument("--k", type=int, default=5, help="Jumlah hasil teratas")
    parser.add_argument("--data", default="data/processed", help="Folder berisi dokumen")
    args = parser.parse_args()

    docs = load_documents(args.data)
    if not docs:
        print("Folder data kosong atau belum di-preprocess!")
        return

    if args.model == "boolean":
        run_boolean(args.query, docs)
    elif args.model == "vsm":
        run_vsm(args.query, docs, args.k)
    elif args.model == "bm25":
        run_bm25(args.query, docs, args.k)
    else:
        print("Model tidak dikenali.")


if __name__ == "__main__":
    main()
