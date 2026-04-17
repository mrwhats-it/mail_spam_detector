#!/usr/bin/env python3
"""Spam Email Detection using a Vector Space Model (TF-IDF + Cosine Similarity)."""

from __future__ import annotations

import argparse
import csv
import random
import re
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np

TOKEN_PATTERN = re.compile(r"[a-zA-Z]+")


@dataclass
class Dataset:
    texts: List[str]
    labels: np.ndarray


@dataclass
class Metrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    confusion_matrix: np.ndarray


def tokenize(text: str) -> List[str]:
    return TOKEN_PATTERN.findall(text.lower())


def load_dataset(path: str) -> Dataset:
    texts: List[str] = []
    labels: List[int] = []

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            label = row["label"].strip().lower()
            text = row["text"].strip()
            if label not in {"spam", "ham"}:
                raise ValueError(f"Invalid label '{label}'. Expected 'spam' or 'ham'.")
            texts.append(text)
            labels.append(1 if label == "spam" else 0)

    if not texts:
        raise ValueError("Dataset is empty.")

    return Dataset(texts=texts, labels=np.array(labels, dtype=np.int32))


def train_test_split(
    texts: Sequence[str],
    labels: np.ndarray,
    test_ratio: float = 0.25,
    seed: int = 42,
) -> Tuple[List[str], np.ndarray, List[str], np.ndarray]:
    if len(texts) != len(labels):
        raise ValueError("texts and labels size mismatch.")

    idx = list(range(len(texts)))
    rng = random.Random(seed)
    rng.shuffle(idx)

    split = int(len(idx) * (1 - test_ratio))
    train_idx = idx[:split]
    test_idx = idx[split:]

    x_train = [texts[i] for i in train_idx]
    y_train = labels[train_idx]
    x_test = [texts[i] for i in test_idx]
    y_test = labels[test_idx]
    return x_train, y_train, x_test, y_test


def build_vocabulary(tokenized_docs: Sequence[List[str]], min_df: int = 1) -> Dict[str, int]:
    doc_freq: Counter[str] = Counter()
    for doc in tokenized_docs:
        doc_freq.update(set(doc))

    terms = [term for term, df in doc_freq.items() if df >= min_df]
    terms.sort()
    return {term: i for i, term in enumerate(terms)}


def compute_idf(tokenized_docs: Sequence[List[str]], vocab: Dict[str, int]) -> np.ndarray:
    n_docs = len(tokenized_docs)
    df = np.zeros(len(vocab), dtype=np.float64)

    for doc in tokenized_docs:
        seen = set()
        for token in doc:
            idx = vocab.get(token)
            if idx is not None and idx not in seen:
                df[idx] += 1.0
                seen.add(idx)

    # Smoothed IDF: log((1 + N)/(1 + df)) + 1
    return np.log((1.0 + n_docs) / (1.0 + df)) + 1.0


def compute_tf_idf(
    tokenized_docs: Sequence[List[str]],
    vocab: Dict[str, int],
    idf: np.ndarray,
) -> np.ndarray:
    x = np.zeros((len(tokenized_docs), len(vocab)), dtype=np.float64)

    for i, doc in enumerate(tokenized_docs):
        if not doc:
            continue
        counts = Counter(token for token in doc if token in vocab)
        if not counts:
            continue

        max_tf = max(counts.values())
        for token, count in counts.items():
            j = vocab[token]
            tf = count / max_tf
            x[i, j] = tf * idf[j]

    return l2_normalize_rows(x)


def l2_normalize_rows(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return mat / norms


def class_centroids(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    spam_vecs = x[y == 1]
    ham_vecs = x[y == 0]
    if len(spam_vecs) == 0 or len(ham_vecs) == 0:
        raise ValueError("Training split must contain both spam and ham classes.")

    spam_centroid = spam_vecs.mean(axis=0)
    ham_centroid = ham_vecs.mean(axis=0)

    spam_norm = np.linalg.norm(spam_centroid)
    ham_norm = np.linalg.norm(ham_centroid)
    if spam_norm > 0:
        spam_centroid = spam_centroid / spam_norm
    if ham_norm > 0:
        ham_centroid = ham_centroid / ham_norm

    return spam_centroid, ham_centroid


def predict(x: np.ndarray, spam_centroid: np.ndarray, ham_centroid: np.ndarray) -> np.ndarray:
    spam_scores = x @ spam_centroid
    ham_scores = x @ ham_centroid
    return (spam_scores >= ham_scores).astype(np.int32)


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> Metrics:
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    accuracy = (tp + tn) / max(len(y_true), 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)

    cm = np.array([[tn, fp], [fn, tp]], dtype=np.int32)
    return Metrics(accuracy=accuracy, precision=precision, recall=recall, f1=f1, confusion_matrix=cm)


def most_informative_terms(vocab: Dict[str, int], spam_centroid: np.ndarray, ham_centroid: np.ndarray, k: int = 10):
    inv_vocab = {i: term for term, i in vocab.items()}
    diff = spam_centroid - ham_centroid
    top_spam_idx = np.argsort(-diff)[:k]
    top_ham_idx = np.argsort(diff)[:k]

    top_spam = [(inv_vocab[i], float(diff[i])) for i in top_spam_idx]
    top_ham = [(inv_vocab[i], float(-diff[i])) for i in top_ham_idx]
    return top_spam, top_ham


def run_experiment(data_path: str, test_ratio: float, min_df: int, seed: int) -> None:
    dataset = load_dataset(data_path)
    x_train_raw, y_train, x_test_raw, y_test = train_test_split(
        dataset.texts,
        dataset.labels,
        test_ratio=test_ratio,
        seed=seed,
    )

    train_tokens = [tokenize(t) for t in x_train_raw]
    test_tokens = [tokenize(t) for t in x_test_raw]

    vocab = build_vocabulary(train_tokens, min_df=min_df)
    if not vocab:
        raise ValueError("Vocabulary is empty; reduce min_df.")

    idf = compute_idf(train_tokens, vocab)
    x_train = compute_tf_idf(train_tokens, vocab, idf)
    x_test = compute_tf_idf(test_tokens, vocab, idf)

    spam_centroid, ham_centroid = class_centroids(x_train, y_train)
    y_pred = predict(x_test, spam_centroid, ham_centroid)
    metrics = evaluate(y_test, y_pred)

    top_spam, top_ham = most_informative_terms(vocab, spam_centroid, ham_centroid, k=8)

    print("=== Spam Detection with Vector Space Model ===")
    print(f"Dataset path: {data_path}")
    print(f"Emails total: {len(dataset.texts)}")
    print(f"Train/Test split: {len(y_train)}/{len(y_test)}")
    print(f"Vocabulary size: {len(vocab)}")
    print()
    print("Confusion Matrix [[TN, FP], [FN, TP]]:")
    print(metrics.confusion_matrix)
    print()
    print(f"Accuracy : {metrics.accuracy:.4f}")
    print(f"Precision: {metrics.precision:.4f}")
    print(f"Recall   : {metrics.recall:.4f}")
    print(f"F1-Score : {metrics.f1:.4f}")
    print()
    print("Top spam-indicative terms:")
    for term, score in top_spam:
        print(f"  {term:<16} {score:.4f}")
    print("Top ham-indicative terms:")
    for term, score in top_ham:
        print(f"  {term:<16} {score:.4f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Spam Email Detection using Vector Space Model (TF-IDF + Cosine Similarity)"
    )
    parser.add_argument("--data", default="data/emails.csv", help="Path to CSV dataset")
    parser.add_argument("--test-ratio", type=float, default=0.25, help="Test split ratio (0,1)")
    parser.add_argument("--min-df", type=int, default=1, help="Minimum document frequency")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not (0.0 < args.test_ratio < 1.0):
        raise ValueError("--test-ratio must be in (0, 1).")
    if args.min_df < 1:
        raise ValueError("--min-df must be >= 1")

    run_experiment(
        data_path=args.data,
        test_ratio=args.test_ratio,
        min_df=args.min_df,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
