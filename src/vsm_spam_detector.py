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


@dataclass
class VSMModel:
    vocab: Dict[str, int]
    idf: np.ndarray
    spam_centroid: np.ndarray
    ham_centroid: np.ndarray


@dataclass
class GramSchmidtReport:
    q1: np.ndarray
    q2: np.ndarray
    coeff_ham_on_q1: float
    residual_norm: float
    centroid_cosine: float


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


def stable_softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    exps = np.exp(shifted)
    return exps / np.sum(exps)


def gram_schmidt_two_vectors(v1: np.ndarray, v2: np.ndarray, eps: float = 1e-12) -> GramSchmidtReport:
    n1 = np.linalg.norm(v1)
    if n1 <= eps:
        raise ValueError("Cannot run Gram-Schmidt with zero first vector.")
    q1 = v1 / n1

    coeff = float(v2 @ q1)
    u2 = v2 - coeff * q1
    residual_norm = float(np.linalg.norm(u2))
    q2 = np.zeros_like(v2)
    if residual_norm > eps:
        q2 = u2 / residual_norm

    centroid_cosine = float(v1 @ v2 / max(np.linalg.norm(v1) * np.linalg.norm(v2), eps))
    return GramSchmidtReport(
        q1=q1,
        q2=q2,
        coeff_ham_on_q1=coeff,
        residual_norm=residual_norm,
        centroid_cosine=centroid_cosine,
    )


def covariance_eigen_analysis(x: np.ndarray, top_k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    cov = x.T @ x
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = eigvals[::-1]
    eigvals[eigvals < 0.0] = 0.0

    total = float(np.sum(eigvals))
    if total <= 1e-12:
        ratios = np.zeros_like(eigvals)
    else:
        ratios = eigvals / total
    k = min(top_k, len(eigvals))
    return eigvals[:k], ratios[:k]


def predict_single_text(model: VSMModel, text: str, temperature: float = 5.0) -> Tuple[str, float, float, float]:
    tokens = [tokenize(text)]
    x = compute_tf_idf(tokens, model.vocab, model.idf)

    spam_score = float((x @ model.spam_centroid)[0])
    ham_score = float((x @ model.ham_centroid)[0])
    probs = stable_softmax(np.array([ham_score * temperature, spam_score * temperature], dtype=np.float64))
    p_ham, p_spam = float(probs[0]), float(probs[1])

    if p_spam >= p_ham:
        return "spam", p_spam, spam_score, ham_score
    return "ham", p_ham, spam_score, ham_score


def most_informative_terms(vocab: Dict[str, int], spam_centroid: np.ndarray, ham_centroid: np.ndarray, k: int = 10):
    inv_vocab = {i: term for term, i in vocab.items()}
    diff = spam_centroid - ham_centroid
    top_spam_idx = np.argsort(-diff)[:k]
    top_ham_idx = np.argsort(diff)[:k]

    top_spam = [(inv_vocab[i], float(diff[i])) for i in top_spam_idx]
    top_ham = [(inv_vocab[i], float(-diff[i])) for i in top_ham_idx]
    return top_spam, top_ham


def run_experiment(
    data_path: str,
    test_ratio: float,
    min_df: int,
    seed: int,
    input_text: str | None = None,
    interactive: bool = False,
) -> None:
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
    model = VSMModel(vocab=vocab, idf=idf, spam_centroid=spam_centroid, ham_centroid=ham_centroid)

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

    gs = gram_schmidt_two_vectors(spam_centroid, ham_centroid)
    test_proj_q1 = x_test @ gs.q1
    test_proj_q2 = x_test @ gs.q2
    spam_mask = y_test == 1
    ham_mask = y_test == 0
    spam_mean_q1 = float(np.mean(test_proj_q1[spam_mask])) if np.any(spam_mask) else 0.0
    spam_mean_q2 = float(np.mean(test_proj_q2[spam_mask])) if np.any(spam_mask) else 0.0
    ham_mean_q1 = float(np.mean(test_proj_q1[ham_mask])) if np.any(ham_mask) else 0.0
    ham_mean_q2 = float(np.mean(test_proj_q2[ham_mask])) if np.any(ham_mask) else 0.0

    top_eigs, top_ratios = covariance_eigen_analysis(x_train, top_k=5)
    cumulative = np.cumsum(top_ratios)

    print()
    print("Linear Algebra Report:")
    print("  Gram-Schmidt on class centroids:")
    print(f"    cosine(c_spam, c_ham): {gs.centroid_cosine:.4f}")
    print(f"    ham projection on q1  : {gs.coeff_ham_on_q1:.4f}")
    print(f"    residual norm for q2  : {gs.residual_norm:.4f}")
    print("    Mean test projections (q1, q2):")
    print(f"      spam: ({spam_mean_q1:.4f}, {spam_mean_q2:.4f})")
    print(f"      ham : ({ham_mean_q1:.4f}, {ham_mean_q2:.4f})")
    print("  Covariance eigen-analysis (C = X^T X on train TF-IDF):")
    for i, (eig, ratio, cum) in enumerate(zip(top_eigs, top_ratios, cumulative), start=1):
        print(f"    lambda_{i}: {eig:.4f} | explained={ratio:.4f} | cumulative={cum:.4f}")

    if input_text:
        label, confidence, spam_score, ham_score = predict_single_text(model, input_text)
        print()
        print("Custom Input Prediction:")
        print(f"Text      : {input_text}")
        print(f"Prediction: {label}")
        print(f"Confidence: {confidence:.4f}")
        print(f"Spam score: {spam_score:.4f}")
        print(f"Ham score : {ham_score:.4f}")

    if interactive:
        print()
        print("Interactive mode (multiline).")
        print("Enter/paste email lines, then type /send on a new line to classify.")
        print("Type /quit to exit.")
        buffer: List[str] = []
        while True:
            try:
                prompt = "> " if not buffer else "... "
                line = input(prompt)
            except EOFError:
                break
            text = line.strip()
            if text == "/quit":
                break
            if text == "/send":
                full_text = "\n".join(buffer).strip()
                if not full_text:
                    print("No text provided. Paste email content first, then /send.")
                    continue
                label, confidence, spam_score, ham_score = predict_single_text(model, full_text)
                print(
                    f"prediction={label} confidence={confidence:.4f} "
                    f"(spam_score={spam_score:.4f}, ham_score={ham_score:.4f})"
                )
                buffer = []
                continue
            buffer.append(line)

        if buffer:
            full_text = "\n".join(buffer).strip()
            if full_text:
                label, confidence, spam_score, ham_score = predict_single_text(model, full_text)
                print(
                    f"prediction={label} confidence={confidence:.4f} "
                    f"(spam_score={spam_score:.4f}, ham_score={ham_score:.4f})"
                )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Spam Email Detection using Vector Space Model (TF-IDF + Cosine Similarity)"
    )
    parser.add_argument("--data", default="data/email2.csv", help="Path to CSV dataset")
    parser.add_argument("--test-ratio", type=float, default=0.25, help="Test split ratio (0,1)")
    parser.add_argument("--min-df", type=int, default=1, help="Minimum document frequency")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--text", default=None, help="Single custom email text for prediction")
    parser.add_argument("--interactive", action="store_true", help="Interactive custom text prediction mode")
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
        input_text=args.text,
        interactive=args.interactive,
    )


if __name__ == "__main__":
    main()
