# Spam Email Detection Using Vector Space Model

This project implements **spam email detection** as a **Linear Algebra and Its Applications** mini-project using a **Vector Space Model (VSM)**.

## Problem Statement
Given an email text, classify it into:
- `spam` (unsolicited/malicious/promotional)
- `ham` (legitimate email)

## Linear Algebra Formulation
Each email is transformed into a vector in a high-dimensional term space.

1. Tokenize email text and build vocabulary of size `d`.
2. Build TF-IDF document vectors, giving matrix:

   `X ∈ R^(n x d)` where:
   - `n` = number of emails
   - `d` = vocabulary size

3. L2-normalize each row vector, so cosine similarity becomes dot product.
4. Compute class centroids from training vectors:

   - `c_spam = mean(X_spam)`
   - `c_ham  = mean(X_ham)`

5. Predict class of email vector `x` by cosine score:

   - `score_spam = x · c_spam`
   - `score_ham  = x · c_ham`
   - predict spam if `score_spam >= score_ham`, else ham.

This is a pure vector-space, linear-algebra-based classifier.

## Project Structure
- `data/email2.csv` - labeled dataset (spam/ham)
- `src/vsm_spam_detector.py` - full implementation (NumPy)
- `requirements.txt` - dependencies

## Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run
```bash
python3 src/vsm_spam_detector.py
```

Optional arguments:
```bash
python3 src/vsm_spam_detector.py --data data/email2.csv --test-ratio 0.25 --min-df 1 --seed 42
```

Predict on your own single input text (while still printing dataset metrics):
```bash
python3 src/vsm_spam_detector.py --text "Urgent account verification required. Click now."
```

Interactive mode for multiple custom inputs:
```bash
python3 src/vsm_spam_detector.py --interactive
```
In interactive mode, paste full mail content line by line, then type `/send` on a new line to classify.  
Type `/quit` to exit.

## Output
The script prints:
- dataset and split info
- confusion matrix `[[TN, FP], [FN, TP]]`
- Accuracy, Precision, Recall, F1-score
- most spam-indicative and ham-indicative terms
- Linear Algebra Report:
  - Gram-Schmidt orthogonalization on spam/ham centroid vectors
  - projection statistics on orthonormal basis vectors
  - covariance eigen-analysis (`C = X^T X`) with explained variance

## Notes for Project Report
You can include these topics:
1. Why TF-IDF reduces dominance of frequent but non-informative words.
2. Why cosine similarity is suitable for text vectors.
3. How centroids represent class prototypes in vector space.
4. Limitations: small dataset, vocabulary mismatch, no deep semantics.
5. Possible extension: compare with Naive Bayes / Logistic Regression.
