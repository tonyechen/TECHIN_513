"""
models/baselines.py
-------------------
Baseline classifiers for EDM section labeling.

Both models are frame-level (no temporal context) and are compared against
the CNN+BiLSTM in Step 14 to quantify the value of sequential modeling.

Public API
----------
    train_logistic_regression(X_train, y_train) → fitted Pipeline
    train_random_forest(X_train, y_train)       → fitted RandomForestClassifier
    evaluate_baseline(clf, X_eval, y_eval, name, label_names) → dict of scores
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, f1_score, accuracy_score

LABEL_NAMES = ['calm', 'breakdown', 'buildup', 'drop']
MAX_SUBSAMPLE = 100_000   # cap training frames for speed


def _subsample(X, y, max_n=MAX_SUBSAMPLE, seed=42):
    """Randomly subsample training data if it exceeds max_n frames."""
    if len(X) <= max_n:
        return X, y
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), max_n, replace=False)
    print(f'  (Subsampled to {max_n:,} frames for speed)')
    return X[idx], y[idx]


# ── Logistic Regression ───────────────────────────────────────────────────────

def train_logistic_regression(X_train, y_train, C=1.0):
    """
    Frame-level linear classifier on raw features.

    Includes its own StandardScaler so it is self-contained and does not
    interfere with the CNN+BiLSTM scaler.

    Parameters
    ----------
    X_train : np.ndarray  (N, n_features)
    y_train : np.ndarray  (N,)  integer labels
    C       : float  inverse regularisation strength (default 1.0)

    Returns
    -------
    clf : fitted sklearn Pipeline (StandardScaler + LogisticRegression)
    """
    X_sub, y_sub = _subsample(X_train, y_train)

    clf = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(
            max_iter=1000,
            class_weight='balanced',   # same imbalance handling as CNN+BiLSTM
            solver='lbfgs',            # lbfgs handles multinomial natively (multi_class removed in sklearn 1.5)
            random_state=42,
            C=C,
        ))
    ])
    print('  Training Logistic Regression...')
    clf.fit(X_sub, y_sub)
    return clf


# ── Random Forest ─────────────────────────────────────────────────────────────

def train_random_forest(X_train, y_train, n_estimators=100, max_depth=20):
    """
    Frame-level non-linear classifier. Captures feature interactions that
    logistic regression misses, but still has no temporal context.

    Parameters
    ----------
    X_train      : np.ndarray  (N, n_features)
    y_train      : np.ndarray  (N,)  integer labels
    n_estimators : int  number of trees
    max_depth    : int  maximum tree depth

    Returns
    -------
    clf : fitted RandomForestClassifier
    """
    X_sub, y_sub = _subsample(X_train, y_train)

    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1,
    )
    print(f'  Training Random Forest ({n_estimators} trees)...')
    clf.fit(X_sub, y_sub)
    return clf


# ── Shared evaluation helper ──────────────────────────────────────────────────

def evaluate_baseline(clf, X_eval, y_eval, name, label_names=LABEL_NAMES):
    """
    Evaluate a fitted classifier and print a classification report.

    Parameters
    ----------
    clf         : fitted classifier with a .predict() method
    X_eval      : np.ndarray  (N, n_features)
    y_eval      : np.ndarray  (N,)  integer ground-truth labels
    name        : str  display name for the model
    label_names : list of str  class names in label-integer order

    Returns
    -------
    scores : dict  with keys 'f1', 'acc', 'y_pred'
    """
    print('=' * 55)
    print(f'{name}')
    print('=' * 55)

    y_pred = clf.predict(X_eval)
    acc    = accuracy_score(y_eval, y_pred)
    f1     = f1_score(y_eval, y_pred, average='weighted', zero_division=0)

    print(f'Accuracy     : {acc:.3f}')
    print(f'Weighted F1  : {f1:.3f}')
    print()
    print(classification_report(y_eval, y_pred,
          target_names=label_names, zero_division=0))

    return {'f1': f1, 'acc': acc, 'y_pred': y_pred}
