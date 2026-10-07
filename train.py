"""Train and evaluate every RallyIQ model, then save what the Streamlit app needs.

Usage:
    git clone https://github.com/wywyWang/CoachAI-Projects.git
    python train.py            # uses a GPU automatically if one is available
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from rallyiq.features import OUTCOMES, add_features, build_X, load_strokes
from rallyiq.sequence_model import make_sequences, predict, train_transformer

SEED = 42


def scores(name, y_true, probs, train_prior):
    guess = np.tile(train_prior, (len(y_true), 1))
    return {
        "model": name,
        "log_loss": round(log_loss(y_true, probs, labels=[0, 1, 2]), 4),
        "log_loss_guessing": round(log_loss(y_true, guess, labels=[0, 1, 2]), 4),
        "winner_auc": round(roc_auc_score(y_true == 1, probs[:, 1]), 4),
        "error_auc": round(roc_auc_score(y_true == 2, probs[:, 2]), 4),
    }


def rally_winner_diagnostic(s, X, train_idx, test_idx):
    """The first framing: predict who wins the rally from each shot. Shows why we reframed."""
    y = s["hitter_wins"].values
    model = XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05, subsample=0.8,
                          colsample_bytree=0.8, random_state=SEED)
    model.fit(X.iloc[train_idx], y[train_idx])
    pred = model.predict(X.iloc[test_idx])
    correct = pred == y[test_idx]
    left = s["shots_left"].values[test_idx]
    return {
        "overall_accuracy": round(float(correct.mean()), 3),
        "final_shot_accuracy": round(float(correct[left == 0].mean()), 3),
        "six_plus_shots_from_end_accuracy": round(float(correct[left >= 6].mean()), 3),
    }


def main(coachai_dir, out_dir, epochs):
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs("assets", exist_ok=True)

    strokes, matches = load_strokes(coachai_dir)
    s = add_features(strokes)
    X_raw = build_X(s)
    y = s["outcome"].values
    print(f"{s['match_key'].nunique()} matches, {s['rally_id'].nunique()} rallies, {len(s)} shots")

    # split by MATCH so the test set is matches the models have never seen
    split = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    train_idx, test_idx = next(split.split(X_raw, y, s["match_key"]))
    inner = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=SEED)
    fit_rel, val_rel = next(inner.split(train_idx, groups=s["match_key"].values[train_idx]))
    fit_idx, val_idx = train_idx[fit_rel], train_idx[val_rel]

    medians = X_raw.iloc[train_idx].median()
    X = X_raw.fillna(medians)
    prior = np.bincount(y[train_idx], minlength=3) / len(train_idx)
    y_test = y[test_idx]
    results = []

    # 1. logistic regression baseline
    logreg = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    logreg.fit(X.iloc[train_idx], y[train_idx])
    results.append(scores("logistic regression", y_test, logreg.predict_proba(X.iloc[test_idx]), prior))

    # 2. XGBoost
    xgb = XGBClassifier(n_estimators=400, max_depth=5, learning_rate=0.05, subsample=0.8,
                        colsample_bytree=0.8, random_state=SEED)
    xgb.fit(X.iloc[train_idx], y[train_idx])
    xgb_probs_all = xgb.predict_proba(X)
    results.append(scores("xgboost", y_test, xgb_probs_all[test_idx], prior))

    # 3. causal Transformer over the whole rally
    mean = X.iloc[train_idx].mean()
    std = X.iloc[train_idx].std().replace(0, 1)
    X_scaled = ((X - mean) / std).values.astype(np.float32)
    seqs, labels, rows = make_sequences(X_scaled, y, s["rally_id"].values)
    rally_set = {"fit": set(fit_idx), "val": set(val_idx), "test": set(test_idx)}
    pick = lambda part: [i for i, r in enumerate(rows) if r[0] in rally_set[part]]
    fit_r, val_r, test_r = pick("fit"), pick("val"), pick("test")

    model, device = train_transformer([seqs[i] for i in fit_r], [labels[i] for i in fit_r],
                                      [seqs[i] for i in val_r], [labels[i] for i in val_r],
                                      n_features=X.shape[1], epochs=epochs)
    print("transformer trained on", device)
    tf_probs_all = np.zeros((len(s), 3))
    for r_i, p in zip(test_r, predict(model, [seqs[i] for i in test_r], device)):
        tf_probs_all[rows[r_i]] = p
    results.append(scores("transformer", y_test, tf_probs_all[test_idx], prior))

    # 4. ensemble: average the tree model and the sequence model
    ens_probs_all = (xgb_probs_all + tf_probs_all) / 2
    results.append(scores("ensemble (xgboost + transformer)", y_test, ens_probs_all[test_idx], prior))

    for r in results:
        print(r)
    diagnostic = rally_winner_diagnostic(s, X, train_idx, test_idx)
    print("rally-winner framing:", diagnostic)

    # SHAP explanation chart for the XGBoost model
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import shap
        sample = X.iloc[test_idx].sample(2000, random_state=0)
        sv = shap.TreeExplainer(xgb)(sample)
        shap.plots.beeswarm(sv[:, :, 1], max_display=12, show=False)
        plt.title("What makes a shot a winner")
        plt.tight_layout()
        plt.savefig("assets/shap_winner.png", dpi=150)
        plt.close()
    except Exception as e:  # SHAP is optional
        print("skipped SHAP:", e)

    # which shots pros actually play from each court zone (keeps recommendations realistic)
    usage = pd.crosstab(s.iloc[train_idx]["hit_area"], s.iloc[train_idx]["shot"], normalize="index")

    # save everything the app needs
    xgb.save_model(os.path.join(out_dir, "xgb_model.json"))
    torch.save(model.state_dict(), os.path.join(out_dir, "transformer.pt"))
    usage.to_csv(os.path.join(out_dir, "shot_usage.csv"))
    with open(os.path.join(out_dir, "feature_columns.json"), "w") as f:
        json.dump(list(X.columns), f)
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump({
            "matches": int(s["match_key"].nunique()), "rallies": int(s["rally_id"].nunique()),
            "shots": int(len(s)), "test_matches": int(s["match_key"].iloc[test_idx].nunique()),
            "test_shots": int(len(test_idx)), "outcome_share": dict(zip(OUTCOMES, prior.round(4).tolist())),
            "models": results, "rally_winner_framing": diagnostic,
        }, f, indent=2)

    info = ["match_key", "set", "rally", "rally_id", "ball_round", "player", "shot", "hit_area",
            "hit_x", "hit_y", "landing_x", "landing_y", "player_location_x", "player_location_y",
            "opponent_location_x", "opponent_location_y", "start_A", "start_B", "outcome"]
    app = s[info].copy()
    for k, name in enumerate(OUTCOMES):
        app[f"xgb_{name}"] = xgb_probs_all[:, k].round(4)
        app[f"tf_{name}"] = tf_probs_all[:, k].round(4)
        app[f"ens_{name}"] = ens_probs_all[:, k].round(4)
    app = pd.concat([app, X.add_prefix("f__")], axis=1).iloc[test_idx]
    app.to_csv(os.path.join(out_dir, "test_shots.csv.gz"), index=False)
    matches[matches["match_key"].isin(app["match_key"])].to_csv(os.path.join(out_dir, "matches.csv"), index=False)
    print("saved to", out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="CoachAI-Projects")
    parser.add_argument("--out", default="data")
    parser.add_argument("--epochs", type=int, default=25)
    a = parser.parse_args()
    main(a.data, a.out, a.epochs)
