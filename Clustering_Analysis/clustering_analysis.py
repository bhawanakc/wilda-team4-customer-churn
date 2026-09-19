"""
Customer Churn Analysis - Clustering Pipeline
Telecommunications Company Dataset

Steps covered:
  1. Load the prepared train/test sets and split the target off
  2. Search k = 1 to 10 (inertia, % drop, silhouette)
  3. Save the cluster selection metrics
  4. Train the final K-Means model at k = 4 and assign every customer
  5. Profile each cluster on the training set and name it from its profile
  6. Save the profiles, the assignments and the summary
  7. Draw and save the cluster visualisations
  8. Print a final summary of every segment

The model is fitted on the TRAINING set only. The test rows are assigned with
model.predict, so they stay unseen, the same way the scaler was fitted.

Run from the repository root:
    python Clustering_Analysis/clustering_analysis.py
"""

import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# ---------------------------------------------------------------------------
# CONFIG - every path is resolved relative to this script, so the pipeline
# runs from anywhere with no edits.
# ---------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent          # Clustering_Analysis/
ROOT = HERE.parent                              # repository root
PREP = ROOT / "Data_Preparation"                # Rohan's prepared data
OUT_DIR = HERE
PLOTS = HERE / "cluster_visualisations"

TRAIN_PATH = PREP / "train_set.csv"
TEST_PATH = PREP / "test_set.csv"
SCALER_PATH = PREP / "scaler.pkl"

TARGET_COL = "Churn"
RANDOM_STATE = 42                               # fixed seed, reproducible runs
N_INIT = 10                                     # 10 restarts, best inertia kept
K_RANGE = range(1, 11)                          # every k the elbow chart shows
SCALE_COLS = ["tenure", "MonthlyCharges"]       # the two columns the scaler holds

# Chosen by the elbow method. Inertia falls 25.2%, 18.0% and 12.1% for k = 2, 3
# and 4, then only 5.1% for k = 5. See elbow_method_results.pdf, section 5.
OPTIMAL_K = 4

# The eight 0/1 features, with the column name each profile column reports.
BINARY_FEATURES = {
    "gender": "gender_male",
    "SeniorCitizen": "senior_citizen",
    "Dependents": "dependents",
    "PhoneService": "phone_service",
    "MultipleLines": "multiple_lines",
    "InternetService": "fiber_optic",
    "Contract_One_year": "contract_one_year",
    "Contract_Two_year": "contract_two_year",
}

PROFILE_COLS = (
    ["cluster", "segment", "size", "share_pct", "churners", "churn_rate",
     "test_size", "test_churn_rate", "tenure_months", "monthly_charges"]
    + list(BINARY_FEATURES.values())
)

# One fixed colour per segment name, so a segment keeps its colour on every
# chart even if the cluster ID numbers move. Okabe-Ito, colour-blind safe.
SEGMENT_COLOURS = {
    "New high spenders": "#d55e00",
    "New low spenders": "#e69f00",
    "Long-term high spenders": "#0072b2",
    "Long-term low spenders": "#009e73",
}


def load_prepared_data():
    """Step 1: Load the prepared sets and split the target off the features."""
    for path in (TRAIN_PATH, TEST_PATH, SCALER_PATH):
        if not path.exists():
            raise FileNotFoundError(
                f"\n\nCould not find: {path}\n"
                f"Fix: run 'python Data_Preparation/data_preparation.py' first."
            )

    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    scaler = joblib.load(SCALER_PATH)

    X_train = train_df.drop(columns=[TARGET_COL])
    y_train = train_df[TARGET_COL]
    X_test = test_df.drop(columns=[TARGET_COL])
    y_test = test_df[TARGET_COL]

    # The target must never reach the clustering features. Stage 3 predicts
    # Churn, so a cluster built from it would hand the answer to the model.
    assert TARGET_COL not in X_train.columns, "Churn must not be a clustering feature"
    assert TARGET_COL not in X_test.columns, "Churn must not be a clustering feature"
    assert X_train.shape[1] == 10, f"expected 10 features, found {X_train.shape[1]}"
    assert list(X_train.columns) == list(X_test.columns), "train/test columns differ"

    print(f"Training set: {len(X_train)} rows, churn rate {y_train.mean():.2%}")
    print(f"Testing set:  {len(X_test)} rows, churn rate {y_test.mean():.2%}")
    print(f"Clustering features ({X_train.shape[1]}): {list(X_train.columns)}")
    print(f"'{TARGET_COL}' held back, used only to score the finished segments.\n")
    return X_train, y_train, X_test, y_test, scaler


def search_k(X_train: pd.DataFrame) -> pd.DataFrame:
    """
    Step 2: Fit K-Means for every k and record how well it fits.

    Inertia is the within-cluster sum of squares: the total squared distance
    from each customer to its own cluster centre. It always falls as k rises,
    so the elbow is read from the SIZE of each fall, not from the value.
    The silhouette score is a supporting check, computed on all 5,634 rows.
    """
    print(f"--- Searching k = {K_RANGE.start} to {K_RANGE.stop - 1} ---")
    rows, previous_inertia = [], None

    for k in K_RANGE:
        model = KMeans(n_clusters=k, n_init=N_INIT, random_state=RANDOM_STATE)
        labels = model.fit_predict(X_train)

        drop_pct = (None if previous_inertia is None
                    else (previous_inertia - model.inertia_) / previous_inertia * 100)
        sil = None if k == 1 else float(silhouette_score(X_train, labels))

        rows.append({
            "k": k,
            "inertia": round(float(model.inertia_), 4),
            "inertia_drop_pct": None if drop_pct is None else round(drop_pct, 2),
            "silhouette": None if sil is None else round(sil, 4),
        })
        print(f"  k={k:>2}  inertia {model.inertia_:>11,.2f}"
              f"  drop {'    n/a' if drop_pct is None else f'{drop_pct:6.2f}%'}"
              f"  silhouette {'   n/a' if sil is None else f'{sil:.4f}'}")
        previous_inertia = model.inertia_

    metrics = pd.DataFrame(rows)
    print(f"Largest drops: k=2 {metrics.loc[1, 'inertia_drop_pct']}%, "
          f"k=3 {metrics.loc[2, 'inertia_drop_pct']}%, "
          f"k=4 {metrics.loc[3, 'inertia_drop_pct']}%, "
          f"then k=5 {metrics.loc[4, 'inertia_drop_pct']}%.\n")
    return metrics


def train_final_model(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """Step 4: Train the model at the chosen k, then assign both sets."""
    print(f"--- Training the final model (k = {OPTIMAL_K}) ---")
    model = KMeans(n_clusters=OPTIMAL_K, n_init=N_INIT, random_state=RANDOM_STATE)
    model.fit(X_train)

    train_labels = model.labels_                 # the training rows the model saw
    test_labels = model.predict(X_test)          # unseen rows, nearest centre

    final_silhouette = float(silhouette_score(X_train, train_labels))
    print(f"Settings: n_clusters={OPTIMAL_K}, n_init={N_INIT}, "
          f"random_state={RANDOM_STATE}")
    print(f"Inertia: {model.inertia_:,.2f}   Silhouette: {final_silhouette:.4f}")
    print(f"Training rows assigned: {len(train_labels)}")
    print(f"Test rows assigned with model.predict: {len(test_labels)}\n")
    return model, train_labels, test_labels, final_silhouette


def original_units(scaled: pd.DataFrame, scaler) -> pd.DataFrame:
    """Turn the two scaled columns back into months and dollars."""
    values = scaler.inverse_transform(scaled[SCALE_COLS])
    return pd.DataFrame(values, columns=["tenure_months", "monthly_charges"],
                        index=scaled.index)


def name_segments(profile: pd.DataFrame, scaler) -> dict:
    """
    Step 5b: Name every cluster from its own profile, in code.

    Each centre's tenure and monthly charge are compared with the training
    means. Below the mean tenure is "New", above it is "Long-term". Below the
    mean charge is "low spenders", above it is "high spenders". Cluster ID
    numbers change between runs, so every document refers to the names.
    """
    mean_tenure, mean_charge = float(scaler.mean_[0]), float(scaler.mean_[1])
    print("--- Naming the segments ---")
    print(f"Training means used: tenure {mean_tenure:.4f} months, "
          f"monthly charge ${mean_charge:.4f}")

    names = {}
    for row in profile.itertuples():
        age = "New" if row.tenure_months < mean_tenure else "Long-term"
        spend = "low spenders" if row.monthly_charges < mean_charge else "high spenders"
        names[row.cluster] = f"{age} {spend}"
        print(f"  cluster {row.cluster}: {row.tenure_months:.1f} months, "
              f"${row.monthly_charges:.1f} -> {names[row.cluster]}")

    if len(set(names.values())) != len(names):
        raise ValueError(
            f"Two clusters were given the same name: {names}. "
            f"The naming rule needs another dimension before the docs can be built."
        )
    print(f"All {len(names)} names are distinct.\n")
    return names


def build_profiles(X_train, y_train, train_labels, X_test, y_test, test_labels,
                   scaler) -> tuple[pd.DataFrame, dict]:
    """Step 5: Build one profile row per cluster, from the training set."""
    print("--- Building the cluster profiles ---")
    train_units = original_units(X_train, scaler)

    rows = []
    for cluster in range(OPTIMAL_K):
        in_train = train_labels == cluster
        in_test = test_labels == cluster
        row = {
            "cluster": int(cluster),
            "size": int(in_train.sum()),
            "share_pct": round(float(in_train.mean() * 100), 2),
            "churners": int(y_train[in_train].sum()),
            "churn_rate": round(float(y_train[in_train].mean()), 4),
            "test_size": int(in_test.sum()),
            "test_churn_rate": round(float(y_test[in_test].mean()), 4),
            "tenure_months": round(float(train_units.loc[in_train, "tenure_months"].mean()), 1),
            "monthly_charges": round(float(train_units.loc[in_train, "monthly_charges"].mean()), 1),
        }
        for source, name in BINARY_FEATURES.items():
            row[name] = round(float(X_train.loc[in_train, source].mean()), 4)
        rows.append(row)

    profile = pd.DataFrame(rows)
    names = name_segments(profile, scaler)
    profile["segment"] = profile["cluster"].map(names)

    # Highest churn rate first, so the first row is the first retention target.
    profile = profile.sort_values("churn_rate", ascending=False).reset_index(drop=True)
    profile = profile[PROFILE_COLS]
    print(profile[["cluster", "segment", "size", "churn_rate",
                   "tenure_months", "monthly_charges"]].to_string(index=False))
    print()
    return profile, names


def build_assignments(train_labels, test_labels, names) -> pd.DataFrame:
    """Step 6a: One row per customer, for the Stage 3 model to join on."""
    frames = []
    for split, labels in (("train", train_labels), ("test", test_labels)):
        frames.append(pd.DataFrame({
            "split": split,
            "row": np.arange(len(labels)),
            "cluster": labels.astype(int),
            "segment": [names[int(c)] for c in labels],
        }))
    assignments = pd.concat(frames, ignore_index=True)
    print(f"Assignments built for {len(assignments):,} customers "
          f"({int((assignments.split == 'train').sum()):,} train, "
          f"{int((assignments.split == 'test').sum()):,} test).")
    return assignments


def save_outputs(model, metrics, profile, assignments, final_silhouette,
                 scaler, y_train, y_test, feature_names) -> None:
    """Steps 3 and 6: write every file the documents and Stage 3 depend on."""
    print("--- Saving Outputs ---")

    joblib.dump(model, OUT_DIR / "kmeans_model.pkl")
    print(f"Trained model saved to: {(OUT_DIR / 'kmeans_model.pkl').relative_to(ROOT)}")

    metrics.to_csv(OUT_DIR / "cluster_selection_metrics.csv", index=False)
    profile.to_csv(OUT_DIR / "cluster_profiles.csv", index=False)
    assignments.to_csv(OUT_DIR / "cluster_assignments.csv", index=False)
    for name in ("cluster_selection_metrics.csv", "cluster_profiles.csv",
                 "cluster_assignments.csv"):
        print(f"Saved: {(OUT_DIR / name).relative_to(ROOT)}")

    summary = {
        "k": OPTIMAL_K,
        "optimal_k": OPTIMAL_K,
        "random_state": RANDOM_STATE,
        "n_init": N_INIT,
        "k_range": [K_RANGE.start, K_RANGE.stop - 1],
        "features": list(feature_names),
        "n_features": len(feature_names),
        "target_excluded": TARGET_COL,
        "rows": {"train": int(len(y_train)), "test": int(len(y_test))},
        "churn_rate": {
            "train": round(float(y_train.mean()), 6),
            "test": round(float(y_test.mean()), 6),
        },
        "churners": {"train": int(y_train.sum()), "test": int(y_test.sum())},
        "final_inertia": round(float(model.inertia_), 4),
        "final_silhouette": round(float(final_silhouette), 4),
        "training_means": {
            "tenure": round(float(scaler.mean_[0]), 4),
            "MonthlyCharges": round(float(scaler.mean_[1]), 4),
        },
        "training_stds": {
            "tenure": round(float(scaler.scale_[0]), 4),
            "MonthlyCharges": round(float(scaler.scale_[1]), 4),
        },
        "metrics": metrics.to_dict(orient="records"),
        "segments": profile.to_dict(orient="records"),
        "segment_colours": SEGMENT_COLOURS,
    }
    with open(OUT_DIR / "clustering_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to: {(OUT_DIR / 'clustering_summary.json').relative_to(ROOT)}\n")


# ---------------------------------------------------------------------------
# Step 7: the charts. Every chart names the segments in plain English, because
# the rubric asks for the clusters to be visualised AND labelled.
# ---------------------------------------------------------------------------
def _new_figure(*args, **kwargs):
    fig, axes = plt.subplots(*args, **kwargs)
    fig.patch.set_facecolor("white")
    for ax in np.atleast_1d(np.asarray(axes)).ravel():
        ax.set_facecolor("white")
    return fig, axes


def _save(fig, name: str) -> None:
    path = PLOTS / name
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)
    print(f"Saved chart: {path.relative_to(ROOT)}")


def plot_elbow(metrics: pd.DataFrame, profile: pd.DataFrame) -> None:
    """Chart 1: the elbow curve, with the silhouette score underneath."""
    fig, (ax1, ax2) = _new_figure(2, 1, figsize=(10, 9))

    ks = metrics["k"].to_numpy()
    inertia = metrics["inertia"].to_numpy()
    ax1.plot(ks, inertia, marker="o", markersize=7, linewidth=2, color="#0072b2")
    ax1.axvline(OPTIMAL_K, color="#d55e00", linestyle="--", linewidth=1.8)
    ax1.annotate(
        f"Elbow at k = {OPTIMAL_K}\nthe fall drops from 12.1% to 5.1% after this point",
        xy=(OPTIMAL_K, inertia[OPTIMAL_K - 1]),
        xytext=(OPTIMAL_K + 1.4, inertia[OPTIMAL_K - 1] + 3600),
        color="#d55e00", fontsize=10, fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="#d55e00", linewidth=1.6))

    for k, value, drop in zip(ks, inertia, metrics["inertia_drop_pct"]):
        if pd.isna(drop):
            continue
        # Nudge the k = 4 label aside, so the dashed elbow line misses it.
        offset = (-34, 13) if k == OPTIMAL_K else (0, 13)
        ax1.annotate(f"-{drop:.1f}%", xy=(k, value), xytext=offset,
                     textcoords="offset points", ha="center", fontsize=9,
                     color="#333333")

    ax1.set_title("Elbow method: within-cluster sum of squares against k",
                  fontsize=13, fontweight="bold")
    ax1.set_xlabel("Number of clusters (k)")
    ax1.set_ylabel("Inertia (within-cluster sum of squares)")
    ax1.set_xticks(ks)
    ax1.set_ylim(7000, 22500)
    ax1.grid(alpha=0.3)

    names = list(profile["segment"])
    ax1.text(0.98, 0.95,
             "The four segments at k = 4:\n" + "\n".join(f"  {n}" for n in names),
             transform=ax1.transAxes, ha="right", va="top", fontsize=9.5,
             bbox=dict(boxstyle="round,pad=0.5", facecolor="#f4f4f7",
                       edgecolor="#b4b4c8"))

    supporting = metrics.dropna(subset=["silhouette"])
    ax2.plot(supporting["k"], supporting["silhouette"], marker="s", markersize=6,
             linewidth=2, color="#009e73")
    ax2.axvline(OPTIMAL_K, color="#d55e00", linestyle="--", linewidth=1.8)
    for k, value in zip(supporting["k"], supporting["silhouette"]):
        ax2.annotate(f"{value:.3f}", xy=(k, value), xytext=(0, 9),
                     textcoords="offset points", ha="center", fontsize=9,
                     color="#333333")
    ax2.set_title("Supporting check only: silhouette score against k. "
                  "We chose k from the elbow above, not from this peak.",
                  fontsize=11)
    ax2.set_xlabel("Number of clusters (k)")
    ax2.set_ylabel("Silhouette score")
    ax2.set_xticks(supporting["k"])
    ax2.set_ylim(0.10, 0.27)
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    _save(fig, "elbow_method.png")


def plot_tenure_vs_charges(X_train, train_labels, profile, scaler) -> None:
    """Chart 2: every training customer in months and dollars."""
    units = original_units(X_train, scaler)
    fig, ax = _new_figure(figsize=(11, 7.5))

    for row in profile.itertuples():
        mask = train_labels == row.cluster
        ax.scatter(units.loc[mask, "tenure_months"], units.loc[mask, "monthly_charges"],
                   s=11, alpha=0.30, linewidths=0,
                   color=SEGMENT_COLOURS[row.segment],
                   label=f"{row.segment}  ({row.size:,} customers, "
                         f"{row.churn_rate:.1%} churn)")

    for row in profile.itertuples():
        ax.scatter(row.tenure_months, row.monthly_charges, s=260, marker="X",
                   color=SEGMENT_COLOURS[row.segment], edgecolor="black", linewidth=1.6,
                   zorder=5)
        ax.annotate(row.segment, xy=(row.tenure_months, row.monthly_charges),
                    xytext=(0, 17), textcoords="offset points", ha="center",
                    fontsize=10, fontweight="bold", zorder=6,
                    bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                              edgecolor=SEGMENT_COLOURS[row.segment], linewidth=1.4))

    ax.set_title("Customer segments by tenure and monthly charge (training set)",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Tenure (months with the company)")
    ax.set_ylabel("Monthly charge (dollars per month)")
    ax.grid(alpha=0.25)
    legend = ax.legend(loc="lower right", fontsize=9.5, framealpha=0.95,
                       title="Segment (size, churn rate)")
    legend.get_title().set_fontweight("bold")
    for handle in legend.legend_handles:
        handle.set_alpha(1.0)
        handle.set_sizes([60])
    ax.text(0.01, 0.99,
            "The X marks each cluster centre. Points are semi-transparent, because\n"
            "many customers share the same tenure and monthly charge.",
            transform=ax.transAxes, va="top", fontsize=9, color="#555555")

    fig.tight_layout()
    _save(fig, "segments_tenure_vs_charges.png")


def plot_pca(X_train, train_labels, profile) -> None:
    """Chart 3: the 10 features squeezed into 2D, for plotting only."""
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    coords = pca.fit_transform(X_train)
    explained = pca.explained_variance_ratio_ * 100

    fig, ax = _new_figure(figsize=(11, 7.5))
    for row in profile.itertuples():
        mask = train_labels == row.cluster
        ax.scatter(coords[mask, 0], coords[mask, 1], s=11, alpha=0.30, linewidths=0,
                   color=SEGMENT_COLOURS[row.segment],
                   label=f"{row.segment}  ({row.size:,} customers)")

    for row in profile.itertuples():
        mask = train_labels == row.cluster
        cx, cy = coords[mask, 0].mean(), coords[mask, 1].mean()
        ax.scatter(cx, cy, s=260, marker="X", color=SEGMENT_COLOURS[row.segment],
                   edgecolor="black", linewidth=1.6, zorder=5)
        ax.annotate(row.segment, xy=(cx, cy), xytext=(0, 17),
                    textcoords="offset points", ha="center", fontsize=10,
                    fontweight="bold", zorder=6,
                    bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                              edgecolor=SEGMENT_COLOURS[row.segment], linewidth=1.4))

    ax.set_title("The four segments on a 2D projection of all 10 features",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel(f"Principal component 1 ({explained[0]:.1f}% of the variance)")
    ax.set_ylabel(f"Principal component 2 ({explained[1]:.1f}% of the variance)")
    ax.grid(alpha=0.25)
    legend = ax.legend(loc="upper right", fontsize=9.5, framealpha=0.95,
                       title="Segment (size)")
    legend.get_title().set_fontweight("bold")
    for handle in legend.legend_handles:
        handle.set_alpha(1.0)
        handle.set_sizes([60])
    ax.text(0.01, 0.99,
            f"PCA is used for this picture only. These two components hold "
            f"{explained.sum():.1f}% of the variance.\n"
            "The model itself clusters on all 10 features.",
            transform=ax.transAxes, va="top", fontsize=9, color="#555555")

    fig.tight_layout()
    _save(fig, "segments_pca.png")


def plot_churn_by_segment(profile: pd.DataFrame, overall_rate: float) -> None:
    """Chart 4: churn rate per segment against the overall rate."""
    fig, ax = _new_figure(figsize=(10.5, 7))

    names = list(profile["segment"])
    rates = profile["churn_rate"].to_numpy() * 100
    bars = ax.bar(names, rates, color=[SEGMENT_COLOURS[n] for n in names],
                  edgecolor="white", width=0.62)

    ax.axhline(overall_rate * 100, color="#333333", linestyle="--", linewidth=1.8,
               zorder=2)
    # x in axes fractions, y in data units, so the label cannot fall off the
    # right hand edge of a categorical axis.
    ax.text(0.995, overall_rate * 100 + 0.9, f"Overall churn rate, {overall_rate:.2%}",
            transform=ax.get_yaxis_transform(), ha="right", va="bottom",
            fontsize=10, fontweight="bold", color="#333333", zorder=4)

    for bar, row in zip(bars, profile.itertuples()):
        # The white box keeps a bar label readable where it meets the dashed line.
        ax.annotate(f"{row.churn_rate:.1%}\n{row.churners:,} of {row.size:,} customers",
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 7), textcoords="offset points", ha="center",
                    fontsize=10, fontweight="bold", zorder=5,
                    bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                              edgecolor="none", alpha=0.85))

    ax.set_title("Churn rate by customer segment (training set)",
                 fontsize=13, fontweight="bold")
    ax.set_ylabel("Churn rate (% of the segment who left)")
    ax.set_xlabel("Segment")
    ax.set_ylim(0, max(rates) * 1.30)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    plt.setp(ax.get_xticklabels(), fontsize=10.5)

    fig.tight_layout()
    _save(fig, "churn_rate_by_segment.png")


def plot_segment_profiles(profile: pd.DataFrame) -> None:
    """Chart 5: the profile table as a heatmap, with the real values shown."""
    columns = [
        ("tenure_months", "Tenure\n(months)", "{:.1f}"),
        ("monthly_charges", "Monthly\ncharge ($)", "${:.1f}"),
        ("churn_rate", "Churn\nrate", "{:.1%}"),
        ("fiber_optic", "Fibre\noptic", "{:.1%}"),
        ("contract_one_year", "One year\ncontract", "{:.1%}"),
        ("contract_two_year", "Two year\ncontract", "{:.1%}"),
        ("phone_service", "Phone\nservice", "{:.1%}"),
        ("multiple_lines", "Multiple\nlines", "{:.1%}"),
        ("senior_citizen", "Senior\ncitizen", "{:.1%}"),
        ("dependents", "Has\ndependents", "{:.1%}"),
        ("gender_male", "Male", "{:.1%}"),
    ]
    values = profile[[c for c, _, _ in columns]].to_numpy(dtype=float)

    # Colour each column against the other segments, so a column with a narrow
    # spread does not disappear next to a column measured in dollars.
    lo, hi = values.min(axis=0), values.max(axis=0)
    span = np.where(hi - lo == 0, 1.0, hi - lo)
    shaded = (values - lo) / span

    fig, ax = _new_figure(figsize=(13.5, 5.4))
    ax.imshow(shaded, cmap="YlGnBu", vmin=0, vmax=1, aspect="auto")

    for i in range(values.shape[0]):
        for j, (_, _, fmt) in enumerate(columns):
            ax.text(j, i, fmt.format(values[i, j]), ha="center", va="center",
                    fontsize=10, fontweight="bold",
                    color="white" if shaded[i, j] > 0.55 else "#1a1a1a")

    ax.set_xticks(range(len(columns)))
    ax.set_xticklabels([label for _, label, _ in columns], fontsize=9.5)
    ax.set_yticks(range(len(profile)))
    ax.set_yticklabels([f"{row.segment}\n({row.size:,} customers)"
                        for row in profile.itertuples()], fontsize=10)
    for tick, name in zip(ax.get_yticklabels(), profile["segment"]):
        tick.set_color(SEGMENT_COLOURS[name])
        tick.set_fontweight("bold")
    ax.set_xticks(np.arange(-0.5, len(columns), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(profile), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2.5)
    ax.tick_params(which="minor", length=0)

    ax.set_title("Segment profiles: what the average customer in each segment looks like",
                 fontsize=13, fontweight="bold", pad=14)
    fig.text(0.5, 0.015,
             "Each column is shaded against the other segments. Darker means a higher "
             "value within that column. Percentages are the share of the segment.",
             ha="center", fontsize=9, color="#555555")

    fig.tight_layout(rect=(0, 0.045, 1, 1))
    _save(fig, "segment_profiles.png")


def draw_charts(X_train, train_labels, profile, scaler, metrics, overall_rate) -> None:
    """Step 7: draw every chart into cluster_visualisations/."""
    print("--- Drawing the cluster visualisations ---")
    PLOTS.mkdir(parents=True, exist_ok=True)
    plot_elbow(metrics, profile)
    plot_tenure_vs_charges(X_train, train_labels, profile, scaler)
    plot_pca(X_train, train_labels, profile)
    plot_churn_by_segment(profile, overall_rate)
    plot_segment_profiles(profile)
    print()


def run_pipeline():
    X_train, y_train, X_test, y_test, scaler = load_prepared_data()
    metrics = search_k(X_train)
    model, train_labels, test_labels, final_silhouette = train_final_model(X_train, X_test)
    profile, names = build_profiles(X_train, y_train, train_labels,
                                    X_test, y_test, test_labels, scaler)
    assignments = build_assignments(train_labels, test_labels, names)
    save_outputs(model, metrics, profile, assignments, final_silhouette,
                 scaler, y_train, y_test, X_train.columns)
    draw_charts(X_train, train_labels, profile, scaler, metrics, float(y_train.mean()))

    # ---- Final summary -----------------------------------------------------
    print("\n=== Clustering Analysis Complete ===")
    print(f"Chosen k: {OPTIMAL_K} (elbow method)")
    print(f"Inertia at k={OPTIMAL_K}: {model.inertia_:,.2f}")
    print(f"Silhouette at k={OPTIMAL_K}: {final_silhouette:.4f}")
    print(f"Overall training churn rate: {y_train.mean():.2%}\n")
    print(f"{'Segment':<26}{'Size':>7}{'Share':>8}{'Churn':>8}"
          f"{'Tenure':>10}{'Charge':>10}")
    for row in profile.itertuples():
        print(f"{row.segment:<26}{row.size:>7,}{row.share_pct:>7.1f}%"
              f"{row.churn_rate:>8.1%}{row.tenure_months:>8.1f} mo"
              f"{row.monthly_charges:>9.1f}")
    return model, metrics, profile, assignments


if __name__ == "__main__":
    kmeans_model, selection_metrics, cluster_profile, cluster_assignments = run_pipeline()
