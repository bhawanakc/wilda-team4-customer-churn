"""
Builds the two Clustering Analysis documents from the pipeline's own output, so
every number in them comes from clustering_summary.json rather than being
typed by hand:

  - Clustering_Analysis/README.md                 the segments, their profiles and the actions
  - Clustering_Analysis/elbow_method_results.pdf  how k was chosen and the trained model

The code snippets in the PDF are read straight out of clustering_analysis.py.

Run from the repository root, after the pipeline:
    python Clustering_Analysis/clustering_analysis.py
    python Clustering_Analysis/build_docs.py
"""

import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (Image, KeepTogether, ListFlowable, ListItem,
                                PageBreak, Paragraph, SimpleDocTemplate, Spacer,
                                Table, TableStyle)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SUMMARY_PATH = HERE / "clustering_summary.json"
PIPELINE_PATH = HERE / "clustering_analysis.py"
ELBOW_FIGURE = HERE / "cluster_visualisations" / "elbow_method.png"
README_PATH = HERE / "README.md"
PDF_PATH = HERE / "elbow_method_results.pdf"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"{path.name} not found. Run 'python Clustering_Analysis/clustering_analysis.py' first."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def snippet(first_line_contains: str, last_line_equals: str | None = None,
            n_lines: int = 1) -> str:
    """Pull a snippet verbatim out of clustering_analysis.py so it cannot drift."""
    lines = PIPELINE_PATH.read_text(encoding="utf-8").splitlines()
    for i, line in enumerate(lines):
        if first_line_contains in line:
            if last_line_equals is None:
                block = lines[i:i + n_lines]
            else:
                j = i
                while lines[j].rstrip() != last_line_equals:
                    j += 1
                block = lines[i:j + 1]
            # Remove the common leading indentation.
            pad = min(len(b) - len(b.lstrip()) for b in block if b.strip())
            return "\n".join(b[pad:] if b.strip() else "" for b in block)
    raise ValueError(f"Snippet not found in {PIPELINE_PATH.name}: {first_line_contains!r}")


def pct(x: float, places: int = 1) -> str:
    return f"{x * 100:.{places}f}%"


def drop_at(s: dict, k: int) -> float:
    """The percentage fall in inertia when the cluster count rose to k."""
    return next(row["inertia_drop_pct"] for row in s["metrics"] if row["k"] == k)


def contract_share(seg: dict) -> float:
    """Share of the segment on a one year or a two year contract."""
    return seg["contract_one_year"] + seg["contract_two_year"]


def month_to_month(seg: dict) -> float:
    """Share of the segment with no fixed term, so both contract flags are 0."""
    return 1 - contract_share(seg)


# ---------------------------------------------------------------------------
# The written interpretation of each segment. Every figure is read from the
# profile, so the text moves with the data instead of being typed by hand.
# ---------------------------------------------------------------------------
def by_name(s: dict, name: str) -> dict:
    """Look a segment up by name, so the text never depends on row order."""
    return next(seg for seg in s["segments"] if seg["segment"] == name)


def segment_story(seg: dict, s: dict) -> tuple[str, list[str]]:
    """Return the 'who and why' paragraph and the actions for one segment."""
    name = seg["segment"]
    overall = s["churn_rate"]["train"]
    lt_low = by_name(s, "Long-term low spenders")
    lt_high = by_name(s, "Long-term high spenders")
    new_high = by_name(s, "New high spenders")

    common = (
        f"This segment holds {seg['size']:,} of the {s['rows']['train']:,} training customers, "
        f"{seg['share_pct']:.1f}% of the base. {seg['churners']:,} of them left, a churn rate of "
        f"{pct(seg['churn_rate'])}, against {pct(overall, 2)} across all customers. The average "
        f"customer here has been with the company {seg['tenure_months']:.1f} months and pays "
        f"${seg['monthly_charges']:.2f} a month. {pct(month_to_month(seg))} are on a month to month "
        f"plan. In the held out test set the same segment holds {seg['test_size']:,} customers and "
        f"churns at {pct(seg['test_churn_rate'])}."
    )

    if name == "New high spenders":
        why = (
            "Two numbers in the profile line up with the churn rate. These customers are well "
            f"below the {s['training_means']['tenure']:.1f} month average tenure and well above the "
            f"${s['training_means']['MonthlyCharges']:.2f} average bill, so they pay a lot early. "
            "Nothing holds them either, because more than half have no fixed term and can leave "
            "at the end of any month. The segment also carries the largest share of senior "
            f"citizens, {pct(seg['senior_citizen'])}."
        )
        actions = [
            f"Put the retention budget here first. This segment produced {seg['churners']:,} of the "
            f"{s['churners']['train']:,} churners in the training data, more than the other three "
            f"segments added together.",
            f"Offer a fixed term plan to the {pct(month_to_month(seg))} on month to month, priced at "
            f"or below the ${seg['monthly_charges']:.2f} they already pay, within the first "
            f"{seg['tenure_months']:.0f} months.",
        ]

    elif name == "New low spenders":
        why = (
            "This is the newest segment and the cheapest of the four. Its churn rate sits just "
            f"below the {pct(overall, 2)} overall rate, so these customers leave at close to the "
            "average pace. The low bill is what separates them from the New high spenders, who "
            f"pay ${new_high['monthly_charges'] - seg['monthly_charges']:.2f} more a month and churn "
            f"at {pct(new_high['churn_rate'])}. This segment also takes the smallest bundle, with "
            f"only {pct(seg['phone_service'])} on phone service against "
            f"{pct(new_high['phone_service'])} in the New high spenders."
        )
        actions = [
            f"Target the first year. The average customer is {seg['tenure_months']:.1f} months in, so "
            f"a fixed term offer reaches them well before the two long tenure segments, which sit "
            f"at {lt_low['tenure_months']:.1f} months and above.",
            f"Test a phone service add-on with the {pct(1 - seg['phone_service'])} who do not have "
            f"one, and hold the bill near the ${seg['monthly_charges']:.2f} they pay now.",
        ]

    elif name == "Long-term high spenders":
        monthly_at_risk = seg["churners"] * seg["monthly_charges"]
        why = (
            "These customers pay the most and have stayed the longest of the four segments. Long "
            f"tenure goes with the churn rate of {pct(seg['churn_rate'])}, below the "
            f"{pct(overall, 2)} overall rate. The bills are what makes the segment matter. The "
            f"{seg['churners']:,} who left were billing about ${monthly_at_risk:,.0f} a month "
            "between them, using the segment average."
        )
        actions = [
            f"Watch this segment for revenue rather than for counts. Each departure costs about "
            f"${seg['monthly_charges']:.2f} a month, {seg['monthly_charges'] / lt_low['monthly_charges']:.1f} "
            f"times the bill of a Long-term low spender.",
            f"Review the bills of the {pct(seg['fiber_optic'])} on fibre optic, the highest share of "
            f"the four segments, since this segment pays "
            f"${seg['monthly_charges'] - new_high['monthly_charges']:.2f} more a month than "
            f"the New high spenders.",
        ]

    else:  # Long-term low spenders
        why = (
            "This segment churns least of the four. Long tenure on a small bill is the pattern, "
            f"and it holds the largest share of customers with dependents, {pct(seg['dependents'])}, "
            f"and the smallest share of senior citizens, {pct(seg['senior_citizen'])}. Its churn "
            f"rate is {pct(by_name(s, 'New high spenders')['churn_rate'] - seg['churn_rate'])} below "
            "the New high spenders."
        )
        actions = [
            f"Spend no retention budget here. Only {seg['churners']:,} of {seg['size']:,} customers "
            f"left, so the segment returns the least per dollar of the four.",
            f"Treat it as the upgrade list instead. These customers pay "
            f"${seg['monthly_charges']:.2f} against ${lt_high['monthly_charges']:.2f} in the "
            f"Long-term high spenders, and they have stayed almost as long, "
            f"{seg['tenure_months']:.1f} months against {lt_high['tenure_months']:.1f}.",
        ]

    return f"{common} {why}", actions


# ---------------------------------------------------------------------------
# README.md
# ---------------------------------------------------------------------------
def build_readme(s: dict) -> None:
    segments = s["segments"]
    top = segments[0]

    seg_rows = "\n".join(
        f"| {seg['segment']} | {seg['size']:,} | {seg['share_pct']:.1f}% | "
        f"{pct(seg['churn_rate'])} | {seg['tenure_months']:.1f} | "
        f"${seg['monthly_charges']:.2f} | {pct(seg['fiber_optic'])} | "
        f"{pct(contract_share(seg))} |"
        for seg in segments
    )

    def metric_row(row: dict) -> str:
        fall = "" if row["inertia_drop_pct"] is None else f"{row['inertia_drop_pct']:.2f}%"
        sil = "" if row["silhouette"] is None else f"{row['silhouette']:.4f}"
        return f"| {row['k']} | {row['inertia']:,.2f} | {fall} | {sil} |"

    metric_rows = "\n".join(metric_row(row) for row in s["metrics"])

    priority_rows = "\n".join(
        f"| {i} | {seg['segment']} | {seg['size']:,} | {pct(seg['churn_rate'])} | "
        f"{seg['churners']:,} |"
        for i, seg in enumerate(segments, start=1)
    )

    sections = []
    for seg in segments:
        story, actions = segment_story(seg, s)
        action_lines = "\n".join(f"{n}. {a}" for n, a in enumerate(actions, start=1))
        sections.append(
            f"### {seg['segment']}\n\n{story}\n\nWhat the company can do:\n\n{action_lines}\n"
        )
    segment_sections = "\n".join(sections)

    md = f"""# Clustering analysis

Stage 2 clustering for the customer churn analysis. Owner: Ramesh Kunwar.

## Overview

[`clustering_analysis.py`](clustering_analysis.py) reads the training set that the data
preparation stage wrote and groups the customers into {s['k']} segments with K-Means. The
script runs in this order:

1. Load `Data_Preparation/train_set.csv` and split `Churn` off the {s['n_features']} features.
2. Fit K-Means for every k from {s['k_range'][0]} to {s['k_range'][1]} and record the inertia, the fall in inertia and the silhouette score.
3. Pick k = {s['k']} with the elbow method.
4. Train the final model, then assign the {s['rows']['test']:,} test rows with `model.predict`.
5. Profile each cluster and name it from its own average tenure and monthly charge.
6. Draw the five charts in `cluster_visualisations/`.

The model never sees `Churn`. Stage 3 predicts it, so a cluster built from it would hand
the answer to the model. The pipeline uses `Churn` only after clustering, to measure the
churn rate of each finished segment.

To rebuild every output, run this from the repository root:

```bash
python Clustering_Analysis/clustering_analysis.py
```

[`build_docs.py`](build_docs.py) writes this README and the PDF from the numbers the
pipeline saves, so nobody types the figures by hand:

```bash
python Clustering_Analysis/build_docs.py
```

## Files produced

| File | Contents |
|---|---|
| `clustering_analysis.py` | The pipeline. Chooses k, trains the model, profiles and names the segments, and draws the charts. |
| `clustering_analysis.ipynb` | A notebook that runs the same script and explains each step. Both write identical files. |
| `build_docs.py` | Builds this README and `elbow_method_results.pdf` from `clustering_summary.json`. |
| `kmeans_model.pkl` | The trained K-Means model, fitted on the {s['rows']['train']:,} training rows and the {s['n_features']} features. |
| `cluster_selection_metrics.csv` | Inertia, the percentage fall in inertia and the silhouette score for every k from {s['k_range'][0]} to {s['k_range'][1]}. |
| `cluster_profiles.csv` | One row per segment, sorted by churn rate. Tenure and charges are in months and dollars. |
| `cluster_assignments.csv` | The segment of all {s['rows']['train'] + s['rows']['test']:,} customers, for the Stage 3 model to join on. |
| `clustering_summary.json` | Every number used in this README and the PDF. |
| `elbow_method_results.pdf` | How k was chosen, the metrics table, the elbow chart and the trained model settings. |
| `README.md` | This file. The segments, their profiles and the action for each one. |
| `cluster_visualisations/elbow_method.png` | The inertia curve with the elbow at k = {s['k']} marked, and the silhouette scores below it. |
| `cluster_visualisations/segments_tenure_vs_charges.png` | Every training customer in months and dollars, coloured and labelled by segment. |
| `cluster_visualisations/segments_pca.png` | The {s['n_features']} features projected into 2D, coloured and labelled by segment. |
| `cluster_visualisations/churn_rate_by_segment.png` | Churn rate per segment against the {pct(s['churn_rate']['train'], 2)} overall rate. |
| `cluster_visualisations/segment_profiles.png` | The profile table as a heatmap, with the real months, dollars and percentages shown. |

## How we chose k

We used the elbow method, which the assignment requires. Inertia is the total squared
distance from each customer to its own cluster centre, and it falls every time k rises, so
the choice comes from the size of each fall rather than from the value. Inertia falls
{drop_at(s, 2):.1f}% at k = 2, {drop_at(s, 3):.1f}% at k = 3 and {drop_at(s, 4):.1f}% at k = {s['k']}, then only {drop_at(s, 5):.1f}% at k = 5 and less after
that, so k = {s['k']} is the last cluster that buys a large improvement. The silhouette score is a
supporting check and it peaks at k = 2 ({next(r['silhouette'] for r in s['metrics'] if r['k'] == 2):.3f}), but two clusters split the base into
little more than new and long standing customers, which is too coarse to act on. At k = {s['k']}
the churn rates run from {pct(segments[-1]['churn_rate'])} to {pct(top['churn_rate'])}, which separates the customers worth
contacting from the ones who stay. [`elbow_method_results.pdf`](elbow_method_results.pdf)
sets out the full reasoning.

| k | Inertia | Fall in inertia | Silhouette |
|---|---|---|---|
{metric_rows}

## The segments

| Segment | Customers (train) | % of customers | Churn rate | Avg tenure (months) | Avg monthly charge ($) | % on fibre | % on a one or two year contract |
|---|---|---|---|---|---|---|---|
{seg_rows}

Cluster ID numbers can move between runs and between library versions, so every document
refers to the segments by name. The names come from code, not from a person reading the
table. The pipeline converts each cluster centre back into months and dollars and compares
it with the training means, {s['training_means']['tenure']:.2f} months and ${s['training_means']['MonthlyCharges']:.2f}. Tenure below the mean gives
"New" and above gives "Long-term". A monthly charge below the mean gives "low spenders"
and above gives "high spenders".

{segment_sections}
## Which segment to act on first

Size multiplied by churn rate gives the number of churners a segment produces, which is
the order to work through:

| Priority | Segment | Customers | Churn rate | Churners |
|---|---|---|---|---|
{priority_rows}

**{top['segment']}** come first. They are both the largest segment, at {top['size']:,}
customers, and the one with the highest churn rate, at {pct(top['churn_rate'])}. Those two
together produced {top['churners']:,} churners, {pct(top['churners'] / s['churners']['train'])} of the
{s['churners']['train']:,} in the training data. The same segment churns at {pct(top['test_churn_rate'])} in
the test set, which the model never saw, so the pattern is not an artefact of the rows it
was fitted on.

One qualifier on the order. **Long-term high spenders** produce fewer churners than
**New low spenders**, {by_name(s, 'Long-term high spenders')['churners']:,} against {by_name(s, 'New low spenders')['churners']:,}, but each one
pays ${by_name(s, 'Long-term high spenders')['monthly_charges']:.2f} a month against ${by_name(s, 'New low spenders')['monthly_charges']:.2f}. Ranked by
the monthly billing that leaves with them, the two swap places.

## Assigning new customers to a segment

New data must be scaled with the same scaler the training data used, then passed to the
model. Load both files and call `predict`:

```python
import joblib
import pandas as pd

scaler = joblib.load("Data_Preparation/scaler.pkl")
model = joblib.load("Clustering_Analysis/kmeans_model.pkl")

SCALE_COLS = ["tenure", "MonthlyCharges"]

new_customers = pd.read_csv("new_customers.csv")      # encoded, not yet scaled
new_customers[SCALE_COLS] = scaler.transform(new_customers[SCALE_COLS])

clusters = model.predict(new_customers[list(model.feature_names_in_)])
```

The columns must arrive encoded the same way as the training data, which
`Data_Preparation/encoding_map.json` records, and in the order `model.feature_names_in_`
gives. `cluster_profiles.csv` maps each cluster number back to its segment name.

## Limitations

The segments split on tenure and monthly charge. The other eight features barely move
between them. Fibre optic runs from {pct(min(x['fiber_optic'] for x in segments))} to
{pct(max(x['fiber_optic'] for x in segments))} across the four segments, and the share on a one or two year
contract runs from {pct(min(contract_share(x) for x in segments))} to {pct(max(contract_share(x) for x in segments))}.

This happens because of how K-Means measures distance. `tenure` and `MonthlyCharges` are
z-scores, so they spread roughly from -2 to 3. The other eight columns hold only 0 or 1.
The two wider columns therefore drive most of the distance between customers, and the
model groups mainly on them.

Two points follow from this. The segment names describe tenure and spending, and nothing
else, so they should not be read as statements about contract type or internet type.
Stage 3 should keep using contract type and internet type as predictors in their own
right, because this clustering does not capture them. The segment in
`cluster_assignments.csv` is one extra feature for that model, not a replacement for the
columns it is built from.

The silhouette score at k = {s['k']} is {s['final_silhouette']:.4f}. A score near 0 means the clusters sit close
together rather than in separate groups, which the tenure and charge chart shows. The
segments are useful for sorting customers into action lists. They are not evidence that
four distinct customer types exist in the data.
"""
    README_PATH.write_text(md, encoding="utf-8")
    print(f"Wrote {README_PATH.relative_to(ROOT)}")


# ---------------------------------------------------------------------------
# elbow_method_results.pdf
# ---------------------------------------------------------------------------
def build_pdf(s: dict) -> None:
    segments = s["segments"]
    top, bottom = segments[0], segments[-1]

    styles = getSampleStyleSheet()
    body = ParagraphStyle("body", parent=styles["BodyText"], fontSize=9.5, leading=13.5,
                          alignment=TA_JUSTIFY, spaceAfter=6)
    h1 = ParagraphStyle("h1", parent=styles["Heading1"], fontSize=15, spaceAfter=2,
                        textColor=colors.HexColor("#1f3864"))
    h2 = ParagraphStyle("h2", parent=styles["Heading2"], fontSize=11, spaceBefore=8,
                        spaceAfter=3, textColor=colors.HexColor("#1f3864"))
    sub = ParagraphStyle("sub", parent=body, fontSize=9, textColor=colors.HexColor("#555555"),
                         spaceAfter=10)
    code = ParagraphStyle("code", parent=styles["Code"], fontSize=7.6, leading=9.6,
                          backColor=colors.HexColor("#f4f4f7"), borderPadding=5,
                          leftIndent=4, spaceBefore=3, spaceAfter=8)
    caption = ParagraphStyle("caption", parent=body, fontSize=8,
                             textColor=colors.HexColor("#555555"), alignment=1, spaceBefore=2)

    def code_block(text: str):
        html = (text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                    .replace(" ", "&nbsp;").replace("\n", "<br/>"))
        return Paragraph(html, code)

    def bullets(items):
        return ListFlowable([ListItem(Paragraph(t, body), leftIndent=12) for t in items],
                            bulletType="bullet", start="square", leftIndent=12,
                            bulletFontSize=6)

    def table(data, widths, pad=4):
        t = Table(data, colWidths=widths, hAlign="LEFT")
        t.setStyle(TableStyle([
            ("TOPPADDING", (0, 0), (-1, -1), pad),
            ("BOTTOMPADDING", (0, 0), (-1, -1), pad),
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f3864")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
            ("FONTSIZE", (0, 0), (-1, -1), 8.5),
            ("ALIGN", (1, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#b4b4c8")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#eef1f7")]),
        ]))
        return t

    story = []
    story.append(Paragraph("Choosing the number of clusters with the elbow method", h1))
    story.append(Paragraph(
        "WILDA Team 4 &middot; ACS Work Integrated Learning, Data Analytics &middot; Stage 2 "
        "Clustering Analysis &middot; Ramesh Kunwar", sub))

    # 1. Purpose
    story.append(Paragraph("1. Purpose", h2))
    story.append(Paragraph(
        "K-Means splits customers into a fixed number of groups. It places k centres in the "
        "data, puts every customer with the nearest centre, moves each centre to the average "
        "of the customers it collected, and repeats until the centres stop moving. The result "
        "is k groups whose members sit closer to their own centre than to any other.", body))
    story.append(Paragraph(
        "K-Means cannot work out k on its own. The number of clusters is an input, fixed "
        "before training starts, and the algorithm returns exactly that many groups whatever "
        "number it is given. Ask for two and it returns two. Ask for ten and it returns ten. "
        "Neither answer says whether that count suits the data. This document sets out how we "
        f"chose k = {s['k']} for the {s['rows']['train']:,} customers in the training set, and what the "
        "trained model produced.", body))

    # 2. Data used
    story.append(Paragraph("2. Data used", h2))
    story.append(Paragraph(
        f"The model is fitted on the {s['rows']['train']:,} rows of "
        "<font face='Courier'>Data_Preparation/train_set.csv</font>, using these "
        f"{s['n_features']} features: " + ", ".join(s["features"]) + ". The data preparation "
        "stage already scaled <b>tenure</b> and <b>MonthlyCharges</b> with a StandardScaler "
        "fitted on this same training set, so no rescaling happens here.", body))
    story.append(bullets([
        f"<b>Churn is excluded.</b> Stage 3 predicts churn. A cluster built from the churn "
        f"column would carry the answer inside the feature, and any model trained on it would "
        f"score well for the wrong reason. The pipeline asserts that the target is absent and "
        f"that exactly {s['n_features']} features remain. Churn is used only after clustering, to "
        f"measure the churn rate of each finished segment.",
        f"<b>Only the training set is used to fit.</b> The {s['rows']['test']:,} test rows are "
        f"assigned afterwards with <font face='Courier'>model.predict</font>, which puts each "
        f"one with its nearest existing centre without moving that centre. This keeps the test "
        f"set unseen, matching how the scaler was fitted in the preparation stage.",
    ]))

    # 3. The elbow method
    story.append(Paragraph("3. The elbow method", h2))
    story.append(Paragraph(
        "Inertia is the within-cluster sum of squares. It adds up the squared distance from "
        "every customer to the centre of its own cluster. A small inertia means customers sit "
        "close to their centres.", body))
    story.append(Paragraph(
        "Inertia always falls as k rises, which is why its value alone cannot choose k. Adding "
        "a centre gives every customer one more centre to be near, so no customer can end up "
        "further away. At k equal to the number of customers, every customer is its own cluster "
        "and inertia reaches 0. Picking the k with the smallest inertia would therefore always "
        "pick the largest k tested, and each cluster would describe fewer customers than the "
        "one before.", body))
    story.append(Paragraph(
        "The elbow method reads the size of each fall instead of the value. Early clusters "
        "split groups that really are separate, so inertia drops sharply. Once the real groups "
        "have been found, another cluster only cuts an existing group in half, and the drop "
        "becomes small. The elbow is the last k before the drops go flat. We report the "
        "percentage fall at each step, so the reading is a number rather than a judgement "
        "about the shape of a line.", body))

    # 4. Results
    story.append(Paragraph("4. Results", h2))
    story.append(Paragraph(
        f"The pipeline fits K-Means for every k from {s['k_range'][0]} to {s['k_range'][1]} with "
        f"n_init={s['n_init']} and random_state={s['random_state']}. The silhouette score is "
        f"computed on all {s['rows']['train']:,} training rows, with no sampling. It is not defined "
        "at k = 1, because a single cluster has nothing to be compared against.", body))
    # KeepTogether stops the 11 row table being split across a page boundary,
    # which would leave the first page holding only its header.
    story.append(KeepTogether(table(
        [["k", "Inertia", "Fall in inertia", "Silhouette"]] +
        [[str(row["k"]), f"{row['inertia']:,.2f}",
          "" if row["inertia_drop_pct"] is None else f"{row['inertia_drop_pct']:.2f}%",
          "" if row["silhouette"] is None else f"{row['silhouette']:.4f}"]
         for row in s["metrics"]],
        [2.4 * cm, 3.6 * cm, 3.6 * cm, 3.4 * cm], pad=2)))
    story.append(Spacer(1, 8))

    story.append(KeepTogether([
        Image(str(ELBOW_FIGURE), width=11.9 * cm, height=10.71 * cm),
        Paragraph(
            f"Inertia against k, with the elbow at k = {s['k']} marked and each point labelled "
            "with its percentage fall. The silhouette score below is a supporting check.",
            caption),
    ]))
    story.append(Spacer(1, 6))

    # 5. Choice of k
    story.append(Paragraph("5. Choice of k", h2))
    story.append(Paragraph(
        f"We chose k = {s['k']}. Inertia falls {drop_at(s, 2):.1f}% at k = 2, {drop_at(s, 3):.1f}% at "
        f"k = 3 and {drop_at(s, 4):.1f}% at k = {s['k']}. The next step gives only {drop_at(s, 5):.1f}%, and every "
        f"step after that gives less than {max(drop_at(s, k) for k in range(6, 11)):.1f}%. The fall from "
        f"{drop_at(s, 4):.1f}% to {drop_at(s, 5):.1f}% is the sharpest change in the sequence, so k = {s['k']} is the "
        "last cluster that buys a large improvement.", body))
    story.append(Paragraph(
        f"The silhouette score is highest at k = 2, at "
        f"{next(r['silhouette'] for r in s['metrics'] if r['k'] == 2):.4f}, and falls to "
        f"{s['final_silhouette']:.4f} at k = {s['k']}. We did not follow that peak, for two reasons. The "
        "assignment asks for the elbow method, and silhouette is recorded here as a check "
        "rather than as the rule. Two clusters would also split the base into little more than "
        "newer and longer standing customers, which gives the business one group to contact and "
        "one to leave alone.", body))
    story.append(Paragraph(
        f"At k = {s['k']} the segments separate on churn, from {pct(bottom['churn_rate'])} to "
        f"{pct(top['churn_rate'])} against {pct(s['churn_rate']['train'], 2)} overall. {top['segment']} "
        f"hold {top['size']:,} customers and {top['churners']:,} of the {s['churners']['train']:,} "
        f"churners. {bottom['segment']} hold {bottom['size']:,} customers and {bottom['churners']:,} "
        "churners. That is a difference the business can act on, because it says where to spend "
        "a retention budget and where not to.", body))

    story.append(PageBreak())

    # 6. The trained model
    story.append(Paragraph("6. The trained model", h2))
    story.append(Paragraph(
        f"The final model is a KMeans with n_clusters={s['k']}, n_init={s['n_init']} and "
        f"random_state={s['random_state']}. n_init={s['n_init']} runs the algorithm {s['n_init']} times from "
        "different starting centres and keeps the run with the lowest inertia, because a single "
        "run can settle on a poor set of centres. random_state fixes the starting points, so "
        f"every run of the pipeline returns the same model. It reaches an inertia of "
        f"{s['final_inertia']:,.2f} and a silhouette score of {s['final_silhouette']:.4f}.", body))
    story.append(table(
        [["Segment", "Customers", "% of base", "Churners", "Churn rate",
          "Tenure (months)", "Charge ($)"]] +
        [[seg["segment"], f"{seg['size']:,}", f"{seg['share_pct']:.1f}%",
          f"{seg['churners']:,}", pct(seg["churn_rate"]),
          f"{seg['tenure_months']:.1f}", f"{seg['monthly_charges']:.2f}"]
         for seg in segments],
        [4.3 * cm, 2.0 * cm, 1.8 * cm, 1.8 * cm, 1.9 * cm, 2.3 * cm, 1.8 * cm]))
    story.append(Spacer(1, 8))
    story.append(Paragraph(
        "The training step fits the model on the training features, then assigns both sets:", body))
    story.append(code_block(snippet("model = KMeans(n_clusters=OPTIMAL_K", n_lines=5)))
    story.append(Paragraph(
        "The fitted model is saved with joblib, so later stages load it instead of training "
        "their own:", body))
    story.append(code_block(snippet("joblib.dump(model,", n_lines=1)))
    story.append(Paragraph(
        "<font face='Courier'>kmeans_model.pkl</font> stores the four centres, the settings "
        f"above and the {s['n_features']} feature names in "
        "<font face='Courier'>feature_names_in_</font>, which does not include Churn.", body))

    # 7. Limitations
    story.append(Paragraph("7. Limitations", h2))
    story.append(Paragraph(
        "The four segments differ on tenure and monthly charge, and barely differ on anything "
        f"else. Fibre optic runs only from {pct(min(x['fiber_optic'] for x in segments))} to "
        f"{pct(max(x['fiber_optic'] for x in segments))} across the four segments. The share on a one or "
        f"two year contract runs from {pct(min(contract_share(x) for x in segments))} to "
        f"{pct(max(contract_share(x) for x in segments))}. Phone service is the only 0/1 feature that "
        "separates the segments at all.", body))
    story.append(Paragraph(
        "This follows from how K-Means measures distance. tenure and MonthlyCharges are "
        "z-scores, so they spread roughly from -2 to 3. The other eight columns hold 0 or 1 and "
        "can differ by at most 1. The two wider columns therefore account for most of the "
        "distance between any two customers, and the model groups on them. The segments should "
        "be read as tenure and spending groups, not as statements about contract type or "
        "internet type.", body))
    story.append(Paragraph(
        "Stage 3 should keep contract type and internet type as predictors in their own right. "
        "This clustering does not capture them, so the segment label cannot stand in for them. "
        "The segment in <font face='Courier'>cluster_assignments.csv</font> is one extra "
        "feature for that model, alongside the original columns rather than instead of them.", body))
    story.append(Paragraph(
        f"The silhouette score of {s['final_silhouette']:.4f} is worth stating plainly. A score near 0 "
        "means neighbouring clusters sit close together rather than in separated groups, which "
        "the tenure and charge chart also shows. The segments sort customers into useful action "
        "lists. They are not evidence that four distinct customer types exist in this data.", body))

    doc = SimpleDocTemplate(
        str(PDF_PATH), pagesize=A4,
        leftMargin=2.2 * cm, rightMargin=2.2 * cm, topMargin=1.8 * cm, bottomMargin=1.8 * cm,
        title="Elbow method results, customer churn clustering analysis",
        author="Ramesh Kunwar, WILDA Team 4",
    )
    doc.build(story)
    print(f"Wrote {PDF_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    summary = load_json(SUMMARY_PATH)
    build_readme(summary)
    build_pdf(summary)
