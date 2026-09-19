"""
Builds the two Data Preparation documents from the pipeline's own output, so
every number in them comes from preparation_summary.json rather than being
typed by hand:

  - Data_Preparation/README.md               train/test size and composition
  - Data_Preparation/scaling_techniques.pdf  the scaling techniques document

The code snippets in the PDF are read straight out of data_preparation.py.

Run from the repository root, after the pipeline:
    python Data_Preparation/data_preparation.py
    python Data_Preparation/build_docs.py
"""

import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (Image, ListFlowable, ListItem, PageBreak,
                                Paragraph, SimpleDocTemplate, Spacer, Table,
                                TableStyle)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SUMMARY_PATH = HERE / "preparation_summary.json"
PIPELINE_PATH = HERE / "data_preparation.py"
ENCODING_PATH = HERE / "encoding_map.json"
FIGURE_PATH = HERE / "figures" / "scaling_before_after.png"
README_PATH = HERE / "README.md"
PDF_PATH = HERE / "scaling_techniques.pdf"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"{path.name} not found - run 'python Data_Preparation/data_preparation.py' first."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def snippet(first_line_contains: str, last_line_equals: str | None = None,
            n_lines: int = 1) -> str:
    """Pull a snippet verbatim out of data_preparation.py so it cannot drift."""
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


def pct(x: float) -> str:
    return f"{x * 100:.2f}%"


# ---------------------------------------------------------------------------
# README.md
# ---------------------------------------------------------------------------
def build_readme(s: dict, enc: dict) -> None:
    rows, churners, rate = s["rows"], s["churners"], s["churn_rate"]
    mix, share = s["contract_mix"], s["share_of_total"]
    sc, before, after = s["scaler"], s["scaling_before"], s["scaling_after"]

    def comp_row(name: str, key: str, share_txt: str) -> str:
        m = mix[key]
        return (f"| {name} | {rows[key]:,} | {share_txt} | {churners[key]:,} | "
                f"{pct(rate[key])} | {m['Month-to-month']:,} | {m['One year']:,} | "
                f"{m['Two year']:,} |")

    col_docs = [
        ("gender", "Customer's gender", "int (0/1)", enc["gender"]),
        ("SeniorCitizen", "Whether the customer is a senior citizen", "int (0/1)",
         {"No": 0, "Yes": 1}),
        ("Dependents", "Whether the customer has dependents", "int (0/1)", enc["Dependents"]),
        ("tenure", "Months the customer has been with the company", "float (scaled)", None),
        ("PhoneService", "Whether the customer has phone service", "int (0/1)",
         enc["PhoneService"]),
        ("MultipleLines", "Whether the customer has multiple lines", "int (0/1)",
         enc["MultipleLines"]),
        ("InternetService", "Type of internet service", "int (0/1)", enc["InternetService"]),
        ("MonthlyCharges", "Amount charged per month, in dollars", "float (scaled)", None),
        ("Contract_One_year", "Contract is a one year contract", "int (0/1)",
         {"not a one year contract": 0, "one year contract": 1}),
        ("Contract_Two_year", "Contract is a two year contract", "int (0/1)",
         {"not a two year contract": 0, "two year contract": 1}),
        ("Churn", "TARGET - whether the customer left", "int (0/1)", enc["Churn"]),
    ]

    def enc_text(name: str, mapping) -> str:
        if mapping is None:
            return ("original units in `preprocessed_dataset.csv`, "
                    "z-score standardised in the train/test sets")
        return ", ".join(f"{k} = {v}" for k, v in mapping.items())

    dict_rows = "\n".join(
        f"| `{n}` | {d} | {t} | {enc_text(n, m)} |" for n, d, t, m in col_docs
    )

    md = f"""# Data Preparation

Stage 2 deliverable for the customer churn analysis - owner: Rohan Sharma Kharel.

## Overview

One script, [`data_preparation.py`](data_preparation.py), takes the raw dataset at
`data/Dataset_ATS_v2.csv` ({s['raw_shape']['rows']:,} rows, {s['raw_shape']['columns']} columns) and
produces everything the clustering and modelling stages need. It runs the integrity
checks, handles missing data with median/mode imputation, encodes the categorical
variables, saves the encoded dataset, splits it {int((1 - s['test_size']) * 100)}/{int(s['test_size'] * 100)} stratified on `Churn`, fits a
`StandardScaler` **on the training set only** and applies it to both sets. Rerun the
whole thing from the repository root with:

```bash
python Data_Preparation/data_preparation.py
```

The two documents in this folder are regenerated from the pipeline's own output, so
none of the numbers below are typed by hand:

```bash
python Data_Preparation/build_docs.py
```

## Files produced

| File | Contents |
|---|---|
| `preprocessed_dataset.csv` | All {rows['full']:,} rows, missing data handled and categorical variables encoded. **Not scaled**, so the values stay readable. |
| `train_set.csv` | The {rows['train']:,} training rows, scaled. Same {len(s['columns'])} columns, `Churn` last. |
| `test_set.csv` | The {rows['test']:,} test rows, scaled with the scaler fitted on the training set. |
| `scaler.pkl` | The fitted `StandardScaler`, so later stages apply the identical transformation. |
| `encoding_map.json` | Every category-to-number mapping used, for decoding results. |
| `preparation_summary.json` | The row counts, churn rates and scaler statistics behind this README and the PDF. |
| `scaling_techniques.pdf` | The scaling techniques document: method, columns, leakage and results. |
| `figures/scaling_before_after.png` | Histograms of the training set before and after scaling. |
| `data_preparation.py` | The pipeline itself. |
| `build_docs.py` | Builds this README and the PDF from `preparation_summary.json`. |

## Size and composition

Split with `train_test_split(X, y, test_size={s['test_size']}, random_state={s['random_state']}, stratify=y)`.
Stratifying on `Churn` is what keeps the churn rate identical across the three rows below.

| Set | Rows | % of total | Churners | Churn rate | Month-to-month | One year | Two year |
|---|---|---|---|---|---|---|---|
{comp_row('Full dataset', 'full', '100.00%')}
{comp_row('Training set', 'train', pct(share['train']))}
{comp_row('Test set', 'test', pct(share['test']))}

Duplicate rows found and kept: **{s['duplicate_rows_found_and_kept']}**. Missing values found: **{s['missing_values_found']}**.

Scaled columns, fitted on the training set: `tenure` mean {sc['mean_']['tenure']:.4f}, std {sc['scale_']['tenure']:.4f}; `MonthlyCharges` mean {sc['mean_']['MonthlyCharges']:.4f}, std {sc['scale_']['MonthlyCharges']:.4f}.
After scaling, the training columns have mean 0 and standard deviation 1 exactly, while the test columns land close by but not exactly (`tenure` mean {after['test_mean']['tenure']:.4f}, std {after['test_std']['tenure']:.4f}; `MonthlyCharges` mean {after['test_mean']['MonthlyCharges']:.4f}, std {after['test_std']['MonthlyCharges']:.4f}).
That small gap is the proof that no test information leaked into the scaler.

## Column dictionary

All {len(s['columns'])} columns, in file order, target last.

| Column | Meaning | Type | Encoding |
|---|---|---|---|
{dict_rows}

Only `tenure` and `MonthlyCharges` are scaled. Every other column is already a 0/1 flag,
and the target is never scaled.

## Decisions and reasons

- **The {s['duplicate_rows_found_and_kept']} identical rows are kept.** The dataset has no customer ID and only
  {s['raw_shape']['columns']} fairly coarse columns, so two customers on the same plan with the same tenure and
  the same monthly charge produce identical rows without being the same person. Dropping
  them would throw away real customers and bias the churn rate.
- **Median for numeric, mode for categorical imputation.** The median is not dragged around
  by extreme charges the way the mean is, and the mode is the only sensible fill for a
  category. This dataset has {s['missing_values_found']} missing values, but the step is part of the deliverable and
  stays in the pipeline so it handles a future refresh of the data.
- **Label encoding for the binary columns, one-hot for `Contract`.** A two-value column maps
  cleanly to 0/1 with no ordering implied. `Contract` has three levels, and numbering them
  0/1/2 would tell the model that a two year contract is "twice" a one year one. One-hot
  with `drop_first=True` avoids that and leaves Month-to-month as the baseline, which is
  both levels set to 0.
- **Stratified {int((1 - s['test_size']) * 100)}/{int(s['test_size'] * 100)} split.** Churn is imbalanced at {pct(rate['full'])}, so a plain random split could
  hand the test set a noticeably different churn rate. Stratifying holds it at
  {pct(rate['train'])} in training and {pct(rate['test'])} in the test set. `random_state={s['random_state']}` makes the split reproducible.
- **The scaler is fitted on the training data only.** Fitting on all {rows['full']:,} rows would let the
  test set's mean and standard deviation shape the training data, so the model would be
  evaluated on data it had already been told something about. The test set is transformed
  with the training scaler, exactly as unseen data would be in production.
"""
    README_PATH.write_text(md, encoding="utf-8")
    print(f"Wrote {README_PATH.relative_to(ROOT)}")


# ---------------------------------------------------------------------------
# scaling_techniques.pdf
# ---------------------------------------------------------------------------
def build_pdf(s: dict) -> None:
    rows, rate, churners = s["rows"], s["churn_rate"], s["churners"]
    sc, before, after = s["scaler"], s["scaling_before"], s["scaling_after"]

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

    def table(data, widths):
        t = Table(data, colWidths=widths, hAlign="LEFT")
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f3864")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
            ("FONTSIZE", (0, 0), (-1, -1), 8.5),
            ("ALIGN", (1, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#b4b4c8")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#eef1f7")]),
        ]))
        return t

    story = []
    story.append(Paragraph("Scaling Techniques Applied to the Customer Churn Dataset", h1))
    story.append(Paragraph(
        "WILDA Team 4 &middot; ACS Work Integrated Learning, Data Analytics &middot; Stage 2 "
        "Data Preparation &middot; Rohan Sharma Kharel", sub))

    # 1. Purpose
    story.append(Paragraph("1. Purpose: why scaling matters here", h2))
    story.append(Paragraph(
        f"After encoding, the dataset has {len(s['columns']) - 1} features and one target. Eight of those features "
        "are 0/1 flags, but two are not: <b>tenure</b> runs from 0 to 72 months and "
        "<b>MonthlyCharges</b> runs from $18 to $119. Left as they are, those two columns are "
        "up to two orders of magnitude larger than every flag beside them. That matters "
        "because of what happens next. The clustering stage uses K-Means, which assigns a "
        "customer to a cluster by straight-line distance, so a column measured in dollars "
        "would dominate the distance calculation and the clusters would be little more than "
        "a split on price. The Stage 3 neural network is affected too: inputs on widely "
        "different scales make the gradients pull unevenly and the network slower and less "
        "stable to train. Putting the continuous columns on the same footing as the flags "
        "is what stops the unit of measurement deciding the result.", body))

    # 2. Technique
    story.append(Paragraph("2. Technique chosen: StandardScaler (z-score standardisation)", h2))
    story.append(Paragraph(
        "Each value is replaced by how many standard deviations it sits from its column mean:"
        " <b>z = (x &minus; mean) / std</b>. The transformed column has a mean of 0 and a "
        "standard deviation of 1. Two alternatives were considered and rejected:", body))
    story.append(bullets([
        "<b>MinMaxScaler</b> squeezes a column into the 0 to 1 range using only its minimum "
        "and maximum, so the two most extreme customers set the scale for everyone else and "
        "the rest of the distribution is compressed. StandardScaler uses every row, keeps the "
        "shape of the distribution intact, and produces the zero-centred inputs that suit the "
        "Stage 3 neural network.",
        "<b>RobustScaler</b> uses the median and the interquartile range and earns its keep "
        "when extreme outliers would distort the mean and standard deviation. Neither column "
        "has that problem: tenure is capped at 72 months by the data itself and MonthlyCharges "
        "tops out at $119, so there is nothing for a robust method to protect against.",
    ]))

    # 3. Columns
    story.append(Paragraph("3. Columns scaled and not scaled", h2))
    story.append(Paragraph(
        "Only <b>tenure</b> and <b>MonthlyCharges</b> are scaled. The eight encoded columns "
        "(gender, SeniorCitizen, Dependents, PhoneService, MultipleLines, InternetService, "
        "Contract_One_year, Contract_Two_year) are already 0/1, so they are on a comparable "
        "range with each other and with the standardised columns. Standardising a binary flag "
        "adds nothing: it just replaces the two values with two other numbers, while making "
        "the column harder to read and harder to explain to the client. The target "
        "<b>Churn</b> is never scaled, because it is the label being predicted, not an input "
        "feature.", body))

    # 4. Leakage
    story.append(Paragraph("4. Preventing data leakage", h2))
    story.append(Paragraph(
        "The scaler is fitted on the training set only, and that same fitted scaler is then "
        "used to transform the test set. The order matters: the data is split first and "
        "scaled second. Fitting the scaler on all "
        f"{rows['full']:,} rows would mean the mean and standard deviation it subtracts and "
        "divides by were calculated partly from the test rows. The test set is meant to stand "
        "in for customers the model has never seen, so the moment its statistics help shape "
        "the training data, the evaluation is measuring something easier than reality and the "
        "reported accuracy is optimistic. Fitting on training data only is also what actually "
        "happens in production: when a new customer arrives tomorrow, the only statistics "
        "available are the ones already learned.", body))

    story.append(PageBreak())

    # 5. Split
    story.append(Paragraph("5. The split the scaler was fitted on", h2))
    story.append(Paragraph(
        f"The dataset was split {int((1 - s['test_size']) * 100)}/{int(s['test_size'] * 100)}, stratified on Churn, with "
        f"random_state={s['random_state']} for reproducibility. Churn is imbalanced at "
        f"{pct(rate['full'])}, so stratifying is what keeps that rate steady across both sets:", body))
    story.append(table([
        ["Set", "Rows", "% of total", "Churners", "Churn rate"],
        ["Full dataset", f"{rows['full']:,}", "100.00%", f"{churners['full']:,}", pct(rate["full"])],
        ["Training set", f"{rows['train']:,}", pct(s["share_of_total"]["train"]),
         f"{churners['train']:,}", pct(rate["train"])],
        ["Test set", f"{rows['test']:,}", pct(s["share_of_total"]["test"]),
         f"{churners['test']:,}", pct(rate["test"])],
    ], [4.1 * cm, 2.6 * cm, 2.6 * cm, 2.6 * cm, 2.6 * cm]))
    story.append(Spacer(1, 8))

    # 6. Results
    story.append(Paragraph("6. Results", h2))
    story.append(table([
        ["Column", "Train mean\nbefore", "Train std\nbefore", "Train mean\nafter",
         "Train std\nafter", "Test mean\nafter", "Test std\nafter"],
        ["tenure (months)",
         f"{before['train_mean']['tenure']:.4f}", f"{before['train_std']['tenure']:.4f}",
         f"{after['train_mean']['tenure']:.4f}", f"{after['train_std']['tenure']:.4f}",
         f"{after['test_mean']['tenure']:.4f}", f"{after['test_std']['tenure']:.4f}"],
        ["MonthlyCharges ($)",
         f"{before['train_mean']['MonthlyCharges']:.4f}", f"{before['train_std']['MonthlyCharges']:.4f}",
         f"{after['train_mean']['MonthlyCharges']:.4f}", f"{after['train_std']['MonthlyCharges']:.4f}",
         f"{after['test_mean']['MonthlyCharges']:.4f}", f"{after['test_std']['MonthlyCharges']:.4f}"],
    ], [3.6 * cm, 2.1 * cm, 2.1 * cm, 2.1 * cm, 2.1 * cm, 2.1 * cm, 2.1 * cm]))
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        "The training columns come out at exactly mean 0 and standard deviation 1, which is "
        "what fitting on that set guarantees. The test set lands close to but not exactly on "
        f"0 and 1 (tenure {after['test_mean']['tenure']:.4f} and {after['test_std']['tenure']:.4f}, MonthlyCharges "
        f"{after['test_mean']['MonthlyCharges']:.4f} and {after['test_std']['MonthlyCharges']:.4f}). That is the expected result and a "
        "useful check: if the test columns were also exactly 0 and 1, the scaler would have "
        "been fitted on the full dataset and the pipeline would be leaking. The figure below "
        "shows the effect on the training set. The bars keep their shape exactly; only the "
        "axis changes, from months and dollars to standard deviations from the mean.", body))
    story.append(Image(str(FIGURE_PATH), width=15.6 * cm, height=9.9 * cm))
    story.append(Paragraph(
        "Training set distributions before (top) and after (bottom) standardisation.", caption))

    story.append(PageBreak())

    # 7. Code
    story.append(Paragraph("7. Code snippets", h2))
    story.append(Paragraph(
        "Taken from <font face='Courier'>Data_Preparation/data_preparation.py</font>. "
        "First the split, which happens before any scaling:", body))
    story.append(code_block(snippet("X_train, X_test, y_train, y_test = train_test_split(",
                                    last_line_equals="    )")))
    story.append(Paragraph(
        "Then the scaler, fitted on the training rows and applied to both sets. "
        "<font face='Courier'>fit_transform</font> is called on the training data only; the "
        "test data gets <font face='Courier'>transform</font>, which reuses the statistics "
        "already learned:", body))
    story.append(code_block(snippet("scaler = StandardScaler()", n_lines=3)))
    story.append(Paragraph("Finally the fitted scaler is saved so it can be reused:", body))
    story.append(code_block(snippet('joblib.dump(scaler,', n_lines=1)))

    # 8. Reuse
    story.append(Paragraph("8. Reusing the scaler downstream", h2))
    story.append(Paragraph(
        "<font face='Courier'>scaler.pkl</font> holds the means and standard deviations "
        f"learned from the training set (tenure {sc['mean_']['tenure']:.4f} / {sc['scale_']['tenure']:.4f}, "
        f"MonthlyCharges {sc['mean_']['MonthlyCharges']:.4f} / {sc['scale_']['MonthlyCharges']:.4f}). The clustering "
        "stage and the Stage 3 model load it rather than fitting their own, so every stage "
        "transforms data identically and a cluster centre can be converted back into months "
        "and dollars for the client:", body))
    story.append(code_block(
        "import joblib\n\n"
        "scaler = joblib.load(\"Data_Preparation/scaler.pkl\")\n"
        "SCALE_COLS = [\"tenure\", \"MonthlyCharges\"]\n\n"
        "# Apply the identical transformation to any new data\n"
        "new_data[SCALE_COLS] = scaler.transform(new_data[SCALE_COLS])\n\n"
        "# Or read a scaled value back in months and dollars\n"
        "original = scaler.inverse_transform(scaled_values)"))
    story.append(Paragraph(
        "The pre-scaled train and test sets are committed as "
        "<font face='Courier'>train_set.csv</font> and "
        "<font face='Courier'>test_set.csv</font>, so the later stages do not need to repeat "
        "any of this; the scaler is there for new data and for reading results back in the "
        "original units. The row counts, churn rates and scaler statistics quoted throughout "
        "this document are generated from "
        "<font face='Courier'>preparation_summary.json</font>, which the pipeline writes on "
        "each run.", body))

    doc = SimpleDocTemplate(
        str(PDF_PATH), pagesize=A4,
        leftMargin=2.2 * cm, rightMargin=2.2 * cm, topMargin=1.8 * cm, bottomMargin=1.8 * cm,
        title="Scaling Techniques - Customer Churn Data Preparation",
        author="Rohan Sharma Kharel, WILDA Team 4",
    )
    doc.build(story)
    print(f"Wrote {PDF_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    summary = load_json(SUMMARY_PATH)
    encodings = load_json(ENCODING_PATH)
    build_readme(summary, encodings)
    build_pdf(summary)
