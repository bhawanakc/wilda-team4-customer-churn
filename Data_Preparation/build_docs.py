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
            f"{path.name} not found. Run 'python Data_Preparation/data_preparation.py' first."
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
        ("Churn", "Target. Whether the customer left", "int (0/1)", enc["Churn"]),
    ]

    def enc_text(name: str, mapping) -> str:
        if mapping is None:
            return ("original units in `preprocessed_dataset.csv`, "
                    "z-scores in the train and test sets")
        return ", ".join(f"{k} = {v}" for k, v in mapping.items())

    dict_rows = "\n".join(
        f"| `{n}` | {d} | {t} | {enc_text(n, m)} |" for n, d, t, m in col_docs
    )

    md = f"""# Data preparation

Stage 2 data preparation for the customer churn analysis. Owner: Rohan Sharma Kharel.

## Overview

[`data_preparation.py`](data_preparation.py) reads the raw dataset at
`data/Dataset_ATS_v2.csv` ({s['raw_shape']['rows']:,} rows, {s['raw_shape']['columns']} columns) and writes every file the
clustering and modelling stages need. The script runs in this order:

1. Check the data for duplicates, stray whitespace and invalid values.
2. Fill any missing values, using the median for numbers and the mode for categories.
3. Encode the categorical columns and save the encoded dataset.
4. Split the rows {int((1 - s['test_size']) * 100)}/{int(s['test_size'] * 100)} into training and test sets, stratified on `Churn`.
5. Fit a `StandardScaler` on the training set only, then use it to scale both sets.

To rebuild every output, run this from the repository root:

```bash
python Data_Preparation/data_preparation.py
```

[`build_docs.py`](build_docs.py) writes this README and the PDF from the numbers the
pipeline saves, so nobody types the figures by hand:

```bash
python Data_Preparation/build_docs.py
```

## Files produced

| File | Contents |
|---|---|
| `preprocessed_dataset.csv` | All {rows['full']:,} rows with missing data handled and categories encoded. Not scaled, so tenure is in months and charges are in dollars. |
| `train_set.csv` | The {rows['train']:,} training rows, scaled. Same {len(s['columns'])} columns, with `Churn` last. |
| `test_set.csv` | The {rows['test']:,} test rows, scaled with the scaler fitted on the training set. |
| `scaler.pkl` | The fitted `StandardScaler`. Later stages load it to scale new data the same way. |
| `encoding_map.json` | The number each category was mapped to. |
| `preparation_summary.json` | The row counts, churn rates and scaler statistics used in this README and the PDF. |
| `scaling_techniques.pdf` | Which scaler we used and why, which columns it scaled, how we avoided data leakage, and the results. |
| `figures/scaling_before_after.png` | Histograms of the training set before and after scaling. |
| `data_preparation.py` | The pipeline. |
| `build_docs.py` | Builds this README and the PDF from `preparation_summary.json`. |

## Size and composition

We split the data with `train_test_split(X, y, test_size={s['test_size']}, random_state={s['random_state']}, stratify=y)`.
Stratifying on `Churn` gives all three rows below the same churn rate.

| Set | Rows | % of total | Churners | Churn rate | Month-to-month | One year | Two year |
|---|---|---|---|---|---|---|---|
{comp_row('Full dataset', 'full', '100.00%')}
{comp_row('Training set', 'train', pct(share['train']))}
{comp_row('Test set', 'test', pct(share['test']))}

The raw data has {s['duplicate_rows_found_and_kept']} duplicate rows, and the pipeline keeps them. It has {s['missing_values_found']} missing values.

The scaler learned these values from the training set:

| Column | Mean | Standard deviation |
|---|---|---|
| `tenure` | {sc['mean_']['tenure']:.4f} | {sc['scale_']['tenure']:.4f} |
| `MonthlyCharges` | {sc['mean_']['MonthlyCharges']:.4f} | {sc['scale_']['MonthlyCharges']:.4f} |

After scaling, both training columns have a mean of 0 and a standard deviation of 1.
The test columns come out close to those values but not equal to them. `tenure` has
mean {after['test_mean']['tenure']:.4f} and standard deviation {after['test_std']['tenure']:.4f}, and `MonthlyCharges` has
mean {after['test_mean']['MonthlyCharges']:.4f} and standard deviation {after['test_std']['MonthlyCharges']:.4f}. The small gap shows the scaler
never saw the test rows.

## Column dictionary

The {len(s['columns'])} columns, in file order, with the target last.

| Column | Meaning | Type | Encoding |
|---|---|---|---|
{dict_rows}

The pipeline scales only `tenure` and `MonthlyCharges`. Every other feature is a 0/1
flag, and the target stays as 0/1.

## Decisions and reasons

- **We kept the {s['duplicate_rows_found_and_kept']} duplicate rows.** The dataset has no customer ID and only
  {s['raw_shape']['columns']} columns, most with two or three possible values. Two different customers on the
  same plan, with the same tenure and the same monthly charge, produce identical rows.
  Dropping the duplicates would remove {s['duplicate_rows_found_and_kept']} real customers from the analysis.
- **We fill missing numbers with the median and missing categories with the mode.** Extreme
  values move the mean more than the median. The mode is the most common category. This dataset has {s['missing_values_found']} missing values, so the step changes
  nothing today. It stays in the pipeline because the assignment requires it and a later
  version of the data may have gaps.
- **We label encode the two-value columns and one-hot encode `Contract`.** A column with two
  values maps to 0 and 1 without implying any order. `Contract` has three values. Numbering
  them 0, 1 and 2 would tell the model that a two year contract is twice a one year
  contract. One-hot encoding with `drop_first=True` creates two 0/1 columns instead. A
  month-to-month customer has 0 in both.
- **We stratified the {int((1 - s['test_size']) * 100)}/{int(s['test_size'] * 100)} split on `Churn`.** Only {pct(rate['full'])} of customers churned,
  so a plain random split could give the test set a different churn rate. Stratifying gives
  {pct(rate['train'])} in the training set and {pct(rate['test'])} in the test set. `random_state={s['random_state']}`
  makes the split the same on every run.
- **We fitted the scaler on the training set only.** If we fitted it on all {rows['full']:,} rows,
  the test set's mean and standard deviation would change how the training data is scaled.
  The test results would then look better than the model would do on new customers. The
  pipeline scales the test set with the training scaler, the same way it would scale new
  customer data.
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
    story.append(Paragraph("Scaling techniques applied to the customer churn dataset", h1))
    story.append(Paragraph(
        "WILDA Team 4 &middot; ACS Work Integrated Learning, Data Analytics &middot; Stage 2 "
        "Data Preparation &middot; Rohan Sharma Kharel", sub))

    # 1. Purpose
    story.append(Paragraph("1. Why the data needs scaling", h2))
    story.append(Paragraph(
        f"After encoding, the dataset has {len(s['columns']) - 1} features and one target. Eight of the features "
        "are 0/1 flags. The other two are much larger. <b>tenure</b> runs from 0 to 72 months, "
        "and <b>MonthlyCharges</b> runs from $18 to $119.", body))
    story.append(Paragraph(
        "K-Means is used in the clustering stage, which puts each customer in the cluster with the "
        "nearest centre. Unscaled, a $50 difference in monthly charge counts 50 times as much "
        "as the difference between having and not having phone service. The clusters would "
        "mostly split customers by price. The Stage 3 neural network also trains more slowly "
        "and less reliably when its inputs have very different ranges. Scaling puts tenure "
        "and MonthlyCharges on a range close to the 0/1 flags, so no feature counts more just "
        "because of its units.", body))

    # 2. Technique
    story.append(Paragraph("2. Technique chosen: StandardScaler", h2))
    story.append(Paragraph(
        "StandardScaler replaces each value with its distance from the column mean, measured "
        "in standard deviations: <b>z = (x &minus; mean) / std</b>. Each scaled column has a "
        "mean of 0 and a standard deviation of 1. We compared it with two other scalers:", body))
    story.append(bullets([
        "<b>MinMaxScaler</b> maps each column to the range 0 to 1 using its minimum and "
        "maximum, so the single most extreme customer at each end sets the scale for everyone. "
        "StandardScaler uses the mean and standard deviation of every row, and it centres each "
        "column on 0, which suits the Stage 3 neural network.",
        "<b>RobustScaler</b> uses the median and the interquartile range, which helps when a "
        "few extreme outliers distort the mean and standard deviation. These two columns have "
        "no such outliers. Tenure never goes above 72 months and MonthlyCharges never goes "
        "above $119.",
    ]))

    # 3. Columns
    story.append(Paragraph("3. Columns scaled and not scaled", h2))
    story.append(Paragraph(
        "The pipeline scales only <b>tenure</b> and <b>MonthlyCharges</b>. The eight encoded "
        "columns (gender, SeniorCitizen, Dependents, PhoneService, MultipleLines, "
        "InternetService, Contract_One_year and Contract_Two_year) already hold 0 or 1, which "
        "is close to the range of the scaled columns. Scaling a 0/1 column would swap its two "
        "values for two other numbers. The model would learn nothing new, and the column would "
        "be harder to read. The pipeline never scales the target, <b>Churn</b>, because it is "
        "the value the model predicts rather than an input.", body))

    # 4. Leakage
    story.append(Paragraph("4. Preventing data leakage", h2))
    story.append(Paragraph(
        "The pipeline splits the data first and scales it second. It fits the scaler on the "
        "training set only, then uses that fitted scaler to transform the test set.", body))
    story.append(Paragraph(
        f"If we fitted the scaler on all {rows['full']:,} rows, the mean and standard deviation "
        "it uses would come partly from the test rows. The test set stands in for customers "
        "the model has never seen. Once its statistics shape the training data, the test "
        "results look better than the model would do on real new customers. Fitting on "
        "the training set only also matches how the model will be used. When a new customer "
        "joins, the only statistics available are the ones learned from the training data.",
        body))

    story.append(PageBreak())

    # 5. Split
    story.append(Paragraph("5. Training and test split", h2))
    story.append(Paragraph(
        f"We split the dataset {int((1 - s['test_size']) * 100)}/{int(s['test_size'] * 100)}, stratified on Churn, with "
        f"random_state={s['random_state']} so the split is the same on every run. Only "
        f"{pct(rate['full'])} of customers churned, and stratifying keeps that rate the same "
        "in both sets.", body))
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
        "Both training columns have a mean of 0 and a standard deviation of 1, because the "
        "scaler learned its statistics from that set. The test columns come out close to "
        f"those values but not equal to them. Tenure has mean {after['test_mean']['tenure']:.4f} and standard "
        f"deviation {after['test_std']['tenure']:.4f}, and MonthlyCharges has mean "
        f"{after['test_mean']['MonthlyCharges']:.4f} and standard deviation {after['test_std']['MonthlyCharges']:.4f}.", body))
    story.append(Paragraph(
        "This gap is expected, and it is a useful check. If the test columns also came out at "
        "exactly 0 and 1, the scaler would have learned from the test rows.", body))
    story.append(Paragraph(
        "The figure below shows the training set before and after scaling. The bars have the "
        "same shape in both rows. Only the axis changes, from months and dollars to standard "
        "deviations from the mean.", body))
    story.append(Image(str(FIGURE_PATH), width=15.6 * cm, height=9.9 * cm))
    story.append(Paragraph(
        "Training set distributions before (top) and after (bottom) standardisation.", caption))

    story.append(PageBreak())

    # 7. Code
    story.append(Paragraph("7. Code snippets", h2))
    story.append(Paragraph(
        "These snippets come from <font face='Courier'>Data_Preparation/data_preparation.py</font>. "
        "The pipeline splits the data before it scales anything:", body))
    story.append(code_block(snippet("X_train, X_test, y_train, y_test = train_test_split(",
                                    last_line_equals="    )")))
    story.append(Paragraph(
        "It then calls <font face='Courier'>fit_transform</font> on the training rows only. "
        "The test rows go through <font face='Courier'>transform</font>, which reuses the "
        "mean and standard deviation learned from the training rows:", body))
    story.append(code_block(snippet("scaler = StandardScaler()", n_lines=3)))
    story.append(Paragraph("Finally, it saves the fitted scaler:", body))
    story.append(code_block(snippet('joblib.dump(scaler,', n_lines=1)))

    # 8. Reuse
    story.append(Paragraph("8. Reusing the scaler in later stages", h2))
    story.append(Paragraph(
        "<font face='Courier'>scaler.pkl</font> stores the mean and standard deviation "
        "learned from the training set. For tenure these are "
        f"{sc['mean_']['tenure']:.4f} and {sc['scale_']['tenure']:.4f}, and for MonthlyCharges "
        f"{sc['mean_']['MonthlyCharges']:.4f} and {sc['scale_']['MonthlyCharges']:.4f}.", body))
    story.append(Paragraph(
        "The clustering stage and the Stage 3 model load this file instead of fitting their "
        "own scaler, so every stage scales data the same way. The same file converts a "
        "cluster centre back into months and dollars for the client:", body))
    story.append(code_block(
        "import joblib\n\n"
        "scaler = joblib.load(\"Data_Preparation/scaler.pkl\")\n"
        "SCALE_COLS = [\"tenure\", \"MonthlyCharges\"]\n\n"
        "# Scale new data the same way as the training data\n"
        "new_data[SCALE_COLS] = scaler.transform(new_data[SCALE_COLS])\n\n"
        "# Or read a scaled value back in months and dollars\n"
        "original = scaler.inverse_transform(scaled_values)"))
    story.append(Paragraph(
        "The repository already contains the scaled sets as "
        "<font face='Courier'>train_set.csv</font> and "
        "<font face='Courier'>test_set.csv</font>, so later stages can use them directly. "
        "They need the scaler only for new data and for converting results back to months "
        "and dollars.", body))
    story.append(Paragraph(
        "The pipeline writes <font face='Courier'>preparation_summary.json</font> on each run, "
        "and every row count, churn rate and scaler statistic in this document comes from "
        "that file.", body))

    doc = SimpleDocTemplate(
        str(PDF_PATH), pagesize=A4,
        leftMargin=2.2 * cm, rightMargin=2.2 * cm, topMargin=1.8 * cm, bottomMargin=1.8 * cm,
        title="Scaling techniques, customer churn data preparation",
        author="Rohan Sharma Kharel, WILDA Team 4",
    )
    doc.build(story)
    print(f"Wrote {PDF_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    summary = load_json(SUMMARY_PATH)
    encodings = load_json(ENCODING_PATH)
    build_readme(summary, encodings)
    build_pdf(summary)
