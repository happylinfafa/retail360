# Retail360: 60-day repeat-purchase prediction

## Research question and user

Can a retailer's historical transactions predict whether an **existing** customer
will make another valid purchase within the next 60 days? The intended user is
a retail analyst who needs to inspect historical predictions and their source
orders. This is a predictive backtest, not an estimate of what a marketing
intervention would cause.

## Data and outcome

The source is the public [UCI Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii)
workbook (1,067,371 raw transaction rows; Chen, DOI 10.24432/C5CG6D).
The frozen [data protocol](repurchase_protocol.md) excludes cancellations,
missing identifiers, invalid dates, and nonpositive quantity or price. An
eligible customer has at least one valid invoice in the 180 days preceding a
prediction cutoff. A label of 1 means at least one valid purchase in the next
60 days. All feature windows end **before** the cutoff; future rows only supply
evaluation labels. The data cannot reveal purchases outside this retailer.

| Partition | Historical prediction cutoffs | Customer-snapshot rows |
|---|---|---:|
| Train | 2010-09-01, 2010-12-01, 2011-03-01 | 9,659 |
| Validation | 2011-06-01 | 2,659 |
| Test | 2011-09-01 | 2,772 |

The same customer may appear at different cutoffs. We do not randomly split
customer rows; the final evaluation is later in time than model fitting.

## Methods

The first feature family is RFM: recency in days, distinct invoice count, and
positive historical spend. The next adds average order value, last-30-day
orders, active purchase days, and mean invoice gap. The third adds distinct
product count. No customer ID, future row, graph feature, or LLM output is a
predictor in this focused study. Imputation and transformations are fitted on
training data only.

The comparison contains a prior-rate dummy classifier, logistic regression,
and histogram gradient boosting. We choose the non-dummy candidate with the
highest **validation average precision (AP)**. Thresholds are selected using
validation F1. The test snapshot is loaded only after selection is saved. The
selected model is **all-history logistic regression**. Because this project
scope was simplified after earlier graph-model test results had been inspected,
the focused comparison should be presented as exploratory rather than as a
fresh untouched confirmatory study.

## Observed results

| Method | Validation AP | Test AP | Test Brier score |
|---|---:|---:|---:|
| Overall repeat-purchase rate | 0.4250 | 0.4906 | 0.2638 |
| RFM logistic regression | 0.7153 | 0.7502 | 0.2242 |
| RFM + temporal logistic regression | 0.7209 | 0.7532 | 0.2224 |
| All-history logistic regression (selected within focused candidates) | 0.7227 | **0.7551** | **0.2219** |
| All-history boosted tree | 0.7210 | 0.7507 | 0.2220 |

AP is non-interpolated average precision, not accuracy. The test cohort's
observed repeat-purchase rate is 49.06%. Among the top-ranked 10% under the
selected focused model, 92.81% actually purchased again (1.89 times the
overall rate). This is ranking evidence, **not** a marketing treatment effect.

The probability reliability check exposes a weakness: every one of five
equal-size test groups has a higher observed repeat-purchase rate than its
mean predicted probability. For example, the highest-score group averages
0.712 predicted versus 0.823 observed. The model is more defensible as a
historical ranking tool than as a source of calibrated individual odds on
this test date. The application warns users accordingly.

One-invoice customers have a 32.7% observed repeat rate in the test snapshot;
customers with five or more invoices have 89.1%. These are group descriptions,
not causal effects. The full aggregate evidence is in
[`../results/repurchase_focused/`](../results/repurchase_focused/).

## Working application and provenance

The Streamlit **Repurchase** page displays model comparisons, probability
reliability, customer-group differences, and an individual historical review.
The complete local run allows review of all 2,772 test customers and their
pre-cutoff source transaction rows. For cloud-friendly review, the repository
ships 10 precomputed real customer examples with original invoice and worksheet
row identifiers. They are a **display sample**, not the evaluation cohort, and
the public page does not train or score a new customer live. The human may
inspect evidence, reveal the held-out outcome, accept or reject the result,
and download a review note.

The earlier descriptive AI explorer and Challenge 3–4 graph experiments remain
in the repository as previous exploratory work. Their graph/GraphRAG/community
results do **not** improve or explain this focused prediction model. A prior
single graph-feature ablation did not show held-out improvement; see
[`model_results.md`](model_results.md). It is not silently omitted from the
project history.

## Reproduce

After downloading the complete official workbook and installing
`requirements.txt` plus scikit-learn/joblib, run:

```sh
python scripts/prepare_repurchase.py /path/to/online_retail_II.xlsx
python scripts/train_focused_repurchase.py
streamlit run app.py
```

The first command creates `data/repurchase_v1/`; the second creates local
`runtime/repurchase_focused_v1/` and the small tracked
`results/repurchase_focused/` display package. The preparation and training
scripts refuse to overwrite nonempty local outputs. Model artifacts and the
full customer-level test table remain local and are excluded from Git.

## Limitations and next experiment

The study has one retailer, one final test date, repeated customers across
time, no refund reconciliation, and incomplete information about purchases
outside this retailer. Probability underestimation requires a calibration
study using data prior to the final test date. The incremental AP gain from
RFM to richer features is small and has not been established as stable across
independent time windows. Future evaluation should repeat chronological
cutoffs and assess calibration and ranking separately. No causal, live
marketing, or current-day forecasting claim follows from this study.
