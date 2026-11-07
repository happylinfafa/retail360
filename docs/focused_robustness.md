# Additional evidence for the focused 60-day study

This is an **exploratory temporal sensitivity analysis**, not independent external
confirmation. The project question was focused after earlier graph-model test
results were viewed. The original September 2011 test result remains visible
and is not retrospectively relabeled as untouched. Only a new retailer or a
truly untouched later period could provide stronger confirmation.

## Frozen evaluation design

- Source: the same complete UCI Online Retail II workbook and cleaning rules.
- Unit: an existing customer at a prediction cutoff. Prior 180 days provide
  features; the following 60 days provide the label.
- The models are fixed: RFM logistic regression trained on the original three
  training snapshots, and the original validation-selected all-history logistic
  model. No model is selected or refitted using the August, September, or
  October outcomes.
- Additional cutoffs: August 1 and October 1, 2011. Their 60-day label windows
  do not overlap. The earlier September cutoff is retained for comparison;
  customer identities and histories may overlap between windows.
- Primary comparison: all-history minus RFM average precision (AP) at each
  cutoff. A paired bootstrap resamples customer rows 2,000 times (seed 5588)
  within each cutoff and reports percentile 95% intervals. These intervals
  account for customer sampling within a cutoff, **not** retailer, time-window,
  or prior research-selection uncertainty.
- Calibration: a logistic (Platt) mapping is fit to the selected model's
  predicted log-odds on the June 2011 validation snapshot. It is applied
  unchanged to the later cutoffs. Brier score measures probability error.
  Validation was also used for model selection, so this is exploratory.

| Cutoff | Customers | RFM AP | Full AP | Difference (95% paired bootstrap interval) | Brier before → after calibration |
|---|---:|---:|---:|---:|---:|
| 2011-08-01 | 2,771 | 0.7254 | 0.7328 | +0.0074 (0.0022, 0.0128) | 0.2058 → 0.1983 |
| 2011-09-01 | 2,772 | 0.7502 | 0.7551 | +0.0049 (0.0006, 0.0094) | 0.2219 → 0.2091 |
| 2011-10-01 | 2,925 | 0.7555 | 0.7601 | +0.0046 (0.0002, 0.0088) | 0.2353 → 0.2189 |

The small AP difference is positive at these three cutoffs, but overlapping
customers and the earlier choice of project scope limit confirmatory claims.
Calibration improves Brier score in all three windows, yet the adjusted
probabilities still underestimate the observed rate in the five September
score groups. Use scores for historical ranking and human review, not as exact
individual odds or a causal estimate of marketing impact.

The public Streamlit application contains **10 precomputed real customer
examples** with worksheet row IDs and invoices. It does not score new customers
live. The research metrics use all 2,772 customers at the September test
cutoff. Full customer-level predictions and fitted models remain local.

## Reproduce

After preparing the frozen data and training the focused model as described in
the [README](../README.md), run:

```sh
python scripts/evaluate_focused_robustness.py /path/to/online_retail_II.xlsx
```

The script verifies that it exactly reproduces the published September AP
before writing aggregate files in `results/repurchase_focused/`. It never
publishes full customer-level predictions or the source workbook.
