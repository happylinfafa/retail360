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

## Fixed review capacity and cohort dependence

To assess practical ranking value, both fixed models select the highest-scoring
10% of eligible customers at each cutoff. The outcome count is the number of
these selected customers who actually repurchased during the next 60 days.
The models use the same capacity and labels; no marketing intervention occurs.

| Cutoff | Review capacity | Repeat buyers found by RFM | Repeat buyers found by full model | Net difference |
|---|---:|---:|---:|---:|
| 2011-08-01 | 278 | 247 | 248 | +1 |
| 2011-09-01 | 278 | 256 | 258 | +2 |
| 2011-10-01 | 293 | 268 | 269 | +1 |

At the September cutoff, the full model selects 33 customers outside the RFM
top 278, but finds only **two additional actual repeat buyers** overall. Thus
the small AP advantage yields a small observed gain at this review capacity.
This is a historical prioritization comparison, not evidence that contacting
the selected customers would increase sales. The exact customer selections and
per-cutoff counts can be reproduced with `decision_utility.csv`.

The conclusion changes with the review capacity. Using the same fixed models
and September labels, the full model finds no additional buyers when reviewing
the top 5%, two more at 10%, and one fewer at 20%:

| September capacity | Customers reviewed | RFM buyers found | Full-model buyers found | Difference |
|---|---:|---:|---:|---:|
| Top 5% | 139 | 134 | 134 | 0 |
| Top 10% | 278 | 256 | 258 | +2 |
| Top 20% | 555 | 458 | 457 | -1 |

The same comparison for all three cutoffs is in `capacity_sensitivity.csv`.
These retrospective counts measure ranking at a chosen capacity, not the
incremental sales that a marketing intervention would cause.

The three windows do not provide three independent replications. August and
October share **2,395 customers**, or **86.4% of the August cohort**, even
though their future 60-day label windows do not overlap. The pairwise counts
are in `cohort_overlap.csv`. The per-window bootstrap intervals do not account
for this cross-window dependence or the earlier post hoc choice of project
scope. Stronger confirmation requires a newly specified, previously unseen
period or an external retailer with the same outcome definition.

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
