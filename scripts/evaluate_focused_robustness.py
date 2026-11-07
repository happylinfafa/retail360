"""Exploratory temporal sensitivity and pre-test calibration for the frozen focused study.

The Sep 2011 test and earlier graph tests were already viewed. Aug/Oct windows
add sensitivity evidence; they are not an independent external confirmation.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss

from modeling import RFM, feature_input, make_model
from repurchase import ROOT, build_snapshot, load_protocol, read_workbook
from train_focused_repurchase import load_split


OUT = ROOT / 'results' / 'repurchase_focused'
DATA = ROOT / 'data' / 'repurchase_v1'
MODEL = ROOT / 'runtime' / 'repurchase_focused_v1' / 'model.joblib'
SEED = 5588


def paired_ap_interval(y, basic, full, repeats=2000):
    """Resample customer rows together, so both models see the same cases."""
    rng = np.random.default_rng(SEED)
    indices = np.arange(len(y))
    differences = []
    for _ in range(repeats):
        sampled = rng.choice(indices, size=len(indices), replace=True)
        if np.unique(y[sampled]).size != 2:
            continue
        differences.append(average_precision_score(y[sampled], full[sampled]) -
                           average_precision_score(y[sampled], basic[sampled]))
    return np.quantile(differences, [.025, .5, .975]).tolist()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('workbook', type=Path,
                        help='Path to the complete official UCI Online Retail II workbook')
    args = parser.parse_args()
    protocol = load_protocol()
    payload = joblib.load(MODEL)
    if payload['selection']['selected_model'] != 'all_history_logistic':
        raise ValueError('Frozen selected model changed.')
    train = load_split(DATA, protocol, 'train')
    validation = load_split(DATA, protocol, 'validation')
    rfm = make_model('logistic')
    rfm.fit(feature_input(train, RFM), train.repurchase_60d)
    full = payload['model']
    full_columns = payload['columns']

    # Platt calibration is frozen before any of these evaluation-window labels
    # are used. The validation set also selected the model, so this remains
    # exploratory and must not be described as pristine confirmation.
    validation_scores = full.predict_proba(feature_input(validation, full_columns))[:, 1]
    logit = lambda x: np.log(np.clip(x, 1e-6, 1-1e-6) /
                             (1-np.clip(x, 1e-6, 1-1e-6))).reshape(-1, 1)
    calibrator = LogisticRegression(solver='lbfgs', random_state=SEED)
    calibrator.fit(logit(validation_scores), validation.repurchase_60d)

    transactions, source_audit = read_workbook(args.workbook, protocol['source_sha256'])
    rows = []
    decision_rows = []
    cohorts = {}
    september = None
    for cutoff in ['2011-08-01', '2011-09-01', '2011-10-01']:
        features, labels, _, _, _, audit = build_snapshot(transactions, cutoff, protocol)
        keys = ['cutoff', 'CustomerID']
        frame = features.merge(labels[keys + ['repurchase_60d']], on=keys, validate='one_to_one')
        y = frame.repurchase_60d.to_numpy(dtype=int)
        basic = rfm.predict_proba(feature_input(frame, RFM))[:, 1]
        advanced = full.predict_proba(feature_input(frame, full_columns))[:, 1]
        calibrated = calibrator.predict_proba(logit(advanced))[:, 1]
        cohorts[cutoff] = set(frame.CustomerID.astype(str))
        if cutoff == '2011-09-01':
            recorded = pd.read_csv(OUT/'metrics.csv')
            reference = float(recorded.query("model == 'all_history_logistic' and split == 'test'").average_precision.iloc[0])
            if abs(average_precision_score(y, advanced) - reference) > 1e-10:
                raise ValueError('September recreation does not match frozen published AP.')
            september = pd.DataFrame({'CustomerID': frame.CustomerID, 'y': y,
                                      'rfm': basic, 'full': advanced, 'calibrated': calibrated})
        ci = paired_ap_interval(y, basic, advanced)
        top = np.argsort(-advanced, kind='stable')[:int(np.ceil(.1*len(y)))]
        basic_top = np.argsort(-basic, kind='stable')[:len(top)]
        full_ids = set(frame.iloc[top].CustomerID.astype(str))
        basic_ids = set(frame.iloc[basic_top].CustomerID.astype(str))
        basic_hits = int(y[basic_top].sum())
        full_hits = int(y[top].sum())
        decision_rows.append({
            'cutoff': cutoff, 'eligible_customers': len(y), 'review_capacity': len(top),
            'rfm_repeat_buyers_found': basic_hits,
            'full_repeat_buyers_found': full_hits,
            'incremental_repeat_buyers_found': full_hits-basic_hits,
            'rfm_precision_at_capacity': float(y[basic_top].mean()),
            'full_precision_at_capacity': float(y[top].mean()),
            'customers_selected_by_both': len(full_ids & basic_ids),
            'customers_selected_only_by_full': len(full_ids-basic_ids),
            'scope': 'Historical fixed-capacity ranking; no intervention or causal lift measured.'
        })
        rows.append({
            'cutoff': cutoff, 'n': len(y), 'positive_rate': float(y.mean()),
            'history_start': audit['history_start_inclusive'],
            'label_end_exclusive': audit['label_end_exclusive'],
            'rfm_ap': float(average_precision_score(y, basic)),
            'all_history_ap': float(average_precision_score(y, advanced)),
            'ap_difference': float(average_precision_score(y, advanced) - average_precision_score(y, basic)),
            'ap_difference_ci_low': float(ci[0]), 'ap_difference_ci_high': float(ci[2]),
            'raw_brier': float(brier_score_loss(y, advanced)),
            'calibrated_brier': float(brier_score_loss(y, calibrated)),
            'raw_mean_probability': float(advanced.mean()),
            'calibrated_mean_probability': float(calibrated.mean()),
            'precision_top_10pct': float(y[top].mean()),
        })
        print(f'{cutoff}: n={len(y)}, RFM AP={rows[-1]["rfm_ap"]:.4f}, full AP={rows[-1]["all_history_ap"]:.4f}, '
              f'delta={rows[-1]["ap_difference"]:.4f}, calibrated Brier={rows[-1]["calibrated_brier"]:.4f}', flush=True)
    if september is None:
        raise AssertionError('Missing September check.')
    result = pd.DataFrame(rows)
    # Save only aggregate results. Customer predictions and fitted models stay local.
    result.to_csv(OUT/'temporal_robustness.csv', index=False)
    pd.DataFrame(decision_rows).to_csv(OUT/'decision_utility.csv', index=False)
    cohort_rows = []
    cutoffs = list(cohorts)
    for i, first in enumerate(cutoffs):
        for second in cutoffs[i+1:]:
            shared = len(cohorts[first] & cohorts[second])
            cohort_rows.append({'first_cutoff': first, 'second_cutoff': second,
                                'shared_customers': shared,
                                'first_customers': len(cohorts[first]),
                                'second_customers': len(cohorts[second]),
                                'overlap_share_of_first': shared/len(cohorts[first]),
                                'jaccard': shared/len(cohorts[first] | cohorts[second])})
    pd.DataFrame(cohort_rows).to_csv(OUT/'cohort_overlap.csv', index=False)
    calibration_params = {
        'model': 'all_history_logistic', 'fit_cutoff': '2011-06-01',
        'method': 'Platt logistic calibration on validation log-odds',
        'slope': float(calibrator.coef_[0, 0]),
        'intercept': float(calibrator.intercept_[0]),
        'warning': 'Exploratory adjustment; still underpredicts in the later test groups.'
    }
    (OUT/'calibration_params.json').write_text(json.dumps(calibration_params, indent=2), encoding='utf-8')
    groups = september.assign(group=pd.qcut(september.full, 5, labels=False, duplicates='drop')+1)
    reliability = groups.groupby('group').agg(
        customers=('y','size'), observed_rate=('y','mean'),
        raw_mean_probability=('full','mean'), calibrated_mean_probability=('calibrated','mean')).reset_index()
    reliability.to_csv(OUT/'calibration_comparison.csv', index=False)
    summary = {
        'method': 'Frozen RFM and selected full logistic models; customer-row paired bootstrap (2,000 resamples, seed 5588); Platt calibration fit on 2011-06-01 validation predictions only.',
        'scope': 'Exploratory temporal sensitivity, not independent confirmation. Aug and Oct 60-day label windows do not overlap; customers and their histories can overlap. Sep results were viewed previously.',
        'source_sha256': source_audit['sha256'],
        'validation_n': len(validation), 'calibration_method': 'Logistic regression on validation log-odds',
        'calibration_fit_cutoff': '2011-06-01',
        'new_cutoffs': ['2011-08-01', '2011-10-01'],
        'bootstrap_repeats': 2000,
        'results': rows,
    }
    (OUT/'robustness_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print('Saved aggregate results to', OUT)


if __name__ == '__main__':
    main()
