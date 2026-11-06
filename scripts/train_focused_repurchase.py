"""Focused 60-day repeat-purchase study, without graph or LLM features.

The existing frozen data protocol is reused. Candidate selection uses only
validation AP; the test snapshot is loaded only after selection is written.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from modeling import RFM, choose_threshold, evaluate, feature_input, make_model
from repurchase import FEATURE_COLUMNS, ROOT, load_protocol


TEMPORAL = RFM + ['aov_gbp', 'orders_last_30d', 'active_purchase_days',
                  'mean_invoice_gap_days']
FEATURE_SETS = {
    'rfm': RFM,
    'rfm_temporal': TEMPORAL,
    'all_history': FEATURE_COLUMNS,
}
PUBLIC = ROOT/'results/repurchase_focused'


def load_split(data, protocol, split):
    """One historical customer-snapshot row; labels are joined only as targets."""
    frames = []
    for spec in protocol['snapshots']:
        if spec['split'] != split:
            continue
        folder = data/split/spec['cutoff']
        keys = ['cutoff', 'CustomerID']
        features = pd.read_csv(folder/'features.csv', dtype={'CustomerID': str, 'cutoff': str})
        labels = pd.read_csv(folder/'labels.csv', dtype={'CustomerID': str, 'cutoff': str})
        audit = pd.read_csv(folder/'audit.csv', dtype={'CustomerID': str, 'cutoff': str})
        if set(map(tuple, features[keys].values)) != set(map(tuple, labels[keys].values)):
            raise ValueError(f'Feature/label keys differ: {folder}')
        table = features.merge(labels[keys+['repurchase_60d']], on=keys, validate='one_to_one')
        table = table.merge(audit[keys+['last_observed_country']], on=keys, validate='one_to_one')
        if len(table) != len(features) or not table.repurchase_60d.isin([0, 1]).all():
            raise ValueError(f'Invalid customer snapshot: {folder}')
        frames.append(table)
    return pd.concat(frames, ignore_index=True).sort_values(['cutoff', 'CustomerID']).reset_index(drop=True)


def summaries(test_predictions):
    table = test_predictions.copy()
    table['probability_group'] = pd.qcut(table.probability, 5, labels=False, duplicates='drop') + 1
    calibration = table.groupby('probability_group').agg(
        customers=('CustomerID', 'size'),
        mean_predicted_probability=('probability', 'mean'),
        actual_repurchase_rate=('repurchase_60d', 'mean')).reset_index()
    table['frequency_group'] = np.select(
        [table.frequency == 1, table.frequency.between(2, 4)],
        ['1 invoice', '2–4 invoices'], default='5+ invoices')
    table['recency_group'] = pd.cut(table.recency_days,
        bins=[-1, 30, 90, np.inf], labels=['0–30 days', '31–90 days', '91+ days'])
    slices = []
    for field in ['frequency_group', 'recency_group']:
        for group, subset in table.groupby(field, observed=True):
            slices.append({'dimension': field, 'group': str(group),
                'customers': int(len(subset)),
                'actual_repurchase_rate': float(subset.repurchase_60d.mean()),
                'mean_predicted_probability': float(subset.probability.mean()),
                'absolute_rate_gap': float(abs(subset.probability.mean()-subset.repurchase_60d.mean()))})
    return calibration, pd.DataFrame(slices)


def public_examples(data, test_predictions, limit_per_group=2):
    """Small public UCI sample with authentic pre-cutoff row provenance."""
    selected = []
    for group, subset in test_predictions.groupby('probability_group'):
        selected.append(subset.sample(n=min(limit_per_group, len(subset)), random_state=5588))
    examples = pd.concat(selected).sort_values(['probability_group', 'CustomerID'])
    source = pd.read_csv(data/'test/2011-09-01/history_transactions.csv.gz',
        dtype={'CustomerID': str, 'Invoice': str, 'StockCode': str})
    source = source[source.CustomerID.isin(examples.CustomerID)]
    records = []
    for row in examples.itertuples(index=False):
        evidence = source[source.CustomerID == row.CustomerID].sort_values('InvoiceDate', ascending=False).head(5)
        if evidence.empty:
            raise ValueError(f'No historical evidence for {row.CustomerID}')
        records.append({
            'customer_id': row.CustomerID, 'cutoff': row.cutoff,
            'probability_group': int(row.probability_group),
            'predicted_probability': float(row.probability),
            'observed_repurchase_60d': int(row.repurchase_60d),
            'features': {name: (None if pd.isna(getattr(row, name)) else float(getattr(row, name)))
                         for name in FEATURE_COLUMNS},
            'source_rows': evidence[['Invoice', 'StockCode', 'InvoiceDate', 'AmountPence',
                                     'Country', 'RowID']].to_dict(orient='records'),
        })
    return records


def run(data, output, public):
    if output.exists() and any(output.iterdir()):
        raise ValueError('Output is not empty; preserve the earlier run and choose a new --output.')
    protocol = load_protocol()
    manifest_path = data/'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    if manifest['protocol'] != protocol or manifest['feature_columns'] != FEATURE_COLUMNS:
        raise ValueError('Prepared data do not match the frozen protocol.')
    train = load_split(data, protocol, 'train')
    valid = load_split(data, protocol, 'validation')
    output.mkdir(parents=True, exist_ok=True)
    candidates, rows = {}, []
    candidates_to_fit = [('overall_rate', 'dummy', RFM)]
    for family, columns in FEATURE_SETS.items():
        for method in ['logistic', 'boosted_tree']:
            candidates_to_fit.append((f'{family}_{method}', method, columns))
    for name, method, columns in candidates_to_fit:
        model = make_model(method)
        model.fit(feature_input(train, columns), train.repurchase_60d)
        scores = model.predict_proba(feature_input(valid, columns))[:, 1]
        threshold = choose_threshold(valid.repurchase_60d, scores)
        metrics = evaluate(valid.repurchase_60d, scores, threshold)
        candidates[name] = (model, columns, threshold)
        rows.append({'model': name, 'split': 'validation', 'features': ', '.join(columns), **metrics})
        print(f'{name}: validation AP={metrics["average_precision"]:.4f}', flush=True)
    selected = max((r for r in rows if r['model'] != 'overall_rate'),
                   key=lambda r: r['average_precision'])['model']
    selection = {
        'selected_model': selected,
        'criterion': 'Highest validation average precision among non-graph candidates.',
        'test_used_for_selection': False,
        'protocol': protocol['version'],
        'manifest_sha256': hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        'scope_note': 'The focused scope was chosen after earlier graph-model test results were viewed; treat this focused comparison as exploratory, not a new untouched confirmation.',
    }
    (output/'selection.json').write_text(json.dumps(selection, indent=2), encoding='utf-8')

    test = load_split(data, protocol, 'test')
    for name, (model, columns, threshold) in candidates.items():
        scores = model.predict_proba(feature_input(test, columns))[:, 1]
        rows.append({'model': name, 'split': 'test', 'features': ', '.join(columns),
                     **evaluate(test.repurchase_60d, scores, threshold)})
    metrics = pd.DataFrame(rows)
    metrics.to_csv(output/'metrics.csv', index=False)
    model, columns, threshold = candidates[selected]
    predictions = test.copy()
    predictions['probability'] = model.predict_proba(feature_input(test, columns))[:, 1]
    predictions['prediction_at_threshold'] = (predictions.probability >= threshold).astype(int)
    predictions['probability_group'] = pd.qcut(predictions.probability, 5, labels=False, duplicates='drop') + 1
    predictions.to_csv(output/'customer_predictions.csv', index=False)
    calibration, slices = summaries(predictions)
    calibration.to_csv(output/'test_calibration.csv', index=False)
    slices.to_csv(output/'test_slices.csv', index=False)
    joblib.dump({'model': model, 'columns': columns, 'threshold': threshold,
                 'selection': selection}, output/'model.joblib')

    public.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(public/'metrics.csv', index=False)
    calibration.to_csv(public/'test_calibration.csv', index=False)
    slices.to_csv(public/'test_slices.csv', index=False)
    examples = public_examples(data, predictions)
    (public/'demo_examples.json').write_text(json.dumps(examples, indent=2, ensure_ascii=False), encoding='utf-8')
    summary = {
        'question': 'Can past purchases predict whether an existing customer buys again within 60 days?',
        'source': 'UCI Online Retail II, doi:10.24432/C5CG6D',
        'raw_rows': manifest['source']['raw_rows'] if 'source' in manifest else None,
        'history_days': protocol['lookback_days'], 'outcome_days': protocol['horizon_days'],
        'train_rows': int(len(train)), 'validation_rows': int(len(valid)), 'test_rows': int(len(test)),
        'selected_model': selected, 'selected_features': columns,
        'selection': selection,
        'test_selected_model': metrics[(metrics.model == selected) & (metrics.split == 'test')].iloc[0].to_dict(),
        'demo_customer_count': len(examples),
        'limitations': ['Historical backtest, not a current-day forecast.',
                        'One retailer and one held-out test cutoff.',
                        'No causal claim about marketing actions.',
                        'Public app examples are a small display sample, not the full test cohort.'],
    }
    (public/'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding='utf-8')
    print(f'Finished: {selected}; test AP={summary["test_selected_model"]["average_precision"]:.4f}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, default=ROOT/'data/repurchase_v1')
    parser.add_argument('--output', type=Path, default=ROOT/'runtime/repurchase_focused_v1')
    parser.add_argument('--public', type=Path, default=PUBLIC)
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        run(args.data, args.output, args.public)
