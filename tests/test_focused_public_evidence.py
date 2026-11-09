"""The small public demonstration must point to real pre-cutoff source rows."""
import json
from pathlib import Path

import pandas as pd
import pytest


def test_public_examples_match_original_history_rows():
    root = Path(__file__).resolve().parents[1]
    examples = json.loads((root/'results/repurchase_focused/demo_examples.json').read_text(encoding='utf-8'))
    history = root/'data/repurchase_v1/test/2011-09-01/history_transactions.csv.gz'
    if not history.exists():
        pytest.skip('Download and prepare the UCI workbook for source-row audit')
    source = pd.read_csv(history,
                         dtype={'CustomerID': str, 'Invoice': str, 'StockCode': str})
    by_row = source.set_index('RowID')
    assert len(examples) == 10
    for example in examples:
        assert example['cutoff'] == '2011-09-01'
        assert example['source_rows']
        for evidence in example['source_rows']:
            original = by_row.loc[evidence['RowID']]
            assert original['CustomerID'] == example['customer_id']
            assert original['Invoice'] == evidence['Invoice']
            assert original['StockCode'] == evidence['StockCode']
            assert pd.Timestamp(evidence['InvoiceDate']) < pd.Timestamp(example['cutoff'])


def test_public_examples_have_traceable_evidence():
    root = Path(__file__).resolve().parents[1]
    examples = json.loads((root/'results/repurchase_focused/demo_examples.json').read_text(encoding='utf-8'))
    assert len(examples) == 10
    for example in examples:
        assert example['customer_id']
        assert example['cutoff'] == '2011-09-01'
        assert example['source_rows']
        for row in example['source_rows']:
            assert row['RowID'] is not None
            assert row['Invoice']
            assert row['StockCode']
            assert pd.Timestamp(row['InvoiceDate']) < pd.Timestamp(example['cutoff'])
