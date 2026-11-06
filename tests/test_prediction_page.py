from pathlib import Path
import pytest
from streamlit.testing.v1 import AppTest


def test_prediction_evidence_and_review_reset():
    root = Path(__file__).resolve().parents[1]
    if not (root/'runtime/repurchase_focused_v1/customer_predictions.csv').exists():
        pytest.skip('Run the focused training pipeline before this integration test.')
    app = AppTest.from_file(str(root/'pages/2_Repurchase.py'), default_timeout=30).run()
    assert not app.exception
    assert len(app.metric) == 4 and len(app.dataframe) >= 3
    app.radio(key='focused_review').set_value('Reject / investigate').run()
    app.text_area(key='focused_note').input('Check source purchase rows.').run()
    customers = app.selectbox(key='focused_customer').options
    app.selectbox(key='focused_customer').select(customers[1]).run()
    assert not app.exception
    assert app.radio(key='focused_review').value == 'Needs review'
    assert app.text_area(key='focused_note').value == ''


def test_public_demo_runs_without_local_model_files(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.setenv('RETAIL360_PUBLIC_DEMO_ONLY', '1')
    app = AppTest.from_file(str(root/'pages/2_Repurchase.py'), default_timeout=30).run()
    assert not app.exception
    assert len(app.selectbox(key='focused_customer').options) == 10
    assert len(app.metric) == 4
    assert len(app.dataframe) >= 3
