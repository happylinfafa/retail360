"""Focused historical 60-day repeat-purchase study and evidence review."""
import json
import os
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT/'results/repurchase_focused'
LOCAL = ROOT/'runtime/repurchase_focused_v1'

st.set_page_config(page_title='Retail360 | 60-day repurchase', layout='wide')
st.title('Will this customer buy again within 60 days?')
st.caption('Historical backtest using UCI Online Retail II · No paid API or LLM required')


@st.cache_data
def read_public():
    return (json.loads((PUBLIC/'summary.json').read_text(encoding='utf-8')),
            pd.read_csv(PUBLIC/'metrics.csv'),
            pd.read_csv(PUBLIC/'test_calibration.csv'),
            pd.read_csv(PUBLIC/'test_slices.csv'),
            json.loads((PUBLIC/'demo_examples.json').read_text(encoding='utf-8')))


@st.cache_data
def read_local():
    return pd.read_csv(LOCAL/'customer_predictions.csv',
                       dtype={'CustomerID': str, 'cutoff': str})


@st.cache_data
def history_rows(customer):
    path = ROOT/'data/repurchase_v1/test/2011-09-01/history_transactions.csv.gz'
    rows = pd.read_csv(path, dtype={'CustomerID': str, 'Invoice': str, 'StockCode': str})
    return rows[rows.CustomerID == customer]


if not (PUBLIC/'summary.json').exists():
    st.error('Focused results are missing. Run scripts/train_focused_repurchase.py first.')
    st.stop()

summary, metrics, calibration, slices, examples = read_public()
local_available = ((LOCAL/'customer_predictions.csv').exists()
                   and os.getenv('RETAIL360_PUBLIC_DEMO_ONLY') != '1')

st.write('We use a customer’s previous **180 days of valid purchases** to predict whether '
         'they make another valid purchase in the **next 60 days**. Future orders are used '
         'only to check predictions, never as model inputs.')
st.info('This is a 2011 historical evaluation. It does not predict customers today or '
        'measure the effect of a marketing campaign.')

selected = summary['selected_model']
test_selected = summary['test_selected_model']
baseline = metrics[(metrics.model == 'overall_rate') & (metrics.split == 'test')].iloc[0]
c1, c2, c3 = st.columns(3)
c1.metric('Held-out customers', f"{summary['test_rows']:,}")
c2.metric('Selected model · test AP', f"{test_selected['average_precision']:.3f}")
c3.metric('Overall-rate baseline · test AP', f"{baseline.average_precision:.3f}")
st.caption('AP (average precision) measures ranking of actual repeat buyers; higher is better. '
           f'The model {selected} was chosen using validation AP. The simplified scope was '
           'chosen after earlier graph-model results were seen, so this is an exploratory '
           'comparison, not a new untouched confirmation.')

st.subheader('1 · Compare models and historical information')
model_order = ['overall_rate', 'rfm_logistic', 'rfm_temporal_logistic',
               'all_history_logistic', 'all_history_boosted_tree']
chart = metrics[(metrics.model.isin(model_order)) & (metrics.split == 'test')]
chart = chart.set_index('model').loc[model_order, ['average_precision']]
st.bar_chart(chart, color='#087ea4', height=310)
st.dataframe(metrics[metrics.split == 'test'][['model', 'average_precision', 'roc_auc',
                                               'brier', 'precision_top_10pct']],
             hide_index=True, use_container_width=True)
st.caption('RFM = days since last purchase, invoice count, and historical spending. '
           'Temporal features add recent activity and purchase gaps; the all-history set '
           'also includes product diversity. These are descriptive purchase features, not a graph model.')

st.subheader('2 · Check whether the probabilities are reliable')
plot = calibration.set_index('probability_group')[['mean_predicted_probability',
                                                    'actual_repurchase_rate']]
st.line_chart(plot, height=300)
st.dataframe(calibration, hide_index=True, use_container_width=True)
if (calibration.actual_repurchase_rate > calibration.mean_predicted_probability).all():
    st.warning('In every test group, the observed purchase rate exceeds the mean predicted '
               'probability. The model ranks customers usefully, but its numeric probabilities '
               'underestimate purchases on this test date. Do not treat them as calibrated odds.')

st.subheader('3 · Inspect a group where the model may be less reliable')
frequency = slices[slices.dimension == 'frequency_group'].set_index('group')
st.bar_chart(frequency[['actual_repurchase_rate', 'mean_predicted_probability']],
             height=280, stack=False)
st.dataframe(slices, hide_index=True, use_container_width=True)
st.caption('Group results are descriptive. They do not show that changing a customer’s '
           'purchase frequency would cause a different outcome.')

st.subheader('4 · Review one real historical customer')
if local_available:
    population = read_local()
    choice = st.selectbox('Customer ID · full local test set',
                          population.CustomerID.tolist(), key='focused_customer')
    row = population[population.CustomerID == choice].iloc[0]
    probability = float(row.probability)
    outcome = int(row.repurchase_60d)
    feature_values = {key: (None if pd.isna(row[key]) else float(row[key]))
                      for key in summary['selected_features']}
    evidence = history_rows(choice).sort_values('InvoiceDate', ascending=False)
    evidence = evidence[['Invoice', 'StockCode', 'InvoiceDate', 'AmountPence',
                         'Country', 'RowID']]
    st.caption('Local full-data review: all 2,772 test customers are available.')
else:
    indexed = {entry['customer_id']: entry for entry in examples}
    choice = st.selectbox('Customer ID · 10 reproducible public examples',
                          list(indexed), key='focused_customer')
    entry = indexed[choice]
    probability = float(entry['predicted_probability'])
    outcome = int(entry['observed_repurchase_60d'])
    feature_values = entry['features']
    evidence = pd.DataFrame(entry['source_rows'])
    st.caption('Cloud-friendly demonstration: these 10 real customers are a display sample '
               'from the test set. The model was evaluated on all 2,772 customers; the '
               'application is showing stored historical predictions, not scoring live orders.')

context = (choice, local_available)
if st.session_state.get('focused_context') != context:
    st.session_state['focused_context'] = context
    st.session_state.pop('focused_review', None)
    st.session_state.pop('focused_note', None)

st.metric('Historical model score', f'{probability:.1%}')
st.caption('This is an estimated probability from the 2011 model, not a guarantee. '
           'The test-group chart above shows that these probabilities are under-calibrated.')
st.dataframe(pd.DataFrame([{'Feature': k, 'Historical value': v}
                           for k, v in feature_values.items()]),
             hide_index=True, use_container_width=True)
with st.expander('Inspect original purchase rows and provenance'):
    st.dataframe(evidence, hide_index=True, use_container_width=True)
    st.caption('RowID identifies the original UCI workbook worksheet and Excel row. '
               'All displayed transactions precede the prediction cutoff.')
with st.expander('Reveal the future outcome · evaluation only'):
    st.write('Purchased again within 60 days:', bool(outcome))
    st.caption('This future outcome was never a model input.')

st.subheader('Human review')
decision = st.radio('Your assessment', ['Needs review', 'Accept for analysis',
                                         'Reject / investigate'], key='focused_review')
note = st.text_area('Reason or correction', key='focused_note')
review = {'customer_id': choice, 'cutoff': '2011-09-01', 'model': selected,
          'historical_model_score': probability, 'decision': decision, 'note': note}
st.download_button('Download review', json.dumps(review, indent=2),
                   'repurchase_review.json', mime='application/json')
st.caption('Source: Daqing Chen, UCI Online Retail II (CC BY 4.0). '
           'The research code and fixed data protocol are in the repository.')
