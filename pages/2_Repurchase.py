"""Focused historical 60-day repeat-purchase study and evidence review."""
import json
import math
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
st.info('Try the demo: compare the models below, then go to **Section 4** to select '
        'one of 10 real customers, inspect their original invoice rows, and record '
        'your own review. This public demonstration uses stored historical scores.')


@st.cache_data
def read_public():
    return (json.loads((PUBLIC/'summary.json').read_text(encoding='utf-8')),
            pd.read_csv(PUBLIC/'metrics.csv'),
            pd.read_csv(PUBLIC/'test_calibration.csv'),
            pd.read_csv(PUBLIC/'test_slices.csv'),
            json.loads((PUBLIC/'demo_examples.json').read_text(encoding='utf-8')))


@st.cache_data
def read_robustness():
    paths = [PUBLIC/'temporal_robustness.csv', PUBLIC/'calibration_comparison.csv',
             PUBLIC/'calibration_params.json']
    if not all(path.exists() for path in paths):
        return None
    return (pd.read_csv(paths[0]), pd.read_csv(paths[1]),
            json.loads(paths[2].read_text(encoding='utf-8')))


@st.cache_data
def read_decision_evidence():
    paths = [PUBLIC/'decision_utility.csv', PUBLIC/'cohort_overlap.csv',
             PUBLIC/'capacity_sensitivity.csv']
    if not all(path.exists() for path in paths):
        return None
    return tuple(pd.read_csv(path) for path in paths)


def calibrated_probability(raw, params):
    value = min(max(float(raw), 1e-6), 1-1e-6)
    logit = math.log(value/(1-value))
    adjusted_logit = params['intercept'] + params['slope']*logit
    return 1/(1+math.exp(-adjusted_logit))


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
robustness = read_robustness()
decision_evidence = read_decision_evidence()
local_available = ((LOCAL/'customer_predictions.csv').exists()
                   and os.getenv('RETAIL360_PUBLIC_DEMO_ONLY') != '1')

st.write('We use a customer’s previous **180 days of valid purchases** to predict whether '
         'they make another valid purchase in the **next 60 days**. Future orders are used '
         'only to check predictions, never as model inputs.')
st.info('This is a 2011 historical evaluation. It does not predict customers today or '
        'measure the effect of a marketing campaign.')
if not local_available:
    st.info('Public demo: 10 stored predictions for real historical customers. It does not '
            'score a new customer live. Research metrics below use the full test cohort, '
            'not just these 10 display examples.')

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
             hide_index=True, width='stretch')
st.caption('RFM = days since last purchase, invoice count, and historical spending. '
           'Temporal features add recent activity and purchase gaps; the all-history set '
           'also includes product diversity. These are descriptive purchase features, not a graph model.')

if robustness is not None:
    temporal, calibrated_groups, params = robustness
    st.subheader('Additional time windows · exploratory robustness')
    st.dataframe(temporal[['cutoff', 'n', 'positive_rate', 'rfm_ap', 'all_history_ap',
                           'ap_difference', 'ap_difference_ci_low', 'ap_difference_ci_high',
                           'precision_top_10pct']], hide_index=True, width='stretch')
    st.caption('The 95% intervals use 2,000 paired customer-row bootstrap resamples within '
               'each cutoff. August and October have non-overlapping 60-day outcome windows, '
               'but customers and histories can overlap; September was examined previously. '
               'These checks are temporal sensitivity evidence, not independent external confirmation.')

if decision_evidence is not None:
    utility, overlap, capacity = decision_evidence
    st.subheader('What changes with a fixed review capacity?')
    st.write('Suppose an analyst can review only the top 10% of eligible customers. '
             'The two models use the same number of review slots at each cutoff.')
    st.dataframe(utility[['cutoff', 'review_capacity', 'rfm_repeat_buyers_found',
                          'full_repeat_buyers_found', 'incremental_repeat_buyers_found',
                          'customers_selected_only_by_full']],
                 hide_index=True, width='stretch')
    st.caption('The full model found only 1, 2, and 1 additional actual repeat buyers '
               'at the August, September, and October cutoffs. This is a historical '
               'ranking comparison, not an estimate of the effect of contacting anyone.')
    st.write('The result also depends on how many customers the analyst can review. '
             'At the September cutoff:')
    st.dataframe(capacity.loc[capacity.cutoff == '2011-09-01', [
        'capacity_share', 'review_capacity', 'rfm_repeat_buyers_found',
        'full_repeat_buyers_found', 'incremental_repeat_buyers_found']],
        hide_index=True, width='stretch')
    st.caption('At 5% capacity the full model finds no extra repeat buyers; '
               'at 10% it finds two more; at 20% it finds one fewer. '
               'No marketing intervention was evaluated.')
    st.write('The evaluation windows share many of the same customers:')
    st.dataframe(overlap[['first_cutoff', 'second_cutoff', 'shared_customers',
                          'overlap_share_of_first']], hide_index=True, width='stretch')

st.subheader('2 · Check whether the probabilities are reliable')
if robustness is not None:
    plot = calibrated_groups.set_index('group')[[
        'raw_mean_probability', 'calibrated_mean_probability', 'observed_rate']]
else:
    plot = calibration.set_index('probability_group')[[
        'mean_predicted_probability', 'actual_repurchase_rate']]
st.line_chart(plot, height=300)
st.dataframe(calibrated_groups if robustness is not None else calibration,
             hide_index=True, width='stretch')
if robustness is not None:
    st.caption('Platt calibration was fit on June 2011 validation predictions, before the '
               'later evaluation windows. September Brier score improved from '
               f"{temporal.loc[temporal.cutoff == '2011-09-01', 'raw_brier'].iloc[0]:.3f} to "
               f"{temporal.loc[temporal.cutoff == '2011-09-01', 'calibrated_brier'].iloc[0]:.3f}. "
               'Observed rates still exceed adjusted scores in the five displayed groups.')
if (calibration.actual_repurchase_rate > calibration.mean_predicted_probability).all():
    st.warning('In every test group, the observed purchase rate exceeds the mean predicted '
               'probability. The model ranks customers usefully, but its numeric probabilities '
               'underestimate purchases on this test date. Do not treat them as calibrated odds.')

st.subheader('3 · Inspect a group where the model may be less reliable')
frequency = slices[slices.dimension == 'frequency_group'].set_index('group')
st.bar_chart(frequency[['actual_repurchase_rate', 'mean_predicted_probability']],
             height=280, stack=False)
st.dataframe(slices, hide_index=True, width='stretch')
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

adjusted = calibrated_probability(probability, params) if robustness is not None else None
if adjusted is None:
    st.metric('Historical model score', f'{probability:.1%}')
else:
    score1, score2 = st.columns(2)
    score1.metric('Original 2011 model score', f'{probability:.1%}')
    score2.metric('Validation-calibrated estimate', f'{adjusted:.1%}')
st.warning('Use this score for historical ranking and evidence review. Even after calibration, '
           'the model underestimates observed repeat-purchase rates, so an individual score '
           'is not a reliable exact probability or a measured marketing effect.')
st.write('Evidence path: customer', choice, '→ prior 180-day invoices → historical '
         'features → model score → human review. Future purchases are revealed only for evaluation.')
st.dataframe(pd.DataFrame([{'Feature': k, 'Historical value': v}
                           for k, v in feature_values.items()]),
             hide_index=True, width='stretch')
with st.expander('Inspect original purchase rows and provenance'):
    st.dataframe(evidence, hide_index=True, width='stretch')
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
          'historical_model_score': probability,
          'validation_calibrated_estimate': adjusted,
          'decision': decision, 'note': note}
st.download_button('Download review', json.dumps(review, indent=2),
                   'repurchase_review.json', mime='application/json')
st.caption('Source: Daqing Chen, UCI Online Retail II (CC BY 4.0). '
           'The research code and fixed data protocol are in the repository.')
