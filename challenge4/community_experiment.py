"""Temporal ablation: does Louvain community filtering help product discovery?

Only pre-cutoff purchases construct the graph, communities, and candidates.
The 60-day future is used exclusively to score held-out new-to-customer items.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime
import json
from pathlib import Path

import networkx as nx
import openpyxl

from ai_design_demo import (DEFAULT_XLSX, END, MIN_JACCARD, MIN_SHARED,
                            SEED, START, load_rows, make_cohort, make_graph)

ROOT = Path(__file__).resolve().parent
HOLDOUT_END = datetime(2011, 11, 1)
TOP_K = 10
PEER_K = 5
NON_PRODUCT_CODES = {'POST', 'DOT', 'M', 'D', 'BANK CHARGES', 'CRUK', 'C2', 'S', 'AMAZONFEE'}


def valid_product(code):
    return code not in NON_PRODUCT_CODES and not code.startswith('gift_')


def future_products(path, selected):
    """Read only the after-cutoff labels; never feed these into retrieval."""
    book = openpyxl.load_workbook(path, read_only=True, data_only=True)
    sheet = book['Year 2010-2011']
    selected = set(selected)
    future = defaultdict(set)
    for row in sheet.iter_rows(min_row=2, values_only=True):
        invoice, code, _description, qty, date, price, customer, _country = row
        if not isinstance(date, datetime) or not (END <= date < HOLDOUT_END):
            continue
        if not customer or not code or not invoice or str(invoice).upper().startswith('C'):
            continue
        if not isinstance(qty, (int, float)) or qty <= 0 or not isinstance(price, (int, float)) or price <= 0:
            continue
        cid = str(int(customer)) if isinstance(customer, (int, float)) else str(customer).strip()
        code = str(code).strip()
        if cid in selected and valid_product(code):
            future[cid].add(code)
    book.close()
    return future


def detect_communities(graph):
    g = nx.Graph()
    g.add_nodes_from(graph)
    for a, peers in graph.items():
        for b, weight in peers.items():
            if a < b:
                g.add_edge(a, b, weight=weight)
    communities = nx.community.louvain_communities(g, weight='weight', seed=SEED)
    assigned = {}
    for members in communities:
        gid = min(members)
        for cid in members:
            assigned[cid] = gid
    assert set(assigned) == set(graph)
    return g, assigned, nx.community.modularity(g, communities, weight='weight')


def rank_products(target, customers, graph, groups=None):
    own = {p for p in customers[target]['products'] if valid_product(p)}
    peers = [(peer, score) for peer, score in graph[target].items()
             if groups is None or groups[peer] == groups[target]]
    peers.sort(key=lambda item: (-item[1], item[0]))
    peers = peers[:PEER_K]
    scores = Counter()
    source = {}
    for peer, score in peers:
        for row in customers[peer]['rows']:
            product = row['product']
            if product in own or not valid_product(product):
                continue
            # One vote per peer/product, not per transaction line.
            if (peer, product) in source:
                continue
            source[(peer, product)] = {'peer':peer, 'invoice':row['invoice'],
                                      'sourceRow':row['sourceRow'], 'similarity':round(score, 6)}
            scores[product] += score
    ranking = sorted(scores, key=lambda p: (-scores[p], p))[:TOP_K]
    evidence = {p: sorted((v for (peer, product), v in source.items() if product == p),
                          key=lambda x: (-x['similarity'], x['peer']))[:2] for p in ranking}
    return ranking, evidence, len(peers)


def describe_groups(customers, selected, groups, sizes):
    """Descriptive, source-backed group tendencies; never label a preference."""
    cohort_prevalence = Counter(p for cid in selected for p in customers[cid]['products'] if valid_product(p))
    descriptions = {}
    source = {}
    for cid in selected:
        for row in customers[cid]['rows']:
            p = row['product']
            if valid_product(p):
                descriptions.setdefault(p,row['description'])
                source.setdefault((cid,p),{'customer':cid,'invoice':row['invoice'],
                                            'sourceRow':row['sourceRow']})
    profiles = []
    for group,size in sorted(sizes.items(),key=lambda x:(-x[1],x[0]))[:6]:
        members = [cid for cid in selected if groups[cid] == group]
        frequency = Counter(p for cid in members for p in customers[cid]['products'] if valid_product(p))
        top = []
        for p,n in sorted(frequency.items(),key=lambda x:(-x[1],x[0]))[:5]:
            witness = next(source[(cid,p)] for cid in members if (cid,p) in source)
            top.append({'product':p,'description':descriptions.get(p,''),
                        'customerCount':n,'groupPrevalence':round(n/size,3),
                        'cohortPrevalence':round(cohort_prevalence[p]/len(selected),3),
                        'prevalenceRatio':round((n/size)/(cohort_prevalence[p]/len(selected)),3),
                        'source':witness})
        profiles.append({'group':group,'size':size,'topProducts':top})
    return profiles


def evaluate(path: Path):
    customers, valid_rows = load_rows(path)
    eligible, selected = make_cohort(customers)
    graph, edge_count = make_graph(customers, selected)
    nx_graph, groups, modularity = detect_communities(graph)
    future = future_products(path, selected)
    counts = Counter(groups.values())
    # Pre-cutoff popularity is an independent, simple comparison.
    popularity = Counter(p for cid in selected for p in customers[cid]['products'] if valid_product(p))
    metrics = {mode:{'hits':0,'recommended':0,'eligibleFuture':0,'queriesWithCandidates':0,
                     'queriesWithHit':0,'peerCount':0} for mode in ('popularity','global_graph','community_graph')}
    cases = []
    eligible_queries = 0
    for cid in selected:
        historical = {p for p in customers[cid]['products'] if valid_product(p)}
        gold = future[cid] - historical
        if not gold:
            continue
        eligible_queries += 1
        outputs = {}
        for mode in metrics:
            if mode == 'popularity':
                ranking = [p for p,_ in sorted(popularity.items(), key=lambda x: (-x[1],x[0]))
                           if p not in historical][:TOP_K]
                evidence = {}; peers = 0
            else:
                ranking, evidence, peers = rank_products(cid, customers, graph,
                                  groups if mode == 'community_graph' else None)
            hits = sorted(set(ranking) & gold)
            m = metrics[mode]
            m['hits'] += len(hits)
            m['recommended'] += len(ranking)
            m['eligibleFuture'] += len(gold)
            m['queriesWithCandidates'] += bool(ranking)
            m['queriesWithHit'] += bool(hits)
            m['peerCount'] += peers
            outputs[mode] = {'top10':ranking,'hits':hits,'peerCount':peers,
                             'firstEvidence':evidence.get(ranking[0],[]) if ranking else []}
        if len(cases) < 8:
            cases.append({'customer':cid,'community':groups[cid],'futureNewProductCount':len(gold),
                          'outputs':outputs})
    for m in metrics.values():
        m['queries'] = eligible_queries
        m['microRecallAt10'] = round(m['hits']/m['eligibleFuture'], 6) if m['eligibleFuture'] else 0
        m['microPrecisionAt10'] = round(m['hits']/m['recommended'], 6) if m['recommended'] else 0
        m['hitRateAt10'] = round(m['queriesWithHit']/eligible_queries, 6) if eligible_queries else 0
        m['candidateCoverage'] = round(m['queriesWithCandidates']/eligible_queries, 6) if eligible_queries else 0
        m['meanPeerCount'] = round(m['peerCount']/eligible_queries, 3) if eligible_queries else 0
    top_groups = sorted(counts.items(), key=lambda x: (-x[1],x[0]))[:8]
    profiles = describe_groups(customers,selected,groups,counts)
    checks = {
        'all_customers_assigned':len(groups)==len(selected),
        'ranking_uses_only_historical_rows':all(
            all(src['sourceRow'].startswith('Year 2010-2011:') for src in case['outputs']['community_graph']['firstEvidence'])
            for case in cases),
        'global_retrieval_coverage_at_least_community':metrics['global_graph']['candidateCoverage'] >= metrics['community_graph']['candidateCoverage'],
        'louvain_modularity_finite':-1 <= modularity <= 1,
    }
    assert all(checks.values()), checks
    return {'source':'UCI Online Retail II, Year 2010-2011',
            'sourceURL':'https://archive.ics.uci.edu/dataset/502/online+retail+ii',
            'historicalWindow':[START.date().isoformat(),END.date().isoformat()],
            'futureEvaluationWindow':[END.date().isoformat(),HOLDOUT_END.date().isoformat()],
            'sample':{'validHistoricalRows':valid_rows,'eligibleCustomers':len(eligible),
                      'selectedCustomers':len(selected),'similarityEdges':edge_count,
                      'seed':SEED,'minSharedProducts':MIN_SHARED,'minJaccard':MIN_JACCARD},
            'communityMethod':{'name':'NetworkX weighted Louvain','version':nx.__version__,
                               'communityCount':len(counts),'modularity':round(modularity,6),
                               'topCommunities':[{'id':k,'size':v} for k,v in top_groups],
                               'assignments':groups,'profiles':profiles},
            'evaluation':{'task':'Recommend new-to-customer products using historical peer products; score against next 60 days',
                          'queryCount':eligible_queries,'topK':TOP_K,'peerK':PEER_K,
                          'baselineDefinitions':{'popularity':'Top cohort products before cutoff, excluding own history',
                                                 'global_graph':'Top 5 directly similar peers, any community',
                                                 'community_graph':'Top 5 directly similar peers restricted to weighted-Louvain community'},
                          'metrics':metrics,'cases':cases},'checks':checks,
            'limitations':['300-customer seeded exploratory cohort, not the full population',
                           'Future product overlap is a proxy for utility, not a human relevance label',
                           'Community assignment and thresholds need temporal stability and sensitivity checks',
                           'No causal claim that a community causes future purchases']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--xlsx',type=Path,default=DEFAULT_XLSX)
    parser.add_argument('--output',type=Path,default=ROOT/'results'/'community_ablation.json')
    args = parser.parse_args()
    report = evaluate(args.xlsx)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({'sample':report['sample'],
                      'communityMethod':{k:v for k,v in report['communityMethod'].items() if k!='assignments'},
                      'metrics':report['evaluation']['metrics'],'checks':report['checks']},indent=2))
