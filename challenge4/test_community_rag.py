"""Focused live checks for the Challenge 4 local application path."""
import argparse
import json
from pathlib import Path

from community_rag import credentials, run_question

ROOT=Path(__file__).resolve().parent


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--credentials',type=Path,required=True)
    args=p.parse_args()
    conf=credentials(args.credentials)
    result=run_question('Which customers resemble customer 15719?',conf,limit=10)
    global_peers=[p['peer'] for p in result['global']['peers']]
    local_peers=[p['peer'] for p in result['community']['peers']]
    missing=run_question('customer 99999',conf)
    high=run_question('customer 15719',conf,minimum=0.5)
    checks={
        'real_neo4j_two_modes':result['global']['status']=='ok' and result['community']['status']=='ok',
        'community_peers_meet_similarity_rule':all(p['jaccard']>=0.1 and p['sharedProductCount']>=3 for p in result['community']['peers']),
        'modes_can_change_ranking':global_peers!=local_peers,
        'top_peer_and_exact_score':global_peers[0]=='15356' and abs(result['global']['peers'][0]['jaccard']-0.218430)<1e-6,
        'same_community_only':all(p['community']==p['peerCommunity'] for p in result['community']['peers']),
        'source_rows_for_each_peer':all(e['rowA'] and e['rowB'] and e['invoiceA'] and e['invoiceB'] for e in result['community']['evidence']),
        'missing_customer_fallback':missing['global']['status']=='no_evidence' and missing['community']['status']=='no_evidence',
        'high_threshold_fallback':high['community']['status']=='no_evidence',
    }
    assert all(checks.values()),checks
    report={'checks':checks,'passed':sum(checks.values()),'total':len(checks),
            'question':result['question'],'globalPeers':global_peers,
            'communityPeers':local_peers,'exampleEvidence':result['community']['evidence'][:2],
            'limitations':['LLM text is not evaluated by this script','No user-study measure of explanatory value']}
    output=ROOT/'results'/'community_app_tests.json'
    output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({'passed':report['passed'],'total':report['total'],
                      'globalPeers':global_peers,'communityPeers':local_peers},indent=2))


if __name__=='__main__':main()
