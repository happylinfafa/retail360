"""Evidence-first comparison of original vs weighted-Louvain GraphRAG retrieval."""
from __future__ import annotations

import json
import re
from pathlib import Path

from neo4j import GraphDatabase

from import_louvain import RUN_ID

ORIGINAL_RUN = 'uci-retail-ii-20110305-20110901-seed5588'
PEERS = '''
MATCH (c:Customer {customerId:$customer})-[:CH4_LOUVAIN_MEMBER_OF {runId:$louvain}]->(g:LouvainCommunity)
MATCH (c)-[s:CH4_SIMILAR {runId:$original}]-(p:Customer)
OPTIONAL MATCH (p)-[:CH4_LOUVAIN_MEMBER_OF {runId:$louvain}]->(pg:LouvainCommunity)
WITH g,p,pg,s
WHERE s.jaccard >= $minimum AND ($mode='global' OR pg.groupId=g.groupId)
RETURN g.groupId AS community,g.size AS communitySize,
       p.customerId AS peer,pg.groupId AS peerCommunity,
       s.jaccard AS jaccard,s.sharedProductCount AS sharedProductCount
ORDER BY jaccard DESC,peer ASC LIMIT $limit
'''
EVIDENCE = '''
MATCH (a:Customer {customerId:$customer})-[x:CH4_BOUGHT {runId:$original}]->(p:Product)
      <-[y:CH4_BOUGHT {runId:$original}]-(b:Customer {customerId:$peer})
RETURN p.stockCode AS product,p.description AS description,
       x.invoiceId AS invoiceA,x.sourceRow AS rowA,
       y.invoiceId AS invoiceB,y.sourceRow AS rowB,
       x.sourceFile AS sourceFile
ORDER BY product LIMIT 4
'''


def customer_id(question):
    ids=re.findall(r'\b\d{5}\b',question)
    if not ids:
        raise ValueError('Please include a five-digit customer ID.')
    return ids[0]


def retrieve(db,customer,mode='community',minimum=0.1,limit=5):
    if mode not in ('global','community') or not (0.1 <= minimum <= 1) or not 1 <= limit <= 10:
        raise ValueError('Invalid retrieval parameters.')
    peers=db.run(PEERS,customer=customer,louvain=RUN_ID,original=ORIGINAL_RUN,
                 mode=mode,minimum=minimum,limit=limit).data()
    evidence=[]
    for peer in peers:
        rows=db.run(EVIDENCE,customer=customer,peer=peer['peer'],original=ORIGINAL_RUN).data()
        for row in rows:
            row['peer']=peer['peer']
            row['jaccard']=peer['jaccard']
        evidence.extend(rows)
    if not peers:
        return {'status':'no_evidence','mode':mode,'customer':customer,'peers':[],
                'evidence':[],'answer':'No eligible similar customers at this threshold; change the ID or filter.'}
    top=peers[0]
    answer=(f"Customer {customer}: {len(peers)} retrieved peers; weighted-Louvain group "
            f"{top['community']} has {top['communitySize']} customers in the sampled graph. "
            f"Top peer {top['peer']} shares {top['sharedProductCount']} distinct products "
            f"(historical Jaccard {top['jaccard']:.3f}). "
            'These are observed purchases, not proven preferences or future outcomes.')
    return {'status':'ok' if evidence else 'no_evidence','mode':mode,'customer':customer,
            'peers':peers,'evidence':evidence,'answer':answer,
            'uncertainty':'Exploratory 300-customer sample. Group membership depends on the historical window, graph thresholds, and Louvain seed.'}


def run_question(question,conf,minimum=0.1,limit=5):
    cid=customer_id(question)
    with GraphDatabase.driver(conf['uri'],auth=(conf['username'],conf['password'])) as driver:
        driver.verify_connectivity()
        with driver.session(database='neo4j') as db:
            return {'question':question,'customer':cid,
                    'global':retrieve(db,cid,'global',minimum,limit),
                    'community':retrieve(db,cid,'community',minimum,limit)}


def credentials(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))
