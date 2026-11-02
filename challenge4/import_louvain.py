"""Add weighted Louvain memberships to the existing Neo4j KG without replacing it."""
import argparse
from collections import Counter
import json
from pathlib import Path

from neo4j import GraphDatabase

ROOT = Path(__file__).resolve().parent
RUN_ID = 'uci-retail-ii-20110305-20110901-louvain-seed5588'


def import_assignments(conf, report):
    groups = report['communityMethod']['assignments']
    sizes = Counter(groups.values())
    rows = [{'customer':cid,'group':gid,'key':f'{RUN_ID}|{gid}',
             'size':sizes[gid]} for cid,gid in sorted(groups.items())]
    with GraphDatabase.driver(conf['uri'],auth=(conf['username'],conf['password'])) as driver:
        driver.verify_connectivity()
        with driver.session(database='neo4j') as db:
            original = db.run('MATCH ()-[s:CH4_SIMILAR {runId:$run}]->() RETURN count(s) AS n',
                              run='uci-retail-ii-20110305-20110901-seed5588').single()['n']
            if original != report['sample']['similarityEdges']:
                raise ValueError('Existing Challenge 4 similarity graph does not match experiment.')
            db.run('CREATE CONSTRAINT ch4_louvain_key IF NOT EXISTS FOR (g:LouvainCommunity) REQUIRE g.key IS UNIQUE').consume()
            # Idempotent import: memberships are tied to a fixed, versioned run.
            for index in range(0,len(rows),100):
                db.run('''
                UNWIND $rows AS r
                MATCH (c:Customer {customerId:r.customer})
                MERGE (g:LouvainCommunity {key:r.key})
                SET g.runId=$run,g.groupId=r.group,g.size=r.size,
                    g.method='NetworkX weighted Louvain',
                    g.windowStart='2011-03-05',g.windowEndExclusive='2011-09-01'
                MERGE (c)-[:CH4_LOUVAIN_MEMBER_OF {runId:$run}]->(g)
                ''',rows=rows[index:index+100],run=RUN_ID).consume()
            result=db.run('''
                MATCH (c:Customer)-[:CH4_LOUVAIN_MEMBER_OF {runId:$run}]->(g:LouvainCommunity)
                RETURN count(DISTINCT c) AS customers,count(DISTINCT g) AS communities
                ''',run=RUN_ID).single().data()
            assert result == {'customers':len(rows),'communities':len(sizes)},result
    return {'runId':RUN_ID,'counts':result,'originalSimilarityEdges':original,
            'note':'Membership added to existing Neo4j; old KG and community run preserved.'}


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--credentials',type=Path,required=True)
    p.add_argument('--report',type=Path,default=ROOT/'results'/'community_ablation.json')
    args=p.parse_args()
    conf=json.loads(args.credentials.read_text(encoding='utf-8'))
    report=json.loads(args.report.read_text(encoding='utf-8'))
    result=import_assignments(conf,report)
    output=ROOT/'results'/'louvain_neo4j_import.json'
    output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))
