# Challenge 4: community-aware GraphRAG research prototype

## New Human–AI application and measured finding (November 1)

This extends the **existing local Challenge 3 Neo4j GraphRAG**, preserving its graph and earlier community run. The new experiment uses [NetworkX weighted Louvain](https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.community.louvain.louvain_communities.html) on the same 300-customer, 1,684-edge historical projection. It found **54 communities** with weighted modularity **0.364886**. These are exploratory descriptions of observed purchasing overlap, not verified personal preferences.

Two examples show domain meaning. Community 12407 has 68 sampled customers; 40 purchased `REGENCY CAKESTAND 3 TIER` (group prevalence 58.8%, about 1.94 times its cohort prevalence). Community 12415 has 62 customers; 38 purchased `PARTY BUNTING` (61.3%, about 1.66 times cohort prevalence). The source invoice and worksheet row for each reported product are saved in `results/community_ablation.json`. These patterns describe the historical sample only.

### Temporal evaluation and ablation

All graph edges, communities, and candidates use purchases from **March 5 through August 31, 2011**. The subsequent **September 1 through October 31, 2011** purchases are held out exclusively for evaluation. For each of the **223** sampled customers who bought at least one new-to-them product during follow-up, the experiment ranks up to 10 candidate products. `global_graph` retrieves the top 5 historical similarity neighbors across communities; `community_graph` restricts them to the same Louvain group. A pre-cutoff popularity ranking is a simpler comparison. The outcome is overlap with each target's future new products; it is a proxy, not a judged relevance label.

| Method | Micro recall@10 | Micro precision@10 | Hit rate@10 | Candidate coverage |
|---|---:|---:|---:|---:|
| Popularity | 0.015955 | 0.044395 | 0.282511 | 1.000000 |
| Original global graph | **0.023046** | **0.075224** | **0.327354** | 0.860987 |
| Louvain community filter | 0.019662 | 0.064177 | 0.309417 | 0.860987 |

Community filtering **did not improve this retrieval outcome**: it lowered micro recall@10 by 0.003384 relative to global graph retrieval. The data support showing communities as an optional explanation/exploration lens, while keeping unrestricted graph retrieval as the default for discovery. No claim of statistical significance or causal business impact is made. The cohort is small, the thresholds are fixed, and the community partition has not been tested for temporal stability.

### Local application

`app.py` accepts a natural-language customer question with a five-digit ID. The human adjusts a historical Jaccard threshold and peer count, compares original and community-filtered Neo4j retrieval, inspects a customer relationship graph and transaction-level source rows, and accepts/rejects/corrects the explanation for the current browser session. The local `qwen2.5:1.5b` model selects observed product examples from retrieved evidence; Python and Cypher retain control over all scores, IDs, and citation paths. An invalid model response is withheld. This interactive, generated evidence story is the GenAI presentation component. The content is factual, and a separate limitation note states what the observations cannot prove.

The local app is **not** the separately deployed Challenge 2 Streamlit Cloud app. To reproduce with your existing local Neo4j and Ollama:

```powershell
pip install -r challenge4/requirements.txt
python challenge4/community_experiment.py --xlsx path/to/online_retail_II.xlsx
python challenge4/import_louvain.py --credentials path/to/private-neo4j-credentials.json
python challenge4/test_community_rag.py --credentials path/to/private-neo4j-credentials.json
$env:RETAIL360_NEO4J_CREDENTIALS='path/to/private-neo4j-credentials.json'
streamlit run challenge4/app.py --server.port 8504
```

Download the public workbook from [UCI Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii) and first prepare the Challenge 3/earlier Challenge 4 KG as described below. The credentials JSON contains `uri`, `username`, `password` and must remain outside Git. The importer creates `LouvainCommunity` nodes and `CH4_LOUVAIN_MEMBER_OF` edges; the original `Customer`, `Product`, `CH4_SIMILAR`, and `CH4_BOUGHT` evidence paths remain intact. It added **300 memberships and 54 communities** in the local test. Eight live Neo4j checks passed, including both retrieval modes, changed rankings, exact scores, source rows, and no-evidence handling. A Streamlit end-to-end test also rendered the graph, evidence table, and validated local-model explanation.

The next iteration should test multiple cutoff windows, cohort sizes, and similarity thresholds; compare stable community descriptions with user judgments; and study whether a community view improves analyst comprehension or decision quality. GNNs are optional and are not used in this Challenge 4 result.

### Source and method references

- [UCI Online Retail II dataset](https://archive.ics.uci.edu/dataset/502/online+retail+ii) — original dated transactions.
- [NetworkX Louvain documentation](https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.community.louvain.louvain_communities.html) — weighted, seeded community algorithm.
- [Neo4j GraphRAG for Python](https://neo4j.com/docs/neo4j-graphrag-python/current/) — reference architecture; this repo retains its existing reviewed Cypher templates rather than adding a new framework.

## Prior AI Design preview

The following sections document the October 29 preview. Its first-phase optimizer and results are retained for comparison; the weighted Louvain implementation and evaluation above supersede it.

This folder contains a small AI Design experiment, based on the Challenge 4 Human Design, **and a local Neo4j GraphRAG integration**. The user-defined priorities are customer communities based on similar purchased products, a customer relationship graph, and evidence-backed descriptions of observed buying tendencies. Codex proposed a larger deterministic cohort, a weighted customer projection, a simple community prototype, a comparison chart, and a provenance-first retrieval card.

The integration adds community and purchase-evidence relationships to the existing local Challenge 3 Neo4j database, preserving its original nodes and relationships. It extends the existing local GraphRAG question router and Streamlit app. It does **not** change the separate deployed Challenge 2 cloud application or implement full Louvain or Leiden.

## Data and reproducibility

Download the public [UCI Online Retail II workbook](https://archive.ics.uci.edu/dataset/502/online+retail+ii). Run:

```powershell
pip install openpyxl
python challenge4/ai_design_demo.py --xlsx path/to/online_retail_II.xlsx
```

The default workbook path is `../data/online_retail_ii/online_retail_II.xlsx` relative to the repository. The script reads the `Year 2010-2011` worksheet and keeps valid positive purchases in `[2011-03-05, 2011-09-01)`. It excludes cancellations, missing customer/product IDs, and nonpositive quantity/price. Eligible customers have at least three invoices and eight distinct products. A seeded random sample of 300 eligible customers (seed 5588) forms the exploratory cohort. No individual purchase rows or the full workbook are committed. The script saves only aggregate and example evidence in `results/ai_design_demo.json`.

For each pair of selected customers, a link requires at least three shared distinct products and Jaccard similarity of at least 0.10. A first-phase local modularity optimizer assigns exploratory groups. This has no graph coarsening and is **not** full Louvain or Leiden. Thresholds and cohort selection are design choices that need sensitivity analysis before using groups for decisions.

## Actual preliminary result

The selected window had 155,362 valid rows from 2,772 customers, of whom 888 met the eligibility rule. The 300-customer projection has 1,684 similarity edges. The local optimizer produced 68 groups, including 17 of size three or more. The three largest have 54, 42, and 34 customers. Four arithmetic/provenance checks pass.

For a concrete community-aware retrieval question, the sample result asks which members of customer 15719's 54-customer group bought similar products and why. Its top same-group peer, 15356, shares 64 distinct products (Jaccard 0.218430). Example evidence includes StockCode 20713 in invoice 547359, source row `Year 2010-2011:128739`, and invoice 557233, source row `Year 2010-2011:231094`. The JSON includes two further peers and more source rows.

These are observed purchases, not direct measures of customer preference. A large group can contain weak edges; group stability, coverage, and usefulness for retrieval have not yet been evaluated.

## Local Neo4j + GraphRAG integration

Start the same local Neo4j used by Challenge 3. Download the workbook above, then run:

```powershell
python challenge4/community_neo4j.py --credentials path/to/private-neo4j-credentials.json --xlsx path/to/online_retail_II.xlsx
python challenge4/evaluate_community.py --credentials path/to/private-neo4j-credentials.json
$env:RETAIL360_NEO4J_CREDENTIALS='path/to/private-neo4j-credentials.json'
streamlit run challenge3/graphrag_app.py
python challenge4/test_streamlit_ui.py
```

The credentials JSON contains `uri`, `username`, and `password`; keep it outside Git. The importer adds 300 Customer memberships, 68 Community nodes, 23,702 `CH4_BOUGHT` relationships with source invoice and worksheet row, and 1,684 `CH4_SIMILAR` relationships. It preserves existing Challenge 3 data. Import results are in `results/neo4j_community_import.json`.

Ask: **“Which customers in customer 15719's community bought similar products, and why?”** The original GraphRAG entry point now uses a parameterized multi-hop Cypher query, finds same-community peers, retrieves shared Product nodes and source purchases, computes an exact response, and asks the local Qwen model to select relevant evidence descriptions. The Streamlit page shows a relationship preview, exact facts, model interpretation, and original row IDs. The integration evaluation in `results/neo4j_community_evaluation.json` passed 8/8 checks, including matching the offline top peer/score, source evidence, missing-customer handling, an old-route regression, and local-model evidence selection.

This is a **local prototype**. It is not deployed to Streamlit Cloud. Community stability and usefulness for downstream marketing decisions remain untested. The next research step is to compare community algorithms and cutoff windows, then evaluate whether filtering by community improves retrieval and decisions.

## Research and comparison

- [UCI Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii) supplies the transaction fields and source workbook.
- [Neo4j Leiden documentation](https://neo4j.com/docs/graph-data-science/current/algorithms/leiden/) motivates testing a full community algorithm on the weighted projection later.
- [Neo4j Label Propagation documentation](https://neo4j.com/docs/graph-data-science/current/algorithms/label-propagation/) provides a simpler alternative.
- [Neo4j GraphRAG for Python](https://neo4j.com/docs/neo4j-graphrag-python/current/) describes graph retrieval followed by grounded generation. The local integration uses the existing Challenge 3 retrieval and LLM answer path with a new community query route.

Human Design selected the community meaning and graph output. AI Design added a reproducible cohort, explicit similarity and source-row rules, and a community-size chart. It rejected speech/video for this analyst workflow and does not infer hidden preferences from purchase records. Human review should now assess whether these additions match the original project goals.
