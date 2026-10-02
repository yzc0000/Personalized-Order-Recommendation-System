# Grocery recommendation implementation and results

This is an offline next-basket study using the six supplied Instacart tables. Its primary task is a probability for each previously bought customer-product pair, followed by a ranked Top-K list. New-to-customer discovery is the additional objective requested during the project.

## Current decision

The `recommend` and `predict-reorders` commands use the **all-labeled deployment fit** of the XGBoost reorder system for known products. `recommend` accepts Repeat suggestions above the **0.5 probability cutoff**, then appends at most two labeled Discovery items from the separate specialist within the five-position limit. Discovery scores are not shown as reorder probabilities. `predict-reorders` returns repeat decisions alone. The evaluated repeat component's precision among shown products was **0.639**, with 1.46 repeat products per customer and 40.9% empty repeat panels. These are repeat-only measurements; the runtime deployment combination has no held-out test score.

The separate evaluated repeat fit's fixed-list Precision@5 is **0.3876**, versus **0.3422** for purchase frequency, meaning 1.94 matches per five possible slots. An answer-seeing oracle restricted to the same repeat candidates reaches **0.7139**, so missing new products explain only part of the gap. The threshold tradeoff is in the [Project 1 final report](PROJECT_1_FINAL.md#confidence-cutoff-for-the-recommendation-display); matched pilot results for the bounded hybrid appear below.

The new experiments below produced useful diagnostics and a runnable new-item preview. **No challenger for the primary five-slot panel** met the prespecified gate of at least **+0.005 absolute validation Precision@5** with a positive paired customer-bootstrap interval. A separate model *did* meet that gate for ranking the new-only preview, although its absolute hit rate remains low. Offline purchase matching cannot establish sales lift or whether showing a recommendation changes what a customer buys.

## Protocol and data

- **Decision time:** before the next order starts. Only completed orders and supplied product metadata are available. Next-order day, hour, elapsed days and contents are excluded from features.
- **Population statistics:** the original 22-feature representation uses product counts and reorder rates from the available prior-order corpus, including histories of validation/test customers but no hidden next-order labels. This is a transductive offline setting, not a strict calendar-time deployment backtest. The added product-prior ablation is fitted only on fit customers.
- **Development users:** sample 10,000 customers with seed 2026 from the *training portion* of the original full-data split. The new split is 7,000 fit, 1,500 validation and 1,500 pilot test customers. The original full-data test users are outside these pilots. Saved user IDs are in [the model benchmark artifacts](../artifacts/benchmark_10k_standard/).
- **Selection:** one fixed validation comparison per method; only the chosen method is refit on fit plus validation customers before scoring the pilot test. Differences use paired customer bootstrap intervals. Candidate metrics include the complete generated set and missed next-basket items; they do not sample easy negatives.
- **Data quality:** `validate_bundle` checks order/product keys, duplicate lines and history-target relationships. The full-data [EDA summary](figures/eda.json) covers 131,209 labeled next orders and 3,214,874 prior orders. Every first order lacks a days-since-prior-order value by definition. About 10.2% of later prior orders have the capped value 30, which limits fine-grained cadence modeling.

The original full-data test was inspected in earlier project work. Its 0.3876 is an established historical benchmark, not an untouched confirmation set. The 1,500 pilot-test customers were reused across these ablations and their outcomes have now been inspected too; these scores are development evidence, not an independent final confirmation. The 10,000-customer pilot also uses different people and less training data, so its raw test score should not be compared as if it were a direct full-model improvement or regression. A future promotion needs new unseen outcomes or a fresh prespecified outer-fold evaluation.

## Exploratory analysis and explanations

The median labeled customer has **9** completed orders; the median prior basket has **8** products and the median labeled next basket **9**. **23.2%** of next baskets contain fewer than five products, so a fixed five-slot panel cannot have five hits for those customers. **40.1%** of labeled next-basket product rows are new to their customer. The top **1%** of catalog products account for **42.8%** of prior purchases. See [EDA distributions](figures/eda_distributions.png) and [machine-readable counts](figures/eda.json).

The tree explanation report uses XGBoost's exact per-tree feature contributions on 3,000 sampled candidate rows from pilot test customers. It verifies that contributions sum to raw log odds and that transforming these margins recovers saved probabilities. The largest mean absolute contributions were days since last purchase (**0.338** log-odds units), purchases in the last five orders (**0.237**), order gap (**0.234**), the product's order rate since its first purchase (**0.209**) and global product reorder rate (**0.200**). These explain fitted scores, not purchase causes. [Global explanation](figures/model_explanations.png) and [25 individual top-five explanations](figures/model_explanations.json) are reproducible with [portfolio_analysis.py](../experiments/portfolio_analysis.py).

The [original evaluated model's calibration plot](figures/score_calibration.png) compares predicted and observed reorder rates by score bin. A separate [pilot top-five calibration audit](figures/top_five_calibration.png) covers **7,447 displayed products** from 7,500 possible slots; [bin counts](figures/top_five_calibration.json) show mild overconfidence around predicted 0.6–0.8. The [display tradeoff plot](figures/display_tradeoff.png) uses decision-validation customers: a fixed five-list yields **0.383** Precision@5, while showing only candidates above 0.5 gives **0.641** precision among shown products, **1.46** shown per customer and **41.6%** empty panels. At 0.6, precision among shown is **0.706**, with **0.87** shown per customer and **58.7%** empty panels. The higher selective precision buys fewer matches and lower coverage; it is not a better ranking model.

## Comparable model benchmark

All five models below use the **same 22 history-only numeric features**, the same 7,000 fit customers, 1,500 validation customers, candidate products and timing. Tree models have fixed settings; this is a controlled comparison rather than exhaustive tuning. See [benchmark code](../experiments/benchmark_models.py) and [metrics](../artifacts/benchmark_10k_standard/metrics.json).

| Model | AP / PR-AUC | Precision at 0.5 | Recall at 0.5 | F1 at 0.5 | Validation P@5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Purchase frequency | 0.3171 | 0.4485 | 0.2604 | 0.3295 | 0.3401 |
| Logistic regression | 0.4044 | 0.6308 | 0.1730 | 0.2716 | 0.3804 |
| Random forest | 0.4067 | 0.6333 | 0.1684 | 0.2661 | 0.3805 |
| XGBoost classifier | 0.4121 | 0.6222 | 0.1848 | 0.2849 | **0.3848** |
| CatBoost classifier | **0.4139** | 0.6301 | 0.1804 | 0.2804 | 0.3829 |

AP is sklearn average precision, used as the PR-AUC summary. The candidate precision, recall and F1 columns use a fixed 0.5 score cutoff rather than a model-specific optimized cutoff. Purchase frequency is a historical rate score, not a calibrated next-order probability, so its cutoff metrics need extra caution. CatBoost led AP; XGBoost led the selection metric, P@5.

XGBoost won validation and was refit on 8,500 pilot customers. Its separate pilot-test Precision@5 was **0.3745** versus **0.3340** for purchase frequency, a gain of about 0.20 matching products per customer; the paired 95% interval for absolute Precision@5 gain was **[0.0329, 0.0484]**. The result supports a learned reorder scorer over simple frequency; the small gap between XGBoost and CatBoost does not establish that one model family is universally better. The original full-data model still has the stronger direct evidence for the main CLI.

## Feature and learning ablations

The [enriched benchmark](../artifacts/benchmark_10k_enriched/metrics.json) adds four history-only replenishment features and four fit-customer product priors. The priors count next-order transitions and one-time buyers with follow-up exposure; validation/test target baskets are excluded. CatBoost additionally treats product, aisle and department IDs as categories, never arbitrary continuous numbers.

| Same 10,000-customer pilot unless stated | Baseline validation P@5 | Variant validation P@5 | Decision |
| --- | ---: | ---: | --- |
| Standard XGBoost vs enriched XGBoost | 0.3848 | 0.3839 | No measured gain |
| Standard XGBoost vs enriched CatBoost with identity | 0.3848 | 0.3833 | No measured gain |
| Standard XGBoost vs XGBoost plus customer-neighbor score | 0.3848 | 0.3817 | No measured gain; paired 95% interval **[-0.0064, 0.0007]** |
| 5,000-customer history-safe XGBoost vs same with three earlier targets/customer | 0.3712 | 0.3717 | No measured gain; paired 95% interval **[-0.0045, 0.0056]** |

A bounded [ranking-objective comparison](../artifacts/ranking_objective_10k/metrics.json) used the same 10,000-customer pilot and 22 features. Ordinary XGBoost scored **0.3848** validation Precision@5; equal-total-weight-per-customer classification scored **0.3833** (paired interval for gain **[-0.0049, 0.0024]**); an XGBoost NDCG@5 ranker scored **0.3801** (interval **[-0.0085, -0.0004]**). The ranker score is not a purchase probability. This tested configuration is worse for the five-slot task, so the classifier remains selected. It does not rule out every ranking objective or parameter setting.

The [time-weighted customer-neighbor scorer](../artifacts/temporal_neighbor_10k/metrics.json) is a TIFU-KNN-inspired, sparse, leave-one-customer-out baseline. It scores a customer's known products using recent personal frequencies and similar customers' frequencies. As a standalone repeat ranker it reached **0.3705 validation P@5**, ahead of purchase frequency **0.3401** but behind XGBoost **0.3848** on the same people. Adding its score as a classifier feature did not help.

The [historical-prefix experiment](../artifacts/historical_prefix_5k/metrics.json) expands 230,337 current-target training pairs to **821,096** including 590,759 earlier-target pairs. Each pseudo-target sees only preceding completed orders. Global product statistics are excluded because the original aggregates include events after historical cutoffs. The ranker helper now supports `(user, snapshot)` query grouping, although this ablation uses a classifier. The additional targets did not deliver a material gain.

These ablations test **specific feature bundles and settings**. They do not prove that all replenishment, identity, collaborative or augmentation approaches are ineffective.

## New-product discovery

The [retrieval experiment](../experiments/discovery_retrieval.py) fits three behavioral sources from fit customers' completed histories—a time-weighted similar-customer index, a sparse co-basket graph and unseen-product popularity—and one content source from catalog names/categories. The basket graph has **469,451 directed sparse links** and can address the 49,688-product catalog without a dense product-product matrix. Each source returns up to 40 unseen products/customer.

| Validation source | Novel-item recall at up to 40/customer |
| --- | ---: |
| Similar customers | 10.03% |
| Sparse co-basket | 7.84% |
| Name/category TF-IDF | 0.87% |
| Unseen popularity | 8.99% |

At an 80-candidate budget, the initial three-source union without TF-IDF reached **13.14%** validation novel recall versus **11.72%** with all four sources. A fairer 80-candidate popularity baseline reached **13.45%**, exposing that the initial union left some slots unfilled. Extending its popularity source to fill those slots raised the three-source union to **13.94%** validation novel recall. This selected filled union reached **14.16%** on the pilot test (**900 of 6,354** actual new product purchases), versus **13.61%** (**865 hits**) for popularity alone at the same 80-candidate limit. The difference is 35 purchases across 1,500 customers and has not been established as a robust gain. The selected pool yields an answer-seeing candidate-restricted Precision@5 ceiling of **0.7468**. Exact source overlap, all budgets and test results are in [retrieval metrics](../artifacts/discovery_10k_ablation/metrics.json).

The [mixed ranker](../experiments/discovery_ranker.py) then trained on the selected filled pool of 80 new-product candidates plus all known-product candidates. On the same 1,500 validation customers, its Precision@5 was **0.3832** versus **0.3848** for a repeat-only XGBoost baseline; the paired 95% interval for the difference was **[-0.0055, 0.0021]**. It failed the prespecified promotion gate. The refit repeat-only pilot scored **0.3745** on its test customers, with **zero** new products shown by design. See [filled-pool mixed-ranker metrics](../artifacts/discovery_ranker_filled_10k/metrics.json). The earlier unfilled pool gave the same validation Precision@5 and also failed its gate.

The new-only preview has a **separate XGBoost binary classifier used for ranking** trained only on retrieved new-product candidates, using history-derived product/category and retrieval-source signals. On validation customers, it raised new-only Precision@5 from **0.0193** for source-round-robin ordering to **0.0248**; its paired 95% interval for absolute gain was **[0.0027, 0.0083]**, meeting the preview promotion gate. After refitting on fit plus validation customers, it matched **181** held-out new purchases across **7,500** possible display slots: **0.0241 new-only Precision@5**, compared with **158** hits and **0.0211** for source ordering. The pilot-test gain (**+0.0031**) was smaller than the validation gain (**+0.0055**). See [the ranker report](../artifacts/discovery_ranker_filled_10k/metrics.json) and [saved novel test lists](../artifacts/discovery_ranker_filled_10k/test_novel_preview.csv).

`recommend-discovery-preview` now uses that saved specialist model when available and shows its source for each item. Its rank is **not** a calibrated purchase probability. This remains too weak to reserve positions in the main five-list; retrieval still misses about 86% of actual novel purchases at the selected budget. The 1,500 pilot-test users have been repeatedly observed during development, so the 0.0241 result is supporting evidence, not independent confirmation.

### Product-grouping experiment

The [grouping experiment](../experiments/product_grouping.py) builds an undirected weighted graph from the training customers' co-basket links, clusters it with Louvain, then uses groups represented in each validation customer's last basket to retrieve unseen products. Candidates within a seeded group are ordered by group support and training-set product popularity. The split matches the 10,000-customer discovery pilot: 7,000 fit users and 1,500 validation users, with an 80-candidate budget.

| Validation retriever | New candidates found | Novel-item recall@80 | Raw group-score new-only P@5 |
| --- | ---: | ---: | ---: |
| Louvain product groups | 873 / 6,649 | 13.13% | 1.64% |
| Current neighbor + co-basket + popularity pool | 927 / 6,649 | **13.94%** | — |
| Popularity alone | 894 / 6,649 | 13.45% | — |

The graph was very fragmented: **34,848 products had no retained co-basket edge**, while the linked products formed 48 multi-product communities; the largest contained 5,044 products. The group retriever also returned fewer than 80 candidates for some users. Its group-score top five found 123 new purchases in 7,500 slots. This is a retrieval and hand-scored ranking test; no group-specific XGBoost scorer was trained. The result does not support replacing the current candidate pool with Louvain groups. A trained scorer or adding groups as a supplementary retrieval source would need a separate validation comparison. Full counts and assignments are in [grouping metrics](../artifacts/product_grouping_10k/metrics.json); rerun with `python -m experiments.product_grouping --max-users 10000 --budget 80`.

### Repeat-first hybrid display

The [hybrid display analysis](../experiments/hybrid_display_analysis.py) evaluates a policy that keeps up to five repeat suggestions above the 0.5 probability cutoff, then fills remaining positions with the saved new-item specialist's ranked suggestions. It uses matched predictions from the same 1,500 pilot-test customers and their 350-round models; these results do not evaluate the full-data deployment model. The pilot-test split has already been inspected during development.

The selected runtime policy is **at most two Discovery suggestions** in available positions. [hybrid.py](../grocery_recommender/hybrid.py) applies this same policy to the CLI and the saved-prediction analysis. Repeat suggestions appear first, discovery suggestions exclude all known products, and the probability field is empty for Discovery rows. `recommend --max-discovery 0` provides repeat suggestions alone.

| Display policy | Shown/customer | Precision among shown | Total matching purchases | Matching new purchases |
| --- | ---: | ---: | ---: | ---: |
| Repeat top five, no cutoff | 4.96 | 37.72% | 2,809 | 0 |
| Repeat cutoff 0.5 only | 1.50 | 61.47% | 1,385 | 0 |
| Repeat cutoff + fill all empty slots with new items | 5.00 | 20.31% | 1,523 | 138 |
| Repeat cutoff + at most two new items | 3.24 | 30.18% | 1,465 | 80 |

Filling all slots preserved all 1,385 accepted repeat matches and added 138 new matches. Repeat precision stayed at 61.47%, while the added 5,247 new-item suggestions had 2.63% precision. Total matches increased by 9.96% over the cutoff-only panel; the increase in display volume lowered overall precision. On average only 1.50 repeats passed the cutoff, so filling all five positions meant adding 3.50 new products, rather than two. The two scores were not compared or calibrated together. These purchase-matching results do not establish whether displaying a new item causes a purchase. Counts and combined lists are in [hybrid metrics](../artifacts/hybrid_display_10k/metrics.json).

## Reproduce

From the repository root with the six source CSVs present:

```powershell
python -m pip install -e ".[analysis]"
python -m unittest discover -s tests -v
python -m experiments.benchmark_models --variant standard --max-users 10000 --rounds 350 --output-dir artifacts/benchmark_10k_standard
python -m experiments.benchmark_models --variant enriched --max-users 10000 --rounds 350 --output-dir artifacts/benchmark_10k_enriched
python -m experiments.discovery_retrieval --max-users 10000 --per-source 40 --budget 80 --output-dir artifacts/discovery_10k_ablation
python -m experiments.temporal_neighbor_benchmark --retrieval-dir artifacts/discovery_10k_ablation
python -m experiments.collaborative_feature_benchmark --retrieval-dir artifacts/discovery_10k_ablation
python -m experiments.historical_prefix_benchmark --max-users 5000 --snapshots 3 --rounds 350
python -m experiments.ranking_objective_benchmark --max-users 10000 --rounds 350
python -m experiments.discovery_ranker --retrieval-dir artifacts/discovery_10k_ablation --output-dir artifacts/discovery_ranker_filled_10k --max-users 10000 --rounds 350
python -m experiments.portfolio_analysis --model-dir artifacts/benchmark_10k_standard
python -m grocery_recommender recommend --user-id 1
python -m grocery_recommender recommend --user-id 1 --min-probability 0.75
python -m experiments.threshold_display_sweep
python -m grocery_recommender recommend-discovery-preview --user-id 1
```

Saved `joblib` files should only be loaded from trusted local experiment outputs. Raw CSVs and model artifacts are intentionally ignored by Git. Exact calendar dates, prices, stock, promotions, customer exposure and purchase responses to recommendations are absent. A future online test would need recommendation impressions and randomized customer-level assignment before claiming business lift.
