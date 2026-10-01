# Personalized Grocery Recommendation Engine

A personalized grocery recommendation system that combines **repeat-purchase prediction** with **new-item discovery** using Instacart purchase histories.

The default output contains up to five clearly labeled suggestions:

- **Repeat:** previously purchased products with estimated next-order reorder probability **at least 0.5**.
- **Discovery:** at most **two products the customer has never purchased**, ranked by a separate new-item recommendation model and added when space remains.

The repeat model achieves **63.92% precision among displayed repeat suggestions** on the established test benchmark at the 0.5 cutoff. Discovery performance and combined-output metrics are reported separately below.

## How recommendations work

Repeat suggestions come first. Discovery suggestions occupy available positions without replacing accepted repeats. The list can contain fewer than five products.

| Accepted repeats | Maximum discoveries | Maximum total shown |
| ---: | ---: | ---: |
| 0 | 2 | 2 |
| 1 | 2 | 3 |
| 2 | 2 | 4 |
| 3 | 2 | 5 |
| 4 | 1 | 5 |
| 5 | 0 | 5 |

For example, the current deployment output for customer 3 is:

| Type | Product | Estimated reorder probability |
| --- | --- | ---: |
| Repeat | Vanilla Unsweetened Almond Milk | 0.741 |
| Repeat | Organic Avocado | 0.682 |
| Repeat | Organic Baby Spinach | 0.559 |
| Discovery | Banana | — |
| Discovery | Large Lemon | — |

Discovery scores are used for ranking and are not displayed as purchase probabilities. This example illustrates the output; customer 3 has no labeled next order for checking these suggestions.

## Methodology

### 1. Data and prediction target

The dataset contains **49,688 products**, **3,214,874 prior orders**, **32,434,489 prior product rows**, **131,209 labeled next orders**, and **75,000 unlabeled next orders**.

The decision point is **before the next order starts**. Only completed purchase histories and product metadata are available as inputs. Next-order contents, basket size, day, hour, and elapsed time are excluded.

For repeat prediction, each customer-product pair is a candidate only if that customer has previously bought the product. Its binary target is:

- **1:** the product appears anywhere in the customer's next order.
- **0:** the product does not appear in that order.

The five-position display limit controls how many suggestions are shown. The prediction horizon is **one complete next order**, regardless of its size. A purchase in a later order does not count as a next-order hit.

### 2. Repeat-purchase model

The selected model is an **XGBoost binary classifier**, with **600 boosting rounds**, maximum tree depth **6**, learning rate **0.05**, and **22 numeric features**.

| Feature family | Signals |
| --- | --- |
| Customer history | Number of orders, total items, unique products, average basket size, average time between orders |
| Customer-product behavior | Purchase count and share, reorder count, recency, recent-three/recent-five order purchases, cart position |
| Replenishment patterns | Typical purchase intervals, order rate since first purchase, elapsed interval relative to the typical interval |
| Product history | Global purchase count and reorder rate from completed prior orders |
| Category preferences | Customer purchase shares by aisle and department |

The classifier estimates the probability of each known product appearing in the next order. Candidates are ranked by this probability, and up to five with probability at least **0.5** are retained. The score is an estimate, not a guarantee for an individual customer.

Purchase frequency, logistic regression, random forest, and CatBoost were also evaluated as baselines. The controlled comparisons and feature experiments are documented in the [Project 1 report](reports/PROJECT_1_FINAL.md) and [implementation report](reports/IMPLEMENTATION_RESULTS.md).

### 3. New-item recommendation model

Discovery uses a separate retrieval and ranking pipeline:

1. **Similar customers:** represent purchase histories as sparse product vectors, give recent baskets more weight, and retrieve unseen products from the 50 nearest customers using cosine similarity.
2. **Related products:** retrieve unseen products associated with the customer's last completed basket through a graph of products purchased together.
3. **Popularity:** supplement the pool with popular unseen products. Combine the sources into at most **80 candidates per customer** and exclude all previously purchased products.
4. **Learned ranking:** use a separate **350-round XGBoost classifier** to rank these unseen candidates using customer preferences, product behavior, and retrieval scores, ranks, and support.
5. **Display:** append at most two ranked discoveries within the five-position limit. Discovery scores are not calibrated purchase probabilities, so the repeat model's 0.5 cutoff is not applied to them.

The discovery model and retrieval indices were developed on a **10,000-customer pilot**, rather than fitted on the entire labeled population. Product clustering with Louvain was also tested; it underperformed the selected retrieval pool and is documented as an experiment in the [grouping results](reports/IMPLEMENTATION_RESULTS.md#product-grouping-experiment).

## Results at the default 0.5 cutoff

### Repeat suggestions

These results evaluate **only repeat suggestions**, after applying both the five-position cap and the **0.5 probability cutoff**. Precision uses the number of products actually shown as its denominator.

| Metric | Decision validation | Established test |
| --- | ---: | ---: |
| Customers evaluated | 9,841 | 19,682 |
| Precision among displayed repeats | **64.06%** | **63.92%** |
| Recall of actual repeat products | 14.91% | 14.84% |
| F1 for repeat suggestions | 24.20% | 24.08% |
| Repeat suggestions shown | 14,389 | 28,734 |
| Suggestions matching the next order | 9,218 | 18,368 |
| Average repeats shown per customer | 1.46 | 1.46 |
| Customers receiving at least one repeat | 58.37% | 59.07% |
| Customers receiving no repeat | 41.63% | 40.93% |

**Precision** is matching repeat suggestions divided by displayed repeat suggestions. **Recall** is matching repeat suggestions divided by all actual repeat products in the next orders, aggregated across customers. **F1** is their harmonic mean. Customer coverage measures whether a customer receives at least one repeat suggestion.

The 0.5 cutoff improves precision by returning fewer products. The resulting 14.84% repeat recall and 40.93% of customers without an accepted repeat show the coverage tradeoff. The **63.92% figure is not the precision of the combined Repeat + Discovery list**.

Source: `artifacts/full_v2/threshold_display_metrics.json`, generated from the separate evaluated model. Raw artifacts are excluded from Git; the key results are recorded here and in the reports.

### New-item and combined-output results

The following comparison uses matched predictions for the **same 1,500 pilot-test customers**, with 350-round pilot models and a 0.5 repeat cutoff.

| Metric | Repeat suggestions only | Repeats + up to two discoveries |
| --- | ---: | ---: |
| Overall precision among displayed suggestions | 61.47% | **30.18%** |
| Precision of the repeat portion | 61.47% | 61.47% |
| Precision of the discovery portion | — | 3.08% |
| Matching repeat suggestions | 1,385 | 1,385 |
| Matching discovery suggestions | 0 | 80 |
| Total matching suggestions | 1,385 | **1,465** |
| Total suggestions shown | 2,253 | 4,854 |
| Average suggestions shown per customer | 1.50 | 3.24 |
| Average matching suggestions per customer | 0.92 | 0.98 |

Discovery added **80 next-order matches** while retaining every accepted repeat. Its low next-order purchase rate also reduced precision across the combined list. It is an experimental discovery component with weaker evidence than repeat prediction.

As a separate five-new-items benchmark, the discovery classifier achieved **2.41% Precision@5** (181 matches across 7,500 positions), compared with **2.11%** for retrieval ordering alone. This differs from the 3.08% discovery precision above because the combined policy shows at most two discoveries and only where space is available.

These pilot results **do not evaluate the full-data deployment repeat model combined with the discovery model**. Source: `artifacts/hybrid_display_10k/metrics.json`; see the [hybrid analysis](reports/IMPLEMENTATION_RESULTS.md#repeat-first-hybrid-display) for details.

### Other repeat probability cutoffs

The following table evaluates repeat suggestions only, with the same five-position cap:

| Minimum probability | Validation precision | Test precision | Test repeats shown/customer | Test customers with no repeat |
| ---: | ---: | ---: | ---: | ---: |
| **0.50 (default)** | **64.06%** | **63.92%** | **1.46** | **40.93%** |
| 0.60 | 70.62% | 70.88% | 0.86 | 58.75% |
| 0.75 | 81.77% | 81.42% | 0.24 | 84.94% |
| 0.90 | 95.35% | 93.83% | 0.008 | 99.34% |

At 0.9, only **162 repeat suggestions** were shown across 19,682 test customers. High precision at that cutoff comes with very limited coverage.

## Evaluation and training protocol

- **Split by customer:** seed 42 assigns 91,846 customers to training, 19,681 to validation, and 19,682 to test. A customer's candidate rows stay together.
- **Separate validation roles:** 9,840 validation customers select the number of boosting rounds; the remaining 9,841 evaluate decision cutoffs. The evaluated repeat model is refitted on training plus round-selection customers.
- **Deployment refit:** the deployment repeat model is fitted on all **131,209 labeled customers**. Reported test metrics come from `evaluated_model.joblib`, not this all-labeled fit in `model.joblib`.
- **Discovery pilot:** sample 10,000 customers from the original training population using seed 2026, then split into 7,000 fit, 1,500 validation, and 1,500 test customers. Retrieval indices use fit customers' prior histories; the selected classifier is refitted on the 8,500 fit-plus-validation customers before pilot-test scoring.
- **Historical benchmarks:** both test splits have subsequently been inspected during development. Their results are established offline benchmarks, not untouched confirmation sets for future changes.
- **Population statistics:** repeat features use global product statistics from the available prior-order corpus, including validation/test customers' histories but no hidden next-order labels. This is a transductive evaluation, rather than a strict calendar-time deployment backtest.

The separate `predict-reorders` command returns a variable-length reorder decision set with no five-item cap. It uses the validation-selected **0.36 cutoff that maximizes F0.5**. Its policy differs from the default recommendation display; the results highlighted above all use the **0.5 display cutoff**.

## Run locally

Use **Python 3.10+**. Place these six Instacart CSV files in the project root:

```text
orders.csv
order_products__prior.csv
order_products__train.csv
products.csv
aisles.csv
departments.csv
```

Install dependencies and train the repeat model:

```powershell
python -m pip install -e ".[analysis]"
python -m grocery_recommender audit --max-users 1000
python -m grocery_recommender train-full --batch-size 5000
```

Build the discovery indices and train the discovery classifier:

```powershell
python -m experiments.discovery_retrieval --max-users 10000 --per-source 40 --budget 80 --output-dir artifacts/discovery_10k_ablation
python -m experiments.discovery_ranker --retrieval-dir artifacts/discovery_10k_ablation --output-dir artifacts/discovery_ranker_filled_10k --max-users 10000 --rounds 350
```

Generate recommendations:

```powershell
# Default: repeats at probability >= 0.5, plus up to two discoveries
python -m grocery_recommender recommend --user-id 3

# Repeat suggestions only
python -m grocery_recommender recommend --user-id 3 --max-discovery 0

# Change the repeat cutoff
python -m grocery_recommender recommend --user-id 3 --min-probability 0.75

# Inspect the separate new-item list
python -m grocery_recommender recommend-discovery-preview --user-id 3
```

The default hybrid command requires saved discovery indices and the specialist classifier when discovery positions are available. Training outputs are saved under `artifacts/`; raw CSVs and generated artifacts are ignored by Git.

Run the tests and reproduce the display analyses:

```powershell
python -m unittest discover -s tests -v
python -m experiments.threshold_display_sweep
python -m experiments.hybrid_display_analysis
```

## Limitations and further documentation

About **40.4% of next-order products** in the established full-data test benchmark were new to the customer. Repeat candidates cannot cover those purchases, and discovery currently matches only a small fraction of them. A product can also be relevant to a customer without being purchased in the exact next order; this evaluation does not measure longer-term relevance.

The public dataset lacks exact calendar timestamps, prices, promotions, inventory, recommendation impressions, and live customer feedback. Days between orders are capped at 30, making elapsed-day features approximate. Offline purchase matching does not establish that recommendations cause purchases or increase sales.

- [Project  final report](reports/PROJECT_1_FINAL.md): project requirements, model comparisons, feature importance, and saved customer examples.
- [Implementation results](reports/IMPLEMENTATION_RESULTS.md): discovery, similarity, clustering, feature experiments, and combined display evaluation.
- [Research and project alignment](reports/MODEL_RESEARCH_AND_PROJECT_ALIGNMENT.md): design rationale and research supporting the experiments.
