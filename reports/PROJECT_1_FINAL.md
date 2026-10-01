# Project 1: Personalized Grocery Reorder and Recommendation System

## Question and delivered system

**Which products is a customer likely to buy again in the next order?** Given only completed orders, the repeat scorer makes one candidate for every product that customer has previously bought. A binary classifier estimates whether each candidate will appear anywhere in the next basket. The `recommend` display first keeps up to five Repeat products scoring at least the default **0.5** cutoff, then adds at most two labeled Discovery items within the five-position limit. The separate `predict-reorders` command uses a validation-selected **0.36** cutoff across repeat candidates and can return a longer variable-size set.

This implements a reorder-probability and Top-K recommendation pipeline. The new-to-customer classifier supplies the exploratory Discovery rows, whose scores are not displayed as reorder probabilities. The [hybrid display evaluation](IMPLEMENTATION_RESULTS.md#repeat-first-hybrid-display) documents this extension separately from the core repeat-model metrics.

| Requested component | Delivered evidence |
| --- | --- |
| Transactional data, EDA, data quality | Six Instacart tables, key/duplicate/order checks, [full-data distributions](figures/eda_distributions.png) and [counts](figures/eda.json) |
| Customer-product features and candidate generation | [History-only feature builder](../grocery_recommender/features.py), one candidate per previously bought product |
| Baseline and Logistic Regression, Random Forest, XGBoost, CatBoost | [Identical-split, identical-feature benchmark](../artifacts/benchmark_10k_standard/metrics.json) below |
| Reorder probability and Top-K ranking | [XGBoost scorer](../grocery_recommender/modeling.py), `recommend`, `predict-reorders`, and [saved customer example](example_recommendations.json) |
| Classification and ranking evaluation | AP, precision, recall, F1, Precision/Recall/NDCG@1/3/5/10, calibration, user-held-out test |
| Business interpretation and reproducibility | Operating tradeoff, failure slices, [explanations](figures/model_explanations.png), this report and runnable commands |

## Data and modeling contract

The supplied data contains **3,214,874 prior orders**, **32,434,489 prior product rows**, **131,209 labeled next orders**, and **75,000 unlabeled next orders**. The target for customer `u` and a previously purchased product `i` is 1 if `i` occurs anywhere in `u`'s next order; otherwise it is 0. Several products may be positive for one customer, so predicted probabilities do not sum to one.

The 22 core features summarize customer order and basket history, customer-product frequency and recency, replenishment intervals, global prior-order product behavior, and aisle/department preferences. The default decision point is **before the next order starts**. Next-order contents, day, hour and elapsed days are unavailable to the model. Global product counts use the available prior-order corpus, including validation/test customers' histories but no hidden next-order labels; this is an offline transductive evaluation rather than a strict calendar-time deployment test.

The data places real limits on the list: **40.1%** of labeled next-basket product rows are new to their customer, and **23.2%** of next baskets contain fewer than five products. The fixed-five ranking benchmark divides by five possible positions, including unfilled positions. Runtime recommendations use the 0.5 cutoff and may show fewer than five products.

## Five-model comparison

The five requested model types were trained or scored on the same 7,000 fit customers, 1,500 validation customers, candidate products and 22 numeric features. These 10,000 sampled customers come from the training portion of the original full-data split. The main selection metric is **validation Precision@5**, since the delivered product is a ranked list. AP is sklearn average precision, used here as the PR-AUC summary. Candidate precision, recall and F1 below use a **fixed score cutoff of 0.5** on the full candidate set; these are descriptive operating-point metrics, not each model's optimally tuned decision policy. Frequency is a historical purchase-rate score, not a calibrated next-order probability, so its cutoff metrics are especially not directly comparable as probabilities.

| Model | AP / PR-AUC | Precision at 0.5 | Recall at 0.5 | F1 at 0.5 | Precision@5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Purchase frequency | 0.3171 | 0.4485 | 0.2604 | 0.3295 | 0.3401 |
| Logistic Regression | 0.4044 | 0.6308 | 0.1730 | 0.2716 | 0.3804 |
| Random Forest | 0.4067 | 0.6333 | 0.1684 | 0.2661 | 0.3805 |
| XGBoost | 0.4121 | 0.6222 | 0.1848 | 0.2849 | **0.3848** |
| CatBoost | **0.4139** | 0.6301 | 0.1804 | 0.2804 | 0.3829 |

CatBoost had the highest AP, while XGBoost had the highest Precision@5. The latter was selected because the project prioritizes correct products in the top five. The small validation gap does not establish broad superiority of either model family. Exact metrics, split IDs, settings and package versions are in [the benchmark artifact](../artifacts/benchmark_10k_standard/metrics.json); the experiment is in [benchmark_models.py](../experiments/benchmark_models.py).

## Selected model and honest evaluation

The established full-data evaluation split has **91,846 model-training customers**, **19,681 validation customers** and **19,682 test customers**. Validation was divided again: 9,840 customers selected the XGBoost round count, while 9,841 different customers selected the reorder cutoff. Only after these choices were fixed was the evaluated model scored on test customers. The deployment artifact was later fit on all 131,209 labeled customers; **its outputs do not have a held-out test score**. The original test has been inspected in subsequent development, so its result is now a historical benchmark rather than an untouched final confirmation for future changes.

| Evaluated full-data result | Value |
| --- | ---: |
| XGBoost Precision@5 | **0.3876**, or 1.94 matching products per customer |
| Purchase-frequency Precision@5 | 0.3422, or 1.71 matching products per customer |
| XGBoost Recall@5 / NDCG@5 | 0.2514 / 0.4543 |
| Candidate-level AP | 0.4205 |
| Repeat-candidate recall of all next-basket items | 0.5961 |
| Answer-seeing oracle restricted to repeat candidates, Precision@5 | 0.7139 |

The default `recommend` cutoff of **0.5** applies only after the top five have been ranked. On decision-validation users, it gave **0.641 precision among displayed products**, with **1.46 products per customer** and empty panels for **41.6%**. On the established test users it gave **0.639 precision**, **1.46 products per customer**, and empty panels for **40.9%**. The [threshold sweep](../artifacts/full_v2/threshold_display_metrics.json) compares 0.5, 0.6, 0.75 and 0.9 on validation and test.

The separate `predict-reorders` cutoff of **0.36** addresses variable-length reorder decisions across all candidate products. On test it returned 3.73 products per customer on average, with **0.523 precision**, **0.310 recall of actual reorders**, and **0.389 F1**. It returned no product for 3,905 of 19,682 customers. The [full result](../artifacts/full_v2/metrics.json) and [probability calibration](figures/score_calibration.png) document this evaluated fit.

### Example recommendation

For unlabeled customer **3**, the all-labeled deployment repeat model uses 12 completed orders and scores 33 previously purchased products. The default 0.5 cutoff accepts these three Repeat recommendations:

| Rank | Product | Estimated next-order reorder probability |
| ---: | --- | ---: |
| 1 | Vanilla Unsweetened Almond Milk | 0.741 |
| 2 | Organic Avocado | 0.682 |
| 3 | Organic Baby Spinach | 0.559 |

The combined `recommend --user-id 3` output adds **Banana** and **Large Lemon** as its two Discovery rows, with no displayed reorder probability. This is a ranked display; scores are estimates and the public dataset does not supply the next basket for this customer. The repeat-only example is saved in [JSON](example_recommendations.json) and recreated by [project1_demo.py](../experiments/project1_demo.py).

## Interpretation and limits

The learned model adds about **0.23 matching products per customer** over purchase frequency in the established full-data test. The 0.5 display cutoff has **0.639 precision among shown products** on test and shows **1.46 per customer**, with **40.9%** getting none. At 0.6, displayed precision rises to **0.709**, while the average falls to **0.86** and empty panels rise to **58.7%**. At 0.75, precision is **0.814** but the average is only **0.24** and **84.9%** get none. At 0.9, precision reaches **0.938**, but fewer than 1% of customers receive any suggestions. Full validation/test tradeoffs are in the [threshold sweep](../artifacts/full_v2/threshold_display_metrics.json); the validation plot is [here](figures/display_tradeoff.png).

The model is weaker for customers with little history: decision-validation Precision@5 was **0.326** after 2–5 prior orders versus **0.449** after 21 or more. The [tree contribution report](figures/model_explanations.json) identifies recency, recent purchase frequency and product reorder behavior as strong influences on fitted scores; contributions explain predictions, not causes of buying.

Offline next-basket matching cannot measure whether a displayed suggestion *caused* a purchase. Prices, promotions, stock, recommendation impressions and exact calendar dates are absent, and inter-order days are capped at 30. The repeat-only candidate set cannot suggest products the customer has never bought; a separately evaluated [new-product preview](IMPLEMENTATION_RESULTS.md#new-product-discovery) currently has only **0.0241 test Precision@5**. The low preview score is a limitation, not a missing step in the core reorder pipeline.

## Reproduce

From the repository root, with the six supplied CSVs present:

```powershell
python -m pip install -e ".[analysis]"
python -m unittest discover -s tests -v
python -m grocery_recommender train-full --batch-size 5000
python -m experiments.benchmark_models --variant standard --max-users 10000 --rounds 350 --output-dir artifacts/benchmark_10k_standard
python -m experiments.portfolio_analysis --model-dir artifacts/benchmark_10k_standard
python -m experiments.project1_demo --user-id 3
python -m grocery_recommender recommend --user-id 3 --max-discovery 0
python -m grocery_recommender recommend --user-id 3 --min-probability 0.75 --max-discovery 0
python -m experiments.threshold_display_sweep
python -m grocery_recommender predict-reorders --user-id 3
```

The [detailed implementation report](IMPLEMENTATION_RESULTS.md) covers the further feature, customer-similarity and discovery experiments. Saved `joblib` files should only be loaded from trusted local artifacts.
