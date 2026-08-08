# Model Evaluation Metrics

A good model is more than just high accuracy. Different metrics tell different stories. You should always look at a few metrics together.

---

## Regression Metrics

Regression predicts continuous values. These metrics measure how close predictions are to the true values.

| Metric | Formula | sklearn | When to Use | Notes |
|---|---|---|---|---|
| MAE (Mean Absolute Error) | $$\frac{1}{m} \sum_{i=1}^{m} \|\hat{y}^{(i)} - y^{(i)}\|$$ | `mean_absolute_error` | You want errors in the same unit as the target. Easy to explain to stakeholders. | Treats all errors equally. Not sensitive to outliers. |
| MSE (Mean Squared Error) | $$\frac{1}{m} \sum_{i=1}^{m} (\hat{y}^{(i)} - y^{(i)})^2$$ | `mean_squared_error` | You want to penalize large errors more than small ones. | Units are squared, harder to interpret directly. Differentiable, so it is used as a loss function. |
| RMSE (Root Mean Squared Error) | $$\sqrt{\frac{1}{m} \sum_{i=1}^{m} (\hat{y}^{(i)} - y^{(i)})^2}$$ | `root_mean_squared_error` (since sklearn 1.4) or `np.sqrt(mean_squared_error(...))` | Most commonly reported metric in industry. Same units as the target, penalizes large errors. | Always report RMSE or MAE, not just R squared. |
| R squared (Coefficient of Determination) | $$1 - \frac{\sum (y^{(i)} - \hat{y}^{(i)})^2}{\sum (y^{(i)} - \bar{y})^2}$$ | `r2_score` | Comparing models on the same dataset. Tells you what fraction of variance the model explains. | Can be negative on test data. Always increases with more features (use adjusted R squared). A very high R squared (0.99+) often means data leakage. |
| MAPE (Mean Absolute Percentage Error) | $$\frac{100\%}{m} \sum \left\|\frac{y^{(i)} - \hat{y}^{(i)}}{y^{(i)}}\right\|$$ | `mean_absolute_percentage_error` | Business reporting, especially forecasting. Stakeholders understand percentages. | Fails when y is zero or near zero. Undefined for actual values of exactly 0. |
| Max Error | $$\max_i \|\hat{y}^{(i)} - y^{(i)}\|$$ | `max_error` | Safety-critical systems. You need to know the worst case. | Very sensitive to a single bad prediction. |
| MedAE (Median Absolute Error) | median of \|\hat{y} - y\| | `median_absolute_error` | Data has many outliers. More robust than MAE. | Ignores the magnitude of half the errors, so not enough on its own. |
| Explained Variance Score | $$1 - \frac{\text{Var}(y - \hat{y})}{\text{Var}(y)}$$ | `explained_variance_score` | Similar to R squared. | Unlike R squared, it does not penalize bias. If the model is systematically off by a constant, R squared drops but EV may not. |

### Regression Metric Selection Guide

| Situation | Recommended Metric | Why |
|---|---|---|
| General purpose, no extreme outliers | RMSE | Most common. Penalizes large errors. Same unit as target. |
| Many outliers in data | MAE | Treats all errors equally. Robust. |
| Business reporting | MAPE or MAE | Percentages are easy to understand. |
| Safety-critical systems | Max Error | You must know the worst-case scenario. |
| Comparing models on same data | R squared | Shows explained variance. Use adjusted R squared for many features. |
| Probabilistic forecasting | Log Likelihood | Proper scoring rule for probabilistic predictions. |

---

## Classification Metrics

Classification predicts categories. These metrics measure how well the model separates classes.

### Confusion Matrix Reference

| | Predicted Positive | Predicted Negative |
|---|---|---|
| Actual Positive | TP (True Positive) | FN (False Negative) |
| Actual Negative | FP (False Positive) | TN (True Negative) |

### Metric Definitions

| Metric | Formula | sklearn | When to Use | Notes |
|---|---|---|---|---|
| Accuracy | $$\frac{TP + TN}{TP + TN + FP + FN}$$ | `accuracy_score` | Balanced classes, quick sanity check. | Misleading for imbalanced data. If 99% of samples are class A, a model that always predicts A gets 99% accuracy but is useless. |
| Precision | $$\frac{TP}{TP + FP}$$ | `precision_score` | False positives are expensive. Example: spam detection (blocking a real email is bad). | Out of all the samples the model called positive, how many were actually positive? |
| Recall (Sensitivity) | $$\frac{TP}{TP + FN}$$ | `recall_score` | False negatives are expensive. Example: disease screening (missing a sick patient is bad). | Out of all the actual positive samples, how many did the model find? |
| Specificity | $$\frac{TN}{TN + FP}$$ | Not directly. Use `recall_score` with `pos_label=0`. | You care about correctly identifying negatives. Often paired with recall in medical testing. | Out of all the actual negative samples, how many did the model find? |
| F1 Score | $$2 \cdot \frac{P \cdot R}{P + R}$$ | `f1_score` | Imbalanced classes, need a single number that balances precision and recall. | The harmonic mean penalizes extreme differences. Use `fbeta_score` with `beta=2` to favor recall or `beta=0.5` to favor precision. |
| ROC AUC | Area under the ROC curve | `roc_auc_score` | Binary classification with probability outputs. Good overall measure. | Plots TPR vs FPR at different thresholds. AUC = 1 means perfect, 0.5 means random. Not reliable for heavily imbalanced datasets. |
| PR AUC (Average Precision) | Area under the precision-recall curve | `average_precision_score` | Imbalanced datasets. More informative than ROC AUC when the positive class is rare. | Plots precision vs recall at different thresholds. Drops when the model fails to find positive samples. |
| Log Loss (Cross-Entropy) | $$-\frac{1}{m} \sum [y \log(\hat{y}) + (1-y) \log(1-\hat{y})]$$ | `log_loss` | Probabilistic models where calibrated confidence matters. | Heavily penalizes confident but wrong predictions. Lower is better. |
| Matthews Correlation Coefficient (MCC) | $$\frac{TP \cdot TN - FP \cdot FN}{\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}}$$ | `matthews_corrcoef` | The best single metric for binary classification, especially with imbalanced data. Ranges from -1 to +1. | Considers all four confusion matrix cells. Only high when the model does well on both classes. |
| Cohen's Kappa | $$\frac{p_o - p_e}{1 - p_e}$$ | `cohen_kappa_score` | Multi-class problems, measuring agreement beyond chance. Ranges from -1 to +1. | 0 means no better than random agreement. |

### Classification Metric Selection Guide

| Situation | Primary Metric | Secondary Metric | Why |
|---|---|---|---|
| Balanced classes, general use | Accuracy | F1 | Simple starting point. |
| Imbalanced, FP is costly | Precision | PR AUC | Minimize false alarms. |
| Imbalanced, FN is costly | Recall | PR AUC | Minimize missed detections. |
| Imbalanced, both matter equally | F1 | MCC or PR AUC | Balanced harmonic mean. |
| Imbalanced, need one best metric | MCC | ROC AUC or PR AUC | Considers all four cells. More honest than F1 for severe imbalance. |
| Ranking quality or threshold tuning | ROC AUC | F1 at chosen threshold | Evaluates across all thresholds. |
| Calibrated probabilities matter | Log Loss | Brier Score | Penalizes overconfidence. |

---

## Clustering Metrics

Clustering is unsupervised. There are no true labels. Metrics measure how well the clusters are separated.

### Metrics Without Ground Truth

These metrics evaluate cluster quality using only the data and the assigned clusters.

| Metric | Formula | sklearn | When to Use | Notes |
|---|---|---|---|---|
| Silhouette Score | $$s = \frac{b - a}{\max(a, b)}$$ where a = mean intra-cluster distance, b = mean nearest-cluster distance | `silhouette_score` | Choosing the number of clusters (k). A higher score means better separation. | Ranges from -1 to +1. Values near 0 mean overlapping clusters. Negative means points might be in the wrong cluster. Works with any distance metric. |
| Davies-Bouldin Index | $$\frac{1}{k} \sum_{i=1}^{k} \max_{j \neq i} \frac{\sigma_i + \sigma_j}{d(c_i, c_j)}$$ | `davies_bouldin_score` | Comparing clustering solutions. Lower is better. | Does not require specifying a distance metric ahead of time. Simpler to compute than silhouette. |
| Calinski-Harabasz Index | $$\frac{\text{between-cluster var}}{\text{within-cluster var}} \cdot \frac{m - k}{k - 1}$$ | `calinski_harabasz_score` | Choosing k. Higher is better. Fast to compute. | Assumes clusters are spherical and Gaussian. Works best with k-means. Does not work well with density-based clusters (like DBSCAN). |

### Metrics With Ground Truth

When you have true labels, use these to compare clustering results to the ground truth.

| Metric | Formula | sklearn | When to Use | Notes |
|---|---|---|---|---|
| Adjusted Rand Index (ARI) | $$\frac{\text{RI} - \text{Expected RI}}{\max(\text{RI}) - \text{Expected RI}}$$ | `adjusted_rand_score` | You have true labels and want to measure agreement, corrected for chance. | Ranges from -1 to +1. 0 means random labeling, 1 means perfect match. Does not assume any mapping between cluster labels and true labels. |
| Normalized Mutual Information (NMI) | Information-theoretic measure | `normalized_mutual_info_score` | You have true labels and the cluster sizes vary a lot. | Ranges from 0 to 1. 1 means perfect agreement. Can be adjusted for chance with `adjusted_mutual_info_score`. |
| Homogeneity Score | Whether each cluster contains only points of a single class | `homogeneity_score` | You care about cluster purity. | A trivial solution (each point its own cluster) gives perfect homogeneity but zero completeness. Always report together with completeness and V-measure. |
| Completeness Score | Whether all points of a class are in the same cluster | `completeness_score` | You care about capturing all members of each class. | Complement to homogeneity. See homogeneity note. |
| V-Measure | $$2 \cdot \frac{H \cdot C}{H + C}$$ | `v_measure_score` | You need a single balanced score combining homogeneity and completeness. | The harmonic mean of homogeneity and completeness. |

---

## Industry Best Practices

| Practice | What It Means | Why It Matters |
|---|---|---|
| Always report 2 to 3 metrics | Never rely on a single metric. For regression: RMSE or MAE plus R squared. For classification: F1 plus ROC AUC or PR AUC. | Each metric has blind spots. Using multiple metrics gives a more complete picture. |
| Always use a holdout test set | Train on one split, tune hyperparameters on a validation split, report final metrics on a test split. | Never report metrics on training data as if they represent real performance. That measures memorization, not generalization. |
| Use cross-validation for small data | For datasets under 10,000 samples, use k-fold cross-validation (5 or 10 folds). Report the mean and standard deviation. | Shows how stable the model is across different data splits. Reduces variance from a single lucky or unlucky split. |
| Compare against a baseline | A model with 90% accuracy sounds good. But if the baseline (always predict the most common class) gets 89%, the model adds little value. | Context matters. Absolute numbers mean nothing without a reference point. |
| Plot the residuals (regression) | Plot residuals (y_true minus y_pred) against predicted values. | Patterns in the plot indicate the model is missing something. Random scatter is what you want. |
| Plot the learning curve | Plot metric over time or over dataset size. | Tells you if more data would help or if the model has hit its ceiling. Guides decisions about collecting more data vs. changing the model. |
| Check the confusion matrix | Always look at the raw TP, TN, FP, FN counts. | Accuracy can hide serious problems. A confusion matrix shows exactly where the model fails. |
| Report per-class metrics (multiclass) | For 3+ classes, report precision, recall, and F1 per class. Use macro, micro, and weighted averages. | Overall accuracy hides poor performance on minority classes. |

### Metric Averages for Multiclass

| Average Type | How It Works | When to Use |
|---|---|---|
| Macro | Computes metric per class, then takes unweighted mean. | All classes are equally important, regardless of size. |
| Micro | Aggregates all TP, FP, FN across classes, then computes metric. | Overall correctness, biased toward large classes. |
| Weighted | Computes metric per class, then takes mean weighted by class size. | You want a single number that accounts for class imbalance. |

---

## Domain-Specific Guidance

| Domain | Primary Metric | Why | Secondary Metrics | Key Warning |
|---|---|---|---|---|
| Healthcare / Medical | Recall (Sensitivity) | Missing a diagnosis (false negative) can cost a life. A false positive means an extra test, which is less harmful. | Specificity, PR AUC | PR AUC is better than ROC AUC here because disease data is often heavily imbalanced (most people are healthy). Calibration matters: when the model says 80% chance of disease, it should truly be 80%, not 60%. |
| Finance / Fraud Detection | Precision | A false positive means blocking a real customer's transaction. Too many false positives and customers leave. | PR AUC, F1, Confusion Matrix | Fraud typically affects 0.1% to 1% of transactions. Accuracy is useless here. Report the value recovered vs false positive cost in dollar terms. |
| Forecasting / Time Series | MAPE (business), RMSE (engineering) | MAPE is the standard in business forecasting. RMSE is for the engineering team. | sMAPE (if values near zero) | Report both MAPE and RMSE alongside each other. MAPE is for the business team. RMSE is for the engineering team. If your product has seasonal patterns, report metrics per season, not just overall. |
| NLP / LLMs | Task-dependent (see below) | Different NLP tasks have different standard metrics. | Human evaluation | Human evaluation is still the gold standard for most NLP tasks. Automated metrics correlate only partially with human judgment. |
| Computer Vision | Task-dependent (see below) | Varies by task: classification, detection, segmentation, generation. | Qualitative inspection (visual examples) | Always look at actual predictions. Metrics can be gamed. |
| Search / Ranking | NDCG, MRR | NDCG accounts for position and relevance level. MRR measures where the first relevant result appears. | Precision@k | Users rarely click past the first 3 to 5 results. Top-k metrics matter far more than metrics over the full result list. Online A/B tests often disagree with offline metrics. |
| Recommendation Systems | Recall@k, Precision@k | Offline metrics are a starting point. Online A/B tests (CTR, watch time, purchase rate) always have the final say. | Hit Rate@k, Catalog Coverage | Diversity and novelty matter too. A system that only recommends the most popular items gets good offline metrics but fails in production. |
| Autonomous / Safety-Critical | Max Error, Worst-Case Bounds | Averages hide catastrophic failures. A model that works 99.9% of the time but fails catastrophically the other 0.1% is unacceptable. | Calibration, Failure Mode Analysis | Report failure modes, not just accuracy. Define explicit safety constraints. Redundancy (multiple models voting) is common. Calibration is mandatory. |
| Marketing / Churn Prediction | Lift, ROI | Business stakeholders do not care about F1 scores. Translate metrics into business outcomes. | Response Rate | Uplift modeling (predicting the causal effect of an intervention) is more useful than pure prediction here. Translate metrics into dollars saved, customers retained, conversion rate increase. |

### NLP Metrics by Task

| Task | Metric | Description | Direction |
|---|---|---|---|
| Machine Translation | BLEU (1-gram to 4-gram) | Measures n-gram overlap with reference translations. | Higher is better |
| Text Summarization | ROUGE (ROUGE-1, ROUGE-2, ROUGE-L) | Measures n-gram and longest common subsequence overlap with reference summaries. | Higher is better |
| Semantic Similarity | BERTScore | Uses contextual embeddings to measure similarity when exact word overlap is not enough. | Higher is better |
| Language Modeling | Perplexity | Measures how well a probability distribution predicts a sample. Exponentiated cross-entropy. | Lower is better |
| Question Answering | Exact Match (EM), F1 | EM checks if prediction exactly matches answer. F1 measures token overlap. | Higher is better |

### Computer Vision Metrics by Task

| Task | Metric | Description | Direction |
|---|---|---|---|
| Image Classification | Top-1 Accuracy, Top-5 Accuracy | Top-1: single best prediction must match label. Top-5: any of the top 5 predictions matches the label. | Higher is better |
| Object Detection | mAP (Mean Average Precision) | Standard benchmark: mAP@0.5, mAP@0.5:0.95. Measures precision at different IoU thresholds. | Higher is better |
| Object Detection | IoU (Intersection over Union) | Measures overlap between predicted and ground truth bounding boxes. | Higher is better |
| Image Segmentation | Dice Coefficient | Like F1 score but computed over pixels. Measures overlap between predicted and ground truth masks. | Higher is better |
| Image Generation | PSNR, SSIM | PSNR measures pixel-level fidelity. SSIM measures perceived structural similarity. | Higher is better |

---

## Data Splitting

| Concept | Description | Best Practice |
|---|---|---|
| Train / Validation / Test Split | Data is divided into three parts: training (model fitting), validation (hyperparameter tuning), and test (final evaluation). | A standard split is 70% train, 15% validation, 15% test. The test set is locked away until the very end. Hyperparameters and model selection happen on the validation set. |
| K-Fold Cross-Validation | Split data into k parts. Train on k-1 parts, validate on the remaining part. Repeat k times. Report mean and standard deviation of the metric. | Use when data is limited (under 10,000 samples). k=5 or k=10 are common choices. Never use for time series without special handling. |
| Stratified K-Fold | Like K-Fold but preserves class proportions in each fold. | Always use for classification with imbalanced data. Prevents folds with zero samples of a minority class. |
| Time Series Split | Respects chronological order. Training data always comes before validation data. | Mandatory for forecasting. Never shuffle time series data. Use expanding or sliding window approaches. |
| Holdout Method | Single train/test split. No cross-validation. | Only use when data is very large (over 100,000 samples). Otherwise you risk a lucky or unlucky split. |

## Common Mistakes and Anti-Patterns

| Mistake | Why It Happens | How to Avoid It |
|---|---|---|
| Evaluating on training data | Tempting to check how well the model learned the data it saw. | Always use a separate test set. Training metrics tell you about memorization, not generalization. |
| Scaling before splitting | You want to normalize the whole dataset, but scaling fit on all data leaks test information into training. | Always split first, then fit the scaler on training data only, and apply the same transformation to validation and test. |
| Including future data when training on historical data | Time-based features are not handled correctly. Example: using next month's data to predict this month. | Always respect chronological order. Never shuffle time series data. Verify that all training features are available at prediction time. |
| Using the same samples in both train and test (duplicates) | Data cleaning creates near-duplicate rows. | Deduplicate your dataset before splitting. Check for duplicate rows and near-duplicate rows. |
| Features that contain the target or a proxy | A feature like "total amount" when the target is "amount per item" multiplied by "quantity" (which is also a feature). | Audit every feature. Ask: does this feature depend on information that would not be available at prediction time? |
| Reporting only accuracy on imbalanced data | Accuracy is the default metric people reach for. | Always check class distribution first. If classes are imbalanced, use F1, MCC, or PR AUC instead. |
| Over-interpreting small differences | Two models with 0.94 and 0.93 AUC may not be meaningfully different. | Use statistical tests (McNemar, paired bootstrap) to check if differences are significant. Report confidence intervals, not just point estimates. |
| Tuning hyperparameters on the test set | Repeatedly checking test performance and adjusting the model until the test score looks good. | The test set is for final evaluation only. Use a separate validation set or cross-validation for tuning. If you peek at the test set, it becomes part of the training process. |
