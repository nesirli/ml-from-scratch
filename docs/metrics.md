# Model Evaluation Metrics

A good model is more than just high accuracy. Different metrics tell different stories. You should always look at a few metrics together.

## Regression Metrics

Regression predicts continuous values. These metrics measure how close predictions are to the true values.

### MAE (Mean Absolute Error)

$$\text{MAE} = \frac{1}{m} \sum_{i=1}^{m} |\hat{y}^{(i)} - y^{(i)}|$$

- **sklearn:** `mean_absolute_error`
- **When to use:** You want errors in the same unit as the target. Easy to explain to stakeholders
- **Note:** Treats all errors equally. Not sensitive to outliers

### MSE (Mean Squared Error)

$$\text{MSE} = \frac{1}{m} \sum_{i=1}^{m} (\hat{y}^{(i)} - y^{(i)})^2$$

- **sklearn:** `mean_squared_error`
- **When to use:** You want to penalize large errors more than small ones
- **Note:** Units are squared, harder to interpret directly. Differentiable, so it is used as a loss function

### RMSE (Root Mean Squared Error)

$$\text{RMSE} = \sqrt{\frac{1}{m} \sum_{i=1}^{m} (\hat{y}^{(i)} - y^{(i)})^2}$$

- **sklearn:** `root_mean_squared_error` (since scikit-learn 1.4) or `np.sqrt(mean_squared_error(...))`
- **When to use:** Most commonly reported metric in industry. Same units as the target, penalizes large errors
- **Note:** Always report RMSE or MAE, not just R²

### R² (Coefficient of Determination)

$$R^2 = 1 - \frac{\sum_{i=1}^{m} (y^{(i)} - \hat{y}^{(i)})^2}{\sum_{i=1}^{m} (y^{(i)} - \bar{y})^2}$$

- **sklearn:** `r2_score`
- **When to use:** Comparing models on the same dataset. Tells you what fraction of variance the model explains
- **Note:** Can be negative on test data if the model is worse than predicting the mean. Always increases with more features (use adjusted R²). A very high R² (0.99+) often means data leakage

### MAPE (Mean Absolute Percentage Error)

$$\text{MAPE} = \frac{100\%}{m} \sum_{i=1}^{m} \left|\frac{y^{(i)} - \hat{y}^{(i)}}{y^{(i)}}\right|$$

- **sklearn:** `mean_absolute_percentage_error`
- **When to use:** Business reporting, especially forecasting. Stakeholders understand percentages
- **Note:** Fails when y is zero or near zero. Undefined for actual values of exactly 0

### Max Error

$$\text{Max Error} = \max_i |\hat{y}^{(i)} - y^{(i)}|$$

- **sklearn:** `max_error`
- **When to use:** Safety-critical systems. You need to know the worst case
- **Note:** Very sensitive to a single bad prediction

### MedAE (Median Absolute Error)

- **sklearn:** `median_absolute_error`
- **When to use:** Data has many outliers. More robust than MAE
- **Note:** Ignores the magnitude of half the errors, so not enough on its own

### Explained Variance Score

$$\text{EV} = 1 - \frac{\text{Var}(y - \hat{y})}{\text{Var}(y)}$$

- **sklearn:** `explained_variance_score`
- **When to use:** Similar to R²
- **Note:** Unlike R², it does not penalize bias. If the model is systematically off by a constant, R² drops but EV may not

## Classification Metrics

Classification predicts categories. These metrics measure how well the model separates classes.

### Accuracy

$$\text{Accuracy} = \frac{\text{TP} + \text{TN}}{\text{TP} + \text{TN} + \text{FP} + \text{FN}}$$

- **sklearn:** `accuracy_score`
- **When to use:** Balanced classes, quick sanity check
- **Note:** Misleading for imbalanced data. If 99% of samples are class A, a model that always predicts A gets 99% accuracy but is useless

### Precision

$$\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}$$

- **sklearn:** `precision_score`
- **When to use:** False positives are expensive. For example, spam detection (blocking a real email is bad)
- **Note:** Out of all the samples the model called positive, how many were actually positive

### Recall (Sensitivity)

$$\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}$$

- **sklearn:** `recall_score`
- **When to use:** False negatives are expensive. For example, disease screening (missing a sick patient is bad)
- **Note:** Out of all the actual positive samples, how many did the model find

### F1 Score

$$F_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$

- **sklearn:** `f1_score`
- **When to use:** Imbalanced classes, need a single number that balances precision and recall
- **Note:** The harmonic mean is used instead of the arithmetic mean. It penalizes extreme differences. Use `fbeta_score` with `beta=2` to favor recall or `beta=0.5` to favor precision

### ROC AUC

- **sklearn:** `roc_auc_score`
- **When to use:** Binary classification with probability outputs. Good overall measure
- **Note:** Plots the true positive rate vs false positive rate at different thresholds. AUC = 1 means perfect separation, AUC = 0.5 means random. Not reliable for heavily imbalanced datasets

### PR AUC (Average Precision)

- **sklearn:** `average_precision_score`
- **When to use:** Imbalanced datasets. More informative than ROC AUC when the positive class is rare
- **Note:** Plots precision vs recall at different thresholds. Drops when the model fails to find positive samples

### Log Loss (Cross-Entropy)

$$\text{Log Loss} = -\frac{1}{m} \sum_{i=1}^{m} \left[ y^{(i)} \log(\hat{y}^{(i)}) + (1 - y^{(i)}) \log(1 - \hat{y}^{(i)}) \right]$$

- **sklearn:** `log_loss`
- **When to use:** Probabilistic models where calibrated confidence matters
- **Note:** Heavily penalizes confident but wrong predictions. Lower is better

### Confusion Matrix

- **sklearn:** `confusion_matrix`
- **When to use:** Understanding which classes get confused. The starting point for any classification task
- **Note:** Shows TP, TN, FP, FN in a table. From this you can derive most other metrics

### Matthews Correlation Coefficient (MCC)

$$\text{MCC} = \frac{\text{TP} \cdot \text{TN} - \text{FP} \cdot \text{FN}}{\sqrt{(\text{TP} + \text{FP})(\text{TP} + \text{FN})(\text{TN} + \text{FP})(\text{TN} + \text{FN})}}$$

- **sklearn:** `matthews_corrcoef`
- **When to use:** The best single metric for binary classification, especially with imbalanced data. Ranges from -1 to +1
- **Note:** Considers all four confusion matrix cells. Only high when the model does well on both classes

### Cohen's Kappa

- **sklearn:** `cohen_kappa_score`
- **When to use:** Multi-class problems, measuring agreement beyond chance
- **Note:** Ranges from -1 to +1. 0 means no better than random agreement

## Clustering Metrics

Clustering is unsupervised. There are no true labels. Metrics measure how well the clusters are separated.

### Metrics Without Ground Truth

These metrics evaluate cluster quality using only the data and the assigned clusters.

#### Silhouette Score

$$s = \frac{b - a}{\max(a, b)}$$

Where a is the mean distance to points in the same cluster and b is the mean distance to points in the nearest other cluster.

- **sklearn:** `silhouette_score`
- **When to use:** Choosing the number of clusters (k). A higher score means better separation
- **Note:** Ranges from -1 to +1. Values near 0 mean overlapping clusters. Negative values mean points might be in the wrong cluster. Works with any distance metric

#### Davies-Bouldin Index

$$\text{DB} = \frac{1}{k} \sum_{i=1}^{k} \max_{j \neq i} \frac{(\sigma_i + \sigma_j)}{d(c_i, c_j)}$$

Where σᵢ is the average distance of points in cluster i to its centroid, and d(cᵢ, cⱼ) is the distance between centroids.

- **sklearn:** `davies_bouldin_score`
- **When to use:** Comparing clustering solutions. Lower is better
- **Note:** Does not require specifying a distance metric ahead of time. Simpler to compute than the silhouette score

#### Calinski-Harabasz Index (Variance Ratio)

$$\text{CH} = \frac{\text{between-cluster variance}}{\text{within-cluster variance}} \cdot \frac{m - k}{k - 1}$$

- **sklearn:** `calinski_harabasz_score`
- **When to use:** Choosing k. Higher is better. Fast to compute
- **Note:** Assumes clusters are spherical and Gaussian. Works best with k-means. Does not work well with density-based clusters (like DBSCAN)

### Metrics With Ground Truth

When you have true labels, use these to compare clustering results to the ground truth.

#### Adjusted Rand Index (ARI)

$$\text{ARI} = \frac{\text{RI} - \text{Expected RI}}{\max(\text{RI}) - \text{Expected RI}}$$

- **sklearn:** `adjusted_rand_score`
- **When to use:** You have true labels and want to measure agreement, corrected for chance
- **Note:** Ranges from -1 to +1. 0 means random labeling, 1 means perfect match. Does not assume any mapping between cluster labels and true labels

#### Normalized Mutual Information (NMI)

- **sklearn:** `normalized_mutual_info_score`
- **When to use:** You have true labels and the cluster sizes vary a lot
- **Note:** Ranges from 0 to 1. 1 means perfect agreement. Can be adjusted for chance with `adjusted_mutual_info_score`

#### Homogeneity, Completeness, V-Measure

- **Homogeneity:** Each cluster contains only points of a single class
- **Completeness:** All points of a class are in the same cluster
- **V-Measure:** The harmonic mean of homogeneity and completeness

- **sklearn:** `homogeneity_score`, `completeness_score`, `v_measure_score`
- **When to use:** You care about both purity and coverage of clusters
- **Note:** A trivial solution (each point its own cluster) gives perfect homogeneity but zero completeness. These metrics are always reported together

## Industry Best Practices

### Always Report 2-3 Metrics

Never rely on a single metric. For regression, use RMSE or MAE plus R². For classification, use F1 plus ROC AUC or PR AUC. The confusion matrix is always worth looking at.

### Always Use a Holdout Test Set

Train on one split, tune hyperparameters on a validation split, and report final metrics on a test split. Never report metrics on training data as if they represent real performance.

### Cross-Validation

For small datasets, use k-fold cross-validation (5 or 10 folds). Report the mean and standard deviation of the metric across folds. This tells you how stable the model is.

### Compare Against a Baseline

A model with 90% accuracy sounds good. But if the baseline (always predict the most common class) gets 89%, the model adds little value. Always report baseline performance.

### Plot the Residuals

For regression, plot residuals (y_true minus y_pred) against predicted values. Patterns in the plot indicate the model is missing something. Random scatter is what you want.

### Plot the Learning Curve

Metric over time or over dataset size tells you if more data would help or if the model has hit its ceiling.

## Domain-Specific Guidance

### Healthcare / Medical

Recall is the most important metric. Missing a diagnosis (false negative) can cost a life. A false positive means an extra test, which is less harmful. Report sensitivity (recall) and specificity together.

PR AUC is better than ROC AUC here because disease data is often heavily imbalanced (most people are healthy).

Calibration matters: when the model says 80% chance of disease, it should truly be 80%, not 60%.

### Finance / Fraud Detection

Precision is critical. A false positive means blocking a real customer's transaction. Too many false positives and customers leave.

Fraud typically affects 0.1% to 1% of transactions. Accuracy is useless here. Use PR AUC, F1, and always check the confusion matrix. Report the value recovered vs false positive cost in dollar terms.

### Forecasting / Time Series

MAPE is the standard in business forecasting. sMAPE (symmetric MAPE) handles cases where actual values are near zero better.

Report RMSE alongside MAPE. MAPE is for the business team. RMSE is for the engineering team. If your product has seasonal patterns, report metrics per season, not just overall.

### NLP / LLMs

- **BLEU** for machine translation (1-gram to 4-gram). Higher is better
- **ROUGE** for text summarization (ROUGE-1, ROUGE-2, ROUGE-L)
- **BERTScore** for semantic similarity when exact word overlap is not enough
- **Perplexity** for language model quality. Lower is better
- **Exact Match (EM)** for question answering tasks

Human evaluation is still the gold standard for most NLP tasks. Automated metrics correlate only partially with human judgment.

### Computer Vision

- **IoU (Intersection over Union)** for object detection. Measures overlap between predicted and ground truth bounding boxes
- **mAP (Mean Average Precision)** for object detection. The standard benchmark metric (mAP@0.5, mAP@0.5:0.95)
- **Dice Coefficient** for image segmentation. Like F1 score but for pixels
- **Top-1 / Top-5 Accuracy** for image classification. Top-5 means any of the top 5 predictions matches the label
- **PSNR / SSIM** for image generation quality

### Search / Ranking

- **MRR (Mean Reciprocal Rank)** measures where the first relevant result appears
- **NDCG (Normalized Discounted Cumulative Gain)** accounts for position and relevance level
- **Precision@k** is the fraction of top-k results that are relevant

Users rarely click past the first 3 to 5 results. Top-k metrics matter far more than metrics over the full result list. Online A/B tests often disagree with offline metrics. Measure clicks and conversions in production.

### Recommendation Systems

- **Hit Rate@k** measures whether a relevant item appears in the top k
- **Recall@k** measures how many relevant items are in the top k
- **Precision@k** measures what fraction of the top k is relevant
- **Catalog Coverage** measures how many unique items the system recommends

Offline metrics are a starting point. Online A/B tests (click-through rate, watch time, purchase rate) always have the final say. Diversity and novelty matter too. A system that only recommends the most popular items gets good offline metrics but fails in production.

### Autonomous / Safety-Critical

Max error and worst-case bounds matter more than averages. A model that works 99.9% of the time but fails catastrophically the other 0.1% is unacceptable.

Calibration is mandatory. If the model outputs a confidence score, it must be accurate. Redundancy (multiple models voting) is common.

Report failure modes, not just accuracy. Define explicit safety constraints that the model must never violate.

### Marketing / Churn Prediction

- **Lift** measures how much better the model is than random targeting
- **ROI** is what the business cares about: revenue gained vs cost of the model
- **Response rate** is the fraction of targeted users who take action

Business stakeholders do not care about F1 scores. Translate metrics into business outcomes: dollars saved, customers retained, conversion rate increase. Uplift modeling (predicting the causal effect of an intervention) is more useful than pure prediction here.

## Data Splitting

### Train / Validation / Test Split

A standard split is 70% train, 15% validation, 15% test. The test set is locked away until the very end. Hyperparameters and model selection happen on the validation set.

### Cross-Validation

When data is limited (under 10,000 samples), use k-fold cross-validation. Split the data into k parts. Train on k-1 parts, validate on the remaining part. Repeat k times. Report the mean and standard deviation.

- **k=5** or **k=10** are common choices
- **Stratified K-Fold** keeps class proportions the same across folds (for classification)
- **Time Series Split** respects chronological order (for forecasting)

### Never Evaluate on Training Data

A model can memorize the training data and get 100% accuracy while being useless on new data. This is overfitting. Always evaluate on data the model has not seen during training.

### Data Leakage

The most common cause of unrealistically good metrics. Examples:

- Scaling before splitting (information from test set leaks into training)
- Including future data when training on historical data
- Using the same samples in both train and test (duplicates)
- Features that contain the target variable or a proxy for it

If your test metrics are suspiciously high, look for leakage first.
