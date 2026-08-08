# Machine Learning Model Classification

Every machine learning model can be described by three structural properties: its **parametric nature**, its **memory type**, and its **learning type**. This document classifies all models in this repository (and a few common additions) along these dimensions.

---

## 1. Parametric vs. Non-Parametric

This distinction tells you whether the model size is fixed or grows with data.

| Property | Parametric | Non-Parametric |
|---|---|---|
| Parameter count | Fixed, decided before training | Flexible, grows with the training data |
| Model size | Independent of dataset size | Depends on dataset size |
| Training speed | Usually faster | Usually slower |
| Memory usage | Constant after training | Can grow with more data |
| Flexibility | Assumes a specific functional form | Makes fewer assumptions about data shape |
| Risk | Underfitting (too simple) | Overfitting (too complex) |
| Examples | Linear Regression, Logistic Regression, Neural Networks, K-Means | KNN, Decision Trees, Gradient Boosting, SVM with RBF kernel |

### Parametric Models

Parametric models summarize the training data into a fixed set of numbers (called weights, coefficients, or parameters). Once trained, the model size stays the same whether you trained on 1,000 or 1,000,000 rows. You can discard the training data after fitting.

**Strengths:**
- Fast predictions
- Low memory after training
- Easier to interpret (fewer parameters means more explainable)
- Work well when the data matches the model's assumptions

**Weaknesses:**
- If the real data pattern does not match the model's functional form, the model will underfit
- Cannot adapt to complex, irregular patterns without adding more parameters by hand

### Non-Parametric Models

Non-parametric models do not assume a fixed functional form. Their internal structure (number of splits, number of stored points, number of kernels) grows as the training dataset grows. The name is misleading: non-parametric does not mean zero parameters. It means the parameter space is not fixed ahead of time.

**Strengths:**
- Flexible: can fit complex, irregular decision boundaries
- Fewer assumptions about the data distribution
- Often achieve higher accuracy on large, messy datasets

**Weaknesses:**
- Slower to train and predict on large datasets
- Higher memory usage (may need to keep training data or a complex structure)
- More prone to overfitting if not regularized properly

---

## 2. Instance-Based vs. Model-Based

This distinction tells you how the model makes predictions: by memorizing data or by building an abstraction.

| Property | Instance-Based | Model-Based |
|---|---|---|
| What it learns | Stores training examples | Builds a mathematical summary (formula, tree, or graph) |
| Prediction method | Compare new point to stored examples | Apply the learned formula |
| Training time | Very fast (just store the data) | Slower (iterative optimization) |
| Prediction time | Slow (computes distances) | Fast (evaluates a formula) |
| Data after training | Must keep all or most training data | Can delete training data |
| Examples | KNN, SVM with RBF kernel | Linear Regression, Decision Trees, Neural Networks |

### Instance-Based (Memory-Based) Models

Instance-based models do not learn explicit equations. They store training data points and make predictions by comparing new inputs to stored examples using a similarity or distance measure.

**Strengths:**
- Training is instant (just remember the data)
- Naturally handle new training data: just add it
- No assumptions about the global structure of the data

**Weaknesses:**
- Prediction is slow for large datasets (every new point must be compared against every stored point)
- High memory cost (must keep the training set)
- Sensitive to noise and irrelevant features (distance metrics treat all features equally)
- Curse of dimensionality: distances become less meaningful in high-dimensional spaces

### Model-Based Models

Model-based models create a generalized blueprint (a formula, a tree, or a graph) during training. Once that blueprint is learned, you can throw away the raw training data. Predictions are made by applying the blueprint to new inputs.

**Strengths:**
- Fast predictions after training
- Low memory footprint after training
- Often more interpretable (you can inspect the learned parameters)
- Better at ignoring noise (generalization)

**Weaknesses:**
- Training takes time and computation
- Assumes a specific functional form that might not match the real data
- Harder to update incrementally when new data arrives

---

## 3. Learning Types

Models are also classified by what kind of signal they learn from.

| Type | Description | Examples |
|---|---|---|
| Supervised | Learns from labeled data (input, output pairs) | Linear Regression, Logistic Regression, Decision Trees, SVM, Neural Networks |
| Unsupervised | Learns patterns from unlabeled data | K-Means, GMM, PCA, Hierarchical Clustering |
| Self-Supervised | Creates labels from the data itself (predict part of input from other parts) | Word2Vec, Transformer Language Model, BERT |
| Reinforcement | Learns by interacting with an environment and receiving rewards | Q-Learning, Policy Gradients (not in this repo) |

---

## Complete Model Classification Table

Each model below is classified by structural type, memory type, and learning type. The table also includes core assumptions, typical use cases, and key limitations.

### Supervised Models

| Model | Parametric / Non-Parametric | Memory Type | Learning Type | Core Assumptions | Typical Use | Key Limitation |
|---|---|---|---|---|---|---|
| Linear Regression | Parametric | Model-Based | Supervised (Regression) | Linear relationship between features and target. Errors are normally distributed, independent, and have constant variance (homoscedasticity). | Predicting house prices, sales forecasts, any continuous target with roughly linear feature relationships. | Cannot capture nonlinear patterns. Sensitive to outliers and multicollinearity. |
| Ridge Regression | Parametric | Model-Based | Supervised (Regression) | Same as linear regression. Adds L2 penalty to shrink coefficients toward zero. | Same as linear regression but when features are highly correlated (multicollinearity). | Same as linear regression. Does not perform feature selection (coefficients shrink but never become zero). |
| Lasso Regression | Parametric | Model-Based | Supervised (Regression) | Same as linear regression. Adds L1 penalty that can shrink coefficients to exactly zero. | Same as linear regression plus automatic feature selection. | May select one feature from a correlated group arbitrarily. Unstable when features are highly correlated. |
| Logistic Regression | Parametric | Model-Based | Supervised (Classification) | Linear relationship between features and the log-odds of the outcome. Observations are independent. | Binary classification: spam detection, disease diagnosis, credit default prediction. | Assumes linear decision boundary. Struggles with complex nonlinear class separation without feature engineering. |
| Linear Discriminant Analysis (LDA) | Parametric | Model-Based | Supervised (Classification) | Features are normally distributed within each class. All classes share the same covariance matrix. | Binary and multiclass classification when normality assumption roughly holds. Often used as a dimensionality reduction step. | Fails when class covariances differ significantly. Sensitive to outliers. |
| Naive Bayes | Parametric | Model-Based | Supervised (Classification) | All features are conditionally independent given the class label. This is almost never true in practice, but the model often works anyway. | Text classification (spam filtering, sentiment analysis), document categorization. Fast baseline model. | The independence assumption is unrealistic for many problems. Struggles when features have strong interactions. |
| K-Nearest Neighbors (KNN) | Non-Parametric | Instance-Based | Supervised (Classification / Regression) | Similar data points exist close to each other in the feature space (measured by Euclidean or Manhattan distance). | Small to medium datasets where decision boundaries are irregular. Good baseline model. | Very slow at prediction time for large datasets. Sensitive to irrelevant features and feature scale. Suffers from the curse of dimensionality. |
| Decision Trees (CART) | Non-Parametric | Model-Based | Supervised (Classification / Regression) | The feature space can be split into orthogonal (axis-aligned) regions. No distributional assumptions. | When model interpretability matters (medical decisions, credit scoring). When you need to explain every prediction. | Highly prone to overfitting if not pruned or constrained. Small changes in data can produce very different trees (high variance). |
| Random Forest | Non-Parametric | Model-Based | Supervised (Classification / Regression) | Aggregating many high-variance, low-bias trees reduces overall error (the wisdom of crowds). Each tree sees a random subset of features and samples. | General purpose: works well on structured/tabular data with little tuning. Feature importance is a built-in byproduct. | Slower and less interpretable than a single decision tree. Less effective on very high-dimensional sparse data (text, images). |
| Gradient Boosting (GBM, XGBoost, LightGBM, CatBoost) | Non-Parametric | Model-Based | Supervised (Classification / Regression) | A sequence of weak learners (shallow trees) can correct each other's errors iteratively. Each new tree fits the residual errors of the ensemble so far. | Tabular data competitions and production systems. Often the go-to model for structured data. Handles missing values and mixed feature types well. | More hyperparameters to tune than Random Forest. Can overfit if too many trees or too high learning rate. Sequential training is harder to parallelize. |
| Support Vector Machine (Linear) | Parametric | Model-Based | Supervised (Classification / Regression) | Classes can be separated by a linear hyperplane with a maximum margin. | Text classification with TF-IDF features (high-dimensional, sparse data). When the number of features exceeds the number of samples. | Cannot separate classes that are not linearly separable. Sensitive to feature scaling. |
| Support Vector Machine (RBF/Kernel) | Non-Parametric | Instance-Based | Supervised (Classification / Regression) | Data that is not linearly separable in the original space becomes separable when projected into a higher-dimensional space via a kernel function. | Moderate-sized datasets with complex, nonlinear decision boundaries. Image classification and bioinformatics. | Slow on large datasets (training time scales quadratically). Choice of kernel and hyperparameters (C, gamma) heavily impacts performance. Less interpretable than linear models. |

### Unsupervised Models

| Model | Parametric / Non-Parametric | Memory Type | Learning Type | Core Assumptions | Typical Use | Key Limitation |
|---|---|---|---|---|---|---|
| K-Means Clustering | Parametric | Model-Based | Unsupervised (Clustering) | Clusters are spherical, have similar sizes, and have similar variances in all directions (isotropic). The number of clusters k is known ahead of time. | Customer segmentation, image compression (color quantization), initial data exploration. | Must specify k manually. Assumes spherical, equally sized clusters. Sensitive to initial centroid placement. Struggles with elongated or irregularly shaped clusters. |
| K-Means++ | Parametric | Model-Based | Unsupervised (Clustering) | Same as K-Means. Uses a smarter initialization strategy: centroids are spread out initially. | Same as K-Means. Almost always preferred over standard K-Means because of better and more consistent results. | Same structural limitations as K-Means. Only the initialization is improved. |
| Gaussian Mixture Models (GMM) | Parametric | Model-Based | Unsupervised (Clustering / Density Estimation) | Data is generated from a mixture of a finite number of Gaussian distributions. Each point has a probability of belonging to each cluster (soft assignment). | When clusters may overlap or have different shapes and sizes. Density estimation and anomaly detection. | Assumes clusters are Gaussian shaped. Must specify the number of components. EM algorithm can converge to local optima. More parameters to estimate than K-Means (covariance matrices). |
| DBSCAN | Non-Parametric | Instance-Based | Unsupervised (Clustering) | Clusters are dense regions of points separated by sparse regions. There is no assumption about cluster shape or count. | When clusters have arbitrary shapes. When the dataset contains noise and outliers. Geospatial data. | Struggles with clusters of varying density. Sensitive to the epsilon and min_samples parameters. Ineffective in high-dimensional spaces (everything becomes sparse). |
| Hierarchical Clustering | Non-Parametric | Model-Based | Unsupervised (Clustering) | Data can be organized into a tree structure (dendrogram) by successively merging or splitting clusters based on a distance metric. | When you want to see the full cluster hierarchy. When the number of clusters is unknown. Biology (phylogenetic trees), document clustering. | Computationally expensive (O(n^3) for some linkage methods). Once a merge or split is made, it cannot be undone. Choice of linkage method strongly affects results. |
| Principal Component Analysis (PCA) | Parametric | Model-Based | Unsupervised (Dimensionality Reduction) | Variance is a measure of information. The directions of maximum variance capture the most important structure in the data. | Dimensionality reduction before applying other models. Data visualization (2D or 3D plots). Noise filtering and feature decorrelation. | Only captures linear relationships. Directions of maximum variance may not correspond to directions useful for a downstream task (unsupervised, not task-aware). Sensitive to feature scale. |

### Deep Learning Models

| Model | Parametric / Non-Parametric | Memory Type | Learning Type | Core Assumptions | Typical Use | Key Limitation |
|---|---|---|---|---|---|---|
| Multi-Layer Perceptron (MLP) | Parametric | Model-Based | Supervised (Classification / Regression) | Patterns in data can be mapped through a sequence of alternating linear transformations and nonlinear activation functions. No strict geometric assumptions. | Universal function approximator for structured/tabular data. Good for problems where feature interactions are complex but the structure is not spatial or sequential. | Requires large amounts of data. Many hyperparameters (layers, units, activation, learning rate). Less interpretable than linear models or trees. |
| Convolutional Neural Network (CNN) | Parametric | Model-Based | Supervised (Classification / Regression) | Local spatial patterns matter (translation invariance). Nearby pixels or values are more related than distant ones. The same pattern can appear anywhere in the input. | Image classification, object detection, image segmentation. Also used for 1D signals (audio, time series) and 3D volumes (medical imaging). | Requires large labeled datasets. Computationally expensive to train. Many architecture choices. Not effective for non-spatial data. |
| Recurrent Neural Network (RNN) | Parametric | Model-Based | Supervised (Sequence Modeling) | Sequential order matters. Information from earlier time steps influences later ones through a hidden state that is updated at each step. | Time series forecasting, sequence classification, language modeling (basic). Short sequences where long-range dependencies are not critical. | Suffers from vanishing and exploding gradients. Cannot capture long-range dependencies effectively. Training is slow (sequential, hard to parallelize). |
| Long Short-Term Memory (LSTM) | Parametric | Model-Based | Supervised (Sequence Modeling) | Same as RNN but adds gating mechanisms (input, forget, output gates) to control information flow. This allows selective memory over long sequences. | Machine translation, speech recognition, time series with long-term patterns, text generation. Any sequence task where long-range dependencies matter. | Still sequential (hard to parallelize). More parameters than simple RNN (four times as many). Outperformed by Transformers on many tasks with enough data. |

### NLP and Representation Learning Models

| Model | Parametric / Non-Parametric | Memory Type | Learning Type | Core Assumptions | Typical Use | Key Limitation |
|---|---|---|---|---|---|---|
| N-Gram Language Model | Parametric | Model-Based | Self-Supervised (Language Modeling) | The probability of a word depends only on the previous N-1 words (Markov assumption). Word frequencies in the training corpus reflect real-world probabilities. | Text generation, spelling correction, speech recognition (as a prior over word sequences). Simple and fast baseline. | Cannot capture context beyond N words. Suffers from data sparsity (most n-grams never appear in training). Vocabulary is fixed at training time. |
| Word2Vec (Skip-Gram / CBOW) | Parametric | Model-Based | Self-Supervised (Representation Learning) | Words that appear in similar contexts have similar meanings (distributional hypothesis). The relationship between word vectors can capture semantic and syntactic patterns. | Generating word embeddings for downstream NLP tasks. Semantic similarity, analogy tasks (king - man + woman = queen). Historical significance: kickstarted modern NLP. | Static embeddings: each word has one vector regardless of context ("bank" means the same thing in "river bank" and "bank account"). Cannot handle out-of-vocabulary words. |
| Attention Mechanism | Parametric | Model-Based | Supervised / Self-Supervised | Not all parts of the input are equally important for each output step. The model can learn to focus on relevant parts by computing weighted sums over the input. | Sequence-to-sequence tasks (neural machine translation, text summarization). Originally designed as an add-on to RNNs. Now a core building block of Transformers. | Standalone attention on RNNs is still sequential. Quadratic complexity in sequence length for self-attention (addressed in later variants). |
| Transformer | Parametric | Model-Based | Supervised / Self-Supervised | All relationships between positions in a sequence can be modeled using only attention, without recurrence or convolution. Positional encodings are added to preserve order information. | Neural machine translation, language modeling, text classification, and increasingly, computer vision and audio processing. The foundation of modern LLMs (GPT, BERT, Claude). | Quadratic memory and compute in sequence length (self-attention is O(n^2)). Requires large datasets to train from scratch. Many design choices and hyperparameters. |
| Transformer Language Model (GPT-style) | Parametric | Model-Based | Self-Supervised (Language Modeling) | Language can be modeled autoregressively: predict the next token given all previous tokens. A large enough model trained on enough data can learn a broad range of linguistic and world knowledge. | Text generation, few-shot learning, code completion, conversational AI. Foundation models that can be fine-tuned or prompted for many downstream tasks. | Extremely expensive to train (millions of dollars in compute). Can hallucinate incorrect information. O(n^2) attention limits context length without optimizations. Static knowledge cutoff (does not learn from new data unless fine-tuned). |

### Comparison of Similar Models

These mini-tables compare models that are often confused with each other.

#### Linear Regression vs. Ridge vs. Lasso

| Criterion | Linear Regression | Ridge Regression | Lasso Regression |
|---|---|---|---|
| Penalty term | None | L2 (sum of squared weights) | L1 (sum of absolute weights) |
| Coefficient behavior | Unconstrained | Shrunk toward zero | Shrunk to exactly zero (sparse) |
| Feature selection | No | No | Yes |
| Best when | All features are relevant | Many correlated features | Only a few features matter |
| Solution method | Closed form (Normal Equation) or GD | Closed form or GD | Iterative (no closed form) |

#### Logistic Regression vs. LDA vs. Naive Bayes

| Criterion | Logistic Regression | LDA | Naive Bayes |
|---|---|---|---|
| Modeling approach | Discriminative (models P(y|x) directly) | Generative (models P(x|y), then applies Bayes rule) | Generative with independence assumption |
| Feature distribution assumption | None | Gaussian per class, shared covariance | Any distribution, but independent |
| Decision boundary | Linear | Linear | Linear (but estimated differently) |
| Training speed | Moderately fast (iterative) | Fast (closed form) | Very fast (count and divide) |
| Best with small data | Moderate | Good | Excellent |

#### K-Means vs. GMM vs. DBSCAN

| Criterion | K-Means | GMM | DBSCAN |
|---|---|---|---|
| Cluster shape | Spherical only | Elliptical | Arbitrary |
| Cluster assignment | Hard (point belongs to exactly one) | Soft (point has a probability per cluster) | Hard (core, border, or noise) |
| Number of clusters | Must specify k | Must specify number of components | Auto-detected (based on density) |
| Handles noise/outliers | Poorly (forces every point into a cluster) | Moderately (low probability for outliers) | Explicitly (marks as noise) |
| Scalability | Excellent (O(n*k*d*i)) | Moderate (O(n*k*d^2*i)) | Good with spatial index (O(n*log n)) |

#### RNN vs. LSTM vs. Transformer

| Criterion | RNN | LSTM | Transformer |
|---|---|---|---|
| Long-range dependencies | Poor (vanishing gradients) | Good (gating mechanism) | Excellent (direct attention connections) |
| Parallelization | None (sequential) | None (sequential) | Full (processes all positions at once) |
| Training speed | Slow | Slower than RNN (more params) | Fast (parallelizable) |
| Memory per layer | O(hidden_size) | O(4 * hidden_size) | O(n^2) for self-attention |
| Best for | Short sequences, real-time inference | Medium-length sequences with dependencies | Long sequences, large datasets, state of the art |

---

## Quick Reference: Choosing a Model

This section gives a practical starting point. Always validate with experiments.

### Your data has labels (Supervised)

**Continuous target (Regression):**
1. Start with Linear Regression. Simple, fast, interpretable.
2. If relationships are nonlinear, try Decision Tree or Random Forest.
3. If you have lots of data and complex patterns, try Gradient Boosting or MLP.

**Categorical target (Classification):**
1. Start with Logistic Regression for binary classification.
2. For small data with many features, try Naive Bayes.
3. For interpretable decisions, try Decision Tree.
4. For best accuracy on tabular data, try Gradient Boosting.
5. For images, use CNN. For sequences, use LSTM or Transformer.

### Your data has no labels (Unsupervised)

**You want to find groups (Clustering):**
1. If clusters are roughly spherical, try K-Means.
2. If clusters may overlap or have different shapes, try GMM.
3. If clusters have arbitrary shapes and there is noise, try DBSCAN.
4. If you want to see the full hierarchy of clusters, try Hierarchical Clustering.

**You want to reduce dimensions:**
1. Start with PCA. Fast, deterministic, widely understood.
2. If you also have labels, consider LDA (supervised dimensionality reduction).

**You want to learn representations from text:**
1. For static word representations, use Word2Vec (historical) or GloVe.
2. For context-aware representations, use a pretrained Transformer (BERT, GPT).

---

## Key Takeaways

1. **Parametric vs Non-Parametric** is about whether the model size is fixed or grows with data. Parametric models are faster and use less memory, but less flexible. Non-parametric models are more flexible but slower and use more memory.

2. **Instance-Based vs Model-Based** is about how the model makes predictions. Instance-based models memorize and compare. Model-based models abstract and compute. Instance-based models are lazy (fast training, slow prediction). Model-based models are eager (slow training, fast prediction).

3. **No single model is best for everything.** The right model depends on your data size, data type, interpretability needs, and computational budget. Start simple and add complexity only when needed.

4. **Assumptions matter.** Every model makes assumptions about the data. If the data violates those assumptions, the model will perform poorly no matter how well you tune it.

5. **This repository focuses on implementation from first principles.** Understanding the math and code behind each model helps you choose better, debug faster, and explain decisions clearly in interviews and on the job.
