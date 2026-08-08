# Model Implementations: Cost, Loss, Gradients, and Update Rules

This document is a reference for all models in this repository. For each model, you will find the prediction formula, the cost or loss function, the gradient (if applicable), the parameter update rule, and key implementation notes. The notation is consistent across all models to make comparison easier.

---

## Notation

| Symbol | Meaning |
|---|---|
| $m$ | Number of training examples |
| $n$ | Number of features |
| $X$ | Feature matrix, shape $(m, n)$ |
| $y$ | True target vector, shape $(m,)$ |
| $\hat{y}$ | Predicted output vector, shape $(m,)$ |
| $W$ | Weight vector, shape $(n,)$ |
| $b$ | Bias term, scalar |
| $\alpha$ | Learning rate |
| $J$ | Cost function (scalar value to minimize) |
| $\sigma(z)$ | Sigmoid function: $\frac{1}{1 + e^{-z}}$ |
| $k$ | Number of classes or number of clusters |
| $p_k(x)$ | Probability that sample $x$ belongs to class $k$ |

---

## Supervised Learning

### Linear Regression

**Prediction:**

$$\hat{y} = XW + b$$

**Cost function:** Mean Squared Error with a factor of 1/2 for convenience when taking the derivative.

$$J(W, b) = \frac{1}{2m} \sum_{i=1}^{m} (\hat{y}^{(i)} - y^{(i)})^2$$

**Why this cost?** Squared error penalizes large errors much more than small ones (the penalty grows quadratically). This makes the model sensitive to outliers but ensures it prioritizes reducing the biggest mistakes. The convex shape guarantees a single global minimum.

**Gradients:**

$$\frac{\partial J}{\partial W} = \frac{1}{m} X^T(\hat{y} - y)$$

$$\frac{\partial J}{\partial b} = \frac{1}{m} \sum_{i=1}^{m} (\hat{y}^{(i)} - y^{(i)})$$

**Derivation:** The gradient of the MSE cost with respect to the prediction error is $(\hat{y} - y)$. By the chain rule, the gradient with respect to $W$ is $X^T$ times that error, scaled by $1/m$. The $1/2$ in the cost cancels with the 2 that appears when differentiating the squared term.

**Update rule:**

$$W := W - \alpha \frac{\partial J}{\partial W}$$

$$b := b - \alpha \frac{\partial J}{\partial b}$$

**Closed form (Normal Equation):** An alternative to gradient descent that solves directly for the optimal weights.

$$W = (X^TX)^{-1} X^T y$$

**Key notes:**

- The cost function is convex. Gradient descent always reaches the global minimum with a small enough learning rate.
- Feature scaling helps gradient descent converge faster because it makes the cost surface more spherical.
- The closed form is $O(n^3)$ due to matrix inversion. Gradient descent is $O(m \cdot n)$ per iteration.
- Use the closed form when $n < 10^4$. Use gradient descent for larger feature spaces.

---

### Ridge Regression

Adds L2 regularization (weight decay) to linear regression. The penalty shrinks all weights toward zero.

**Cost function:**

$$J(W, b) = \frac{1}{2m} \sum_{i=1}^{m} (\hat{y}^{(i)} - y^{(i)})^2 + \frac{\lambda}{2m} \sum_{j=1}^{n} W_j^2$$

$\lambda$ controls the regularization strength.

**Gradient for W (L2 term):**

$$\frac{\partial J}{\partial W} = \frac{1}{m} X^T(\hat{y} - y) + \frac{\lambda}{m} W$$

The gradient for $b$ is unchanged (bias is not regularized).

**Closed form:**

$$W = (X^TX + \lambda I)^{-1} X^T y$$

**Key notes:** The added $\lambda I$ term guarantees $X^TX + \lambda I$ is invertible, even when features are highly correlated. Ridge always produces a unique solution.

---

### Lasso Regression

Adds L1 regularization. The penalty can shrink weights to exactly zero, performing automatic feature selection.

**Cost function:**

$$J(W, b) = \frac{1}{2m} \sum_{i=1}^{m} (\hat{y}^{(i)} - y^{(i)})^2 + \frac{\lambda}{m} \sum_{j=1}^{n} |W_j|$$

**Subgradient (L1 is not differentiable at zero):**

$$\frac{\partial J}{\partial W_j} = \frac{1}{m} \sum_{i=1}^{m} (\hat{y}^{(i)} - y^{(i)}) X_{ij} + \frac{\lambda}{m} \cdot \text{sign}(W_j)$$

Where sign$(0)$ can be any value in $[-1, 1]$, typically taken as $0$.

**Update rule (coordinate descent):** Update one weight at a time with soft thresholding.

$$W_j := S\left( W_j - \alpha \frac{\partial \text{MSE}}{\partial W_j},\ \frac{\lambda}{m} \right)$$

Where $S(z, \gamma) = \text{sign}(z) \cdot \max(|z| - \gamma, 0)$ is the soft thresholding operator.

**Key notes:** Lasso produces sparse solutions. Use it when you believe only a few features are truly important. It is not differentiable at zero, so gradient descent needs subgradients. Coordinate descent is the standard optimization method.

---

### Logistic Regression

**Prediction (probability):**

$$z = XW + b$$

$$\hat{y} = \sigma(z) = \frac{1}{1 + e^{-z}}$$

The sigmoid $\sigma(z)$ maps any real number to $(0, 1)$. It is an S-shaped curve: outputs near 0.5 when $z$ is near 0, and saturates at 0 or 1 for large negative or positive $z$.

**Cost function:** Binary Cross-Entropy (Log Loss).

$$J(W, b) = -\frac{1}{m} \sum_{i=1}^{m} \left[ y^{(i)} \log(\hat{y}^{(i)}) + (1 - y^{(i)}) \log(1 - \hat{y}^{(i)}) \right]$$

**Why this cost?** It comes from maximum likelihood estimation under the Bernoulli distribution. When $y=1$, minimizing $-\log(\hat{y})$ pushes $\hat{y}$ toward 1. When $y=0$, minimizing $-\log(1-\hat{y})$ pushes $\hat{y}$ toward 0. The logarithm makes the penalty explode if the model is confidently wrong.

**Gradients:**

$$\frac{\partial J}{\partial W} = \frac{1}{m} X^T(\hat{y} - y)$$

$$\frac{\partial J}{\partial b} = \frac{1}{m} \sum_{i=1}^{m} (\hat{y}^{(i)} - y^{(i)})$$

**Derivation trick:** The sigmoid derivative is $\sigma'(z) = \sigma(z)(1 - \sigma(z))$. When combined with cross-entropy, the sigmoid derivative cancels, giving the same gradient form as linear regression. The only difference is that $\hat{y}$ has been passed through the sigmoid.

**Update rule:**

$$W := W - \alpha \frac{\partial J}{\partial W}$$

$$b := b - \alpha \frac{\partial J}{\partial b}$$

**Numerical stability:**

- For the sigmoid: use $\sigma(z) = 1/(1+e^{-z})$ when $z \ge 0$, and $\sigma(z) = e^z/(1+e^z)$ when $z < 0$.
- For the cost: clip $\hat{y}$ to $[10^{-15}, 1 - 10^{-15}]$ before computing $\log$.

**Key notes:** The cost function is convex. Unlike linear regression, there is no closed-form solution. Feature scaling helps convergence. Weight initialization matters: start with small weights (e.g., `np.random.randn(n) * 0.01`) to avoid saturating the sigmoid.

---

### Multiclass Logistic Regression (Softmax Regression)

For $k$ classes, the model has a weight matrix $W$ of shape $(n, k)$ and bias $b$ of shape $(k,)$.

**Prediction (class probabilities):**

$$z = XW + b \quad \text{(shape: } m \times k\text{)}$$

$$\hat{y}_k = \text{softmax}(z)_k = \frac{e^{z_k}}{\sum_{j=1}^{k} e^{z_j}}$$

The softmax converts raw scores into a probability distribution over $k$ classes. It is a generalization of the sigmoid to multiple classes. Higher scores get exponentially more probability mass.

**Cost function:** Categorical Cross-Entropy.

$$J(W, b) = -\frac{1}{m} \sum_{i=1}^{m} \sum_{c=1}^{k} y_c^{(i)} \log(\hat{y}_c^{(i)})$$

Where $y_c^{(i)}$ is 1 if sample $i$ belongs to class $c$, and 0 otherwise (one-hot encoding).

**Gradients:**

$$\frac{\partial J}{\partial W} = \frac{1}{m} X^T(\hat{y} - y)$$

$$\frac{\partial J}{\partial b} = \frac{1}{m} \sum_{i=1}^{m} (\hat{y}^{(i)} - y^{(i)})$$

The gradient form matches binary logistic regression and linear regression. The softmax combined with cross-entropy produces this clean gradient, just as the sigmoid did for the binary case.

---

### Linear Discriminant Analysis (LDA)

LDA is a generative model. It models $P(X|y)$ for each class, then uses Bayes rule to get $P(y|X)$.

**Class prior:**

$$P(y = k) = \frac{m_k}{m}$$

Where $m_k$ is the number of samples in class $k$.

**Class mean:**

$$\mu_k = \frac{1}{m_k} \sum_{i: y^{(i)} = k} x^{(i)}$$

**Shared covariance matrix:**

$$\Sigma = \frac{1}{m - k} \sum_{c=1}^{k} \sum_{i: y^{(i)} = c} (x^{(i)} - \mu_c)(x^{(i)} - \mu_c)^T$$

**Decision function:** For classification, LDA finds the class that maximizes the discriminant score.

$$\delta_k(x) = x^T \Sigma^{-1} \mu_k - \frac{1}{2} \mu_k^T \Sigma^{-1} \mu_k + \log P(y = k)$$

**Prediction:** Assign the class with the highest discriminant score.

$$\hat{y} = \arg\max_k \delta_k(x)$$

**Key notes:** LDA has no iterative optimization. All parameters are computed in closed form. The decision boundary is linear because of the shared covariance assumption. LDA can also be used for dimensionality reduction (find directions that maximize class separation).

---

### Naive Bayes

Naive Bayes is a generative probabilistic classifier that applies Bayes theorem with the "naive" assumption that all features are independent given the class.

**Bayes rule for classification:**

$$P(y = k \mid x) = \frac{P(x \mid y = k) \cdot P(y = k)}{P(x)}$$

The denominator $P(x)$ is the same for all classes, so we only need the numerator for classification.

**With the independence assumption:**

$$P(x \mid y = k) = \prod_{j=1}^{n} P(x_j \mid y = k)$$

**Gaussian Naive Bayes** assumes each feature follows a normal distribution within each class.

$$P(x_j \mid y = k) = \frac{1}{\sqrt{2\pi \sigma_{k,j}^2}} \exp\left( -\frac{(x_j - \mu_{k,j})^2}{2\sigma_{k,j}^2} \right)$$

Where $\mu_{k,j}$ and $\sigma_{k,j}^2$ are the mean and variance of feature $j$ for class $k$, estimated from the training data.

**Prediction (log space for numerical stability):**

$$\hat{y} = \arg\max_k \left[ \log P(y = k) + \sum_{j=1}^{n} \log P(x_j \mid y = k) \right]$$

**Key notes:** Training is just counting and computing means and variances. There is no gradient descent or iterative optimization. The independence assumption is almost never true, but the model works surprisingly well for text classification where feature interactions are less critical. Use log probabilities to avoid underflow when multiplying many small probabilities.

---

### K-Nearest Neighbors (KNN)

KNN has no training phase and no cost function to optimize. It is a lazy learner.

**Prediction (regression):**

$$\hat{y} = \frac{1}{k} \sum_{i \in N_k(x)} y^{(i)}$$

**Prediction (classification):**

$$\hat{y} = \text{mode}\{ y^{(i)} : i \in N_k(x) \}$$

Where $N_k(x)$ is the set of $k$ nearest training samples to $x$.

**Distance metric (Euclidean):**

$$d(x, x^{(i)}) = \sqrt{\sum_{j=1}^{n} (x_j - x_j^{(i)})^2}$$

**Optimization:** There is no loss function to minimize during training. The only "learning" is choosing $k$ and the distance metric, which are typically tuned via cross-validation.

**Key notes:** Prediction time is $O(m \cdot n)$ per query because every training point must be checked. Feature scaling is critical: features with larger ranges dominate the distance calculation. The curse of dimensionality means KNN becomes less effective as the number of features grows. In high dimensions, all points become approximately equidistant.

---

### Decision Trees (CART)

Decision trees do not use gradient descent. They are built greedily by choosing splits that minimize impurity.

**Regression: Reduction in Variance**

For a node with data $D$ and a candidate split into left $D_L$ and right $D_R$:

$$\text{Gain} = \text{Var}(D) - \left( \frac{|D_L|}{|D|} \text{Var}(D_L) + \frac{|D_R|}{|D|} \text{Var}(D_R) \right)$$

Where $\text{Var}(D)$ is the variance of target values in $D$.

**Classification: Gini Impurity**

$$Gini(D) = 1 - \sum_{c=1}^{k} p_c^2$$

Where $p_c$ is the proportion of samples in class $c$ at node $D$. Gini is 0 when all samples belong to one class (perfect purity) and is maximized when classes are evenly distributed.

**Classification: Entropy (Information Gain)**

$$Entropy(D) = -\sum_{c=1}^{k} p_c \log_2(p_c)$$

$$\text{Information Gain} = Entropy(D) - \left( \frac{|D_L|}{|D|} Entropy(D_L) + \frac{|D_R|}{|D|} Entropy(D_R) \right)$$

**Split selection:** For each feature, try every possible threshold. Choose the (feature, threshold) pair that maximizes gain.

**Prediction (regression):** Average target value of training samples in the leaf.

**Prediction (classification):** Majority class of training samples in the leaf.

**Key notes:** Trees are built top-down and greedily. Each split is locally optimal, not globally optimal. Without constraints (max depth, min samples per leaf), trees overfit perfectly. Gini and Entropy usually produce similar trees. Gini is faster to compute (no log).

---

### Random Forest

A Random Forest is an ensemble of decision trees. It does not optimize a single cost function. It reduces variance through averaging.

**Prediction (regression):**

$$\hat{y} = \frac{1}{T} \sum_{t=1}^{T} f_t(x)$$

**Prediction (classification):**

$$\hat{y} = \text{mode}\{ f_1(x), f_2(x), ..., f_T(x) \}$$

Where $f_t$ is the prediction of tree $t$, and $T$ is the total number of trees.

**Randomness is introduced at two levels:**

1. **Bootstrap sampling:** Each tree is trained on a random sample of $m$ examples drawn with replacement from the training set. On average, about 63.2% of the original samples appear in each bootstrap. The remaining 36.8% are Out-of-Bag (OOB) samples used for unbiased error estimation.
2. **Feature subsampling:** At each split, only a random subset of $p$ features is considered (typically $p = \sqrt{n}$ for classification, $p = n/3$ for regression).

**Out-of-Bag (OOB) Error:** For each sample, average the predictions of trees that did not see it during training. This gives an unbiased estimate of test error without needing a separate validation set.

**Key notes:** Increasing the number of trees never causes overfitting (only diminishing returns). More trees reduce variance but increase computation. The OOB error is a free validation score. Random Forests work well out of the box with default hyperparameters.

---

### Gradient Boosting

Gradient boosting builds trees sequentially. Each new tree tries to correct the errors of the current ensemble by fitting the negative gradient of the loss.

**General form:**

$$F_0(x) = \arg\min_\gamma \sum_{i=1}^{m} L(y^{(i)}, \gamma)$$

For iteration $t = 1, 2, ..., T$:

Compute pseudo-residuals from the negative gradient:

$$r_t^{(i)} = -\left[ \frac{\partial L(y^{(i)}, F(x^{(i)}))}{\partial F(x^{(i)})} \right]_{F = F_{t-1}}$$

Fit a tree $h_t(x)$ to the pseudo-residuals. Update:

$$F_t(x) = F_{t-1}(x) + \eta \cdot h_t(x)$$

Where $\eta$ is the learning rate (shrinkage).

**Regression with MSE loss:** $L = \frac{1}{2}(y - F)^2$

$$r_t^{(i)} = y^{(i)} - F_{t-1}(x^{(i)})$$

The pseudo-residuals simplify to the actual residuals. Each tree fits the mistakes of the current model.

**Binary classification with Log Loss:** $L = -[y \log(p) + (1-y) \log(1-p)]$

First compute probabilities: $p = \sigma(F)$

$$r_t^{(i)} = y^{(i)} - p^{(i)}$$

Each tree fits the difference between true labels and predicted probabilities.

**Key notes:** Unlike Random Forest, boosting can overfit if too many trees are added. The learning rate $\eta$ and number of trees $T$ trade off: smaller $\eta$ requires more trees. Gradient boosting uses shallow trees (depth 1 to 6) as weak learners. XGBoost and LightGBM add regularization terms directly into the tree-splitting objective.

---

### Support Vector Machine (Linear)

SVMs find the maximum-margin separating hyperplane.

**Decision function:**

$$f(x) = W^T x + b$$

**Prediction:** $\hat{y} = \text{sign}(f(x))$

**Hinge Loss (L2-regularized):**

$$J(W, b) = \frac{\lambda}{2} \|W\|^2 + \frac{1}{m} \sum_{i=1}^{m} \max(0, 1 - y^{(i)} (W^T x^{(i)} + b))$$

**Why hinge loss?** The term $\max(0, 1 - y \cdot f(x))$ is zero when the sample is correctly classified with a margin of at least 1. Samples inside the margin or misclassified contribute linear penalty. This creates a "margin" around the decision boundary where the model ignores points that are already far from the boundary. Only support vectors (points on or inside the margin) affect the solution.

**Gradient (with respect to W):**

$$\frac{\partial J}{\partial W} = \lambda W + \frac{1}{m} \sum_{i \in \text{margin}} -y^{(i)} x^{(i)}$$

Where the sum is only over samples where $1 - y^{(i)}f(x^{(i)}) > 0$ (violations of the margin).

**Update rule:**

$$W := W - \alpha \left( \lambda W - \frac{1}{m} \sum_{i \in \text{violations}} y^{(i)} x^{(i)} \right)$$

**Key notes:** The primal formulation above uses gradient descent. SVMs are more commonly solved via the dual formulation using quadratic programming, which allows the kernel trick. Only support vectors matter after training. All other training points can be discarded. Feature scaling is critical for SVMs.

---

### Support Vector Machine (Kernel / RBF)

The kernel trick replaces the dot product $x_i^T x_j$ with a kernel function $K(x_i, x_j)$, allowing nonlinear decision boundaries.

**RBF (Gaussian) Kernel:**

$$K(x_i, x_j) = \exp\left( -\frac{\|x_i - x_j\|^2}{2\sigma^2} \right) = \exp\left( -\gamma \|x_i - x_j\|^2 \right)$$

Where $\gamma = 1/(2\sigma^2)$ controls the influence radius of each training point.

**Decision function (dual form):**

$$f(x) = \sum_{i=1}^{m} \alpha_i y^{(i)} K(x^{(i)}, x) + b$$

Where $\alpha_i$ are the dual coefficients. Only support vectors have $\alpha_i > 0$.

**Dual optimization (soft margin):**

$$\max_\alpha \sum_{i=1}^{m} \alpha_i - \frac{1}{2} \sum_{i=1}^{m} \sum_{j=1}^{m} \alpha_i \alpha_j y^{(i)} y^{(j)} K(x^{(i)}, x^{(j)})$$

Subject to $0 \le \alpha_i \le C$ and $\sum \alpha_i y^{(i)} = 0$.

**Key notes:** The RBF kernel computes similarity in an implicit infinite-dimensional space. High $\gamma$ means each point only influences its immediate neighborhood (complex boundary, risk of overfitting). High $C$ means harder margin (risk of overfitting). The kernel matrix is $m \times m$, making training $O(m^3)$ or $O(m^2)$ with SMO. Not suitable for very large datasets.

---

## Unsupervised Learning

### K-Means Clustering

K-Means minimizes the within-cluster sum of squared distances (inertia).

**Objective (distortion):**

$$J = \sum_{i=1}^{m} \sum_{k=1}^{K} r_{ik} \|x^{(i)} - \mu_k\|^2$$

Where $r_{ik} = 1$ if point $i$ belongs to cluster $k$, and 0 otherwise. $\mu_k$ is the centroid of cluster $k$.

**Algorithm (Lloyd's):**

1. **Assignment step:** Assign each point to the nearest centroid.
   $$r_{ik} = \begin{cases} 1 & \text{if } k = \arg\min_j \|x^{(i)} - \mu_j\|^2 \\ 0 & \text{otherwise} \end{cases}$$

2. **Update step:** Recompute centroids as the mean of assigned points.
   $$\mu_k = \frac{\sum_{i=1}^{m} r_{ik} x^{(i)}}{\sum_{i=1}^{m} r_{ik}}$$

**Convergence:** The objective $J$ decreases monotonically at each step. The algorithm converges to a local minimum (not necessarily global). Both steps together are exactly the Expectation-Maximization (EM) algorithm for a Gaussian mixture with equal, spherical covariances.

**Key notes:** The result depends on initial centroid placement. K-Means++ initialization spreads initial centroids apart, giving better and more consistent results. The algorithm assumes spherical, equally sized clusters. Feature scaling is important because Euclidean distance is used.

**K-Means++ initialization:**

1. Choose the first centroid uniformly at random from the data.
2. For each remaining centroid, choose a point $x$ with probability proportional to $D(x)^2$, where $D(x)$ is the distance from $x$ to the nearest already-chosen centroid.
3. Repeat until $K$ centroids are chosen.

---

### Gaussian Mixture Models (GMM)

GMMs model the data as a mixture of $K$ Gaussian distributions. Unlike K-Means, GMMs provide soft cluster assignments (each point has a probability of belonging to each cluster).

**Model:**

$$p(x) = \sum_{k=1}^{K} \pi_k \cdot \mathcal{N}(x \mid \mu_k, \Sigma_k)$$

Where $\pi_k$ is the mixing coefficient (prior probability of cluster $k$), with $\sum_{k=1}^{K} \pi_k = 1$.

**Gaussian density:**

$$\mathcal{N}(x \mid \mu_k, \Sigma_k) = \frac{1}{(2\pi)^{n/2} |\Sigma_k|^{1/2}} \exp\left( -\frac{1}{2} (x - \mu_k)^T \Sigma_k^{-1} (x - \mu_k) \right)$$

**Log-likelihood (cost to maximize):**

$$\log P(X \mid \pi, \mu, \Sigma) = \sum_{i=1}^{m} \log \left( \sum_{k=1}^{K} \pi_k \cdot \mathcal{N}(x^{(i)} \mid \mu_k, \Sigma_k) \right)$$

**Why EM instead of gradient descent?** The log of a sum makes direct differentiation messy. The Expectation-Maximization algorithm separates the problem into two simpler steps.

**EM Algorithm:**

**E-step (Expectation):** Compute responsibilities (posterior probability that point $i$ belongs to cluster $k$).

$$\gamma_{ik} = \frac{\pi_k \cdot \mathcal{N}(x^{(i)} \mid \mu_k, \Sigma_k)}{\sum_{j=1}^{K} \pi_j \cdot \mathcal{N}(x^{(i)} \mid \mu_j, \Sigma_j)}$$

**M-step (Maximization):** Update parameters using the responsibilities.

$$N_k = \sum_{i=1}^{m} \gamma_{ik}$$

$$\mu_k = \frac{1}{N_k} \sum_{i=1}^{m} \gamma_{ik} x^{(i)}$$

$$\Sigma_k = \frac{1}{N_k} \sum_{i=1}^{m} \gamma_{ik} (x^{(i)} - \mu_k)(x^{(i)} - \mu_k)^T$$

$$\pi_k = \frac{N_k}{m}$$

**Key notes:** GMM generalizes K-Means. If $\Sigma_k = \sigma^2 I$ for all $k$ and $\sigma \to 0$, the EM algorithm becomes exactly the K-Means algorithm. The covariance matrix type (full, tied, diagonal, spherical) is an important hyperparameter. As with K-Means, initialization matters. Use K-Means to initialize the means.

---

### Principal Component Analysis (PCA)

PCA finds orthogonal directions (principal components) that maximize variance in the data. It is solved via eigendecomposition, not gradient descent.

**Objective:** Find projection matrix $W_{n \times d}$ (with $d < n$) that maximizes the variance of the projected data. Equivalently, minimize the reconstruction error.

$$\text{Reconstruction Error} = \frac{1}{m} \sum_{i=1}^{m} \|x^{(i)} - \hat{x}^{(i)}\|^2$$

Where $\hat{x}^{(i)} = W W^T x^{(i)}$ is the reconstruction of $x^{(i)}$ from its lower-dimensional representation.

**Steps:**

1. Center the data: $X_c = X - \bar{X}$ (subtract the mean of each feature).
2. Compute the covariance matrix: $\Sigma = \frac{1}{m} X_c^T X_c$, shape $(n, n)$.
3. Eigendecomposition: $\Sigma = V \Lambda V^T$, where columns of $V$ are eigenvectors and $\Lambda$ is a diagonal matrix of eigenvalues.
4. Sort eigenvectors by decreasing eigenvalues. Keep the top $d$ eigenvectors: $W = V_{:, :d}$.
5. Project: $Z = X_c W$.

**Explained variance ratio:**

$$\text{EV}_j = \frac{\lambda_j}{\sum_{i=1}^{n} \lambda_i}$$

Where $\lambda_j$ is the eigenvalue of the $j$-th principal component.

**Key notes:** PCA assumes variance equals information. Directions of maximum variance may not separate classes well (PCA is unsupervised). Always standardize features before PCA. Use `np.linalg.eigh` for symmetric matrices (faster and numerically stable). PCA can be used for whitening: divide each projected dimension by $\sqrt{\lambda_j}$.

---

### Hierarchical Clustering

Hierarchical clustering builds a tree of clusters (dendrogram). It does not optimize a global cost function. Instead, it follows a greedy linkage strategy.

**Agglomerative (bottom-up) approach:**

1. Start with each point as its own cluster.
2. Repeatedly merge the two closest clusters until one cluster remains.
3. Cut the dendrogram at a chosen height to get the desired number of clusters.

**Linkage criteria** define the distance between two clusters $A$ and $B$:

| Linkage | Formula | Description |
|---|---|---|
| Single | $\min_{a \in A, b \in B} d(a, b)$ | Minimum distance between any two points. Produces elongated, chain-like clusters. Sensitive to noise. |
| Complete | $\max_{a \in A, b \in B} d(a, b)$ | Maximum distance between any two points. Produces compact, spherical clusters. |
| Average | $\frac{1}{\|A\| \|B\|} \sum_{a \in A} \sum_{b \in B} d(a, b)$ | Average distance between all pairs. A compromise between single and complete. |
| Ward | $\sqrt{\frac{2\|A\|\|B\|}{\|A\|+\|B\|}} \cdot \|\mu_A - \mu_B\|$ | Minimizes the increase in within-cluster variance after merging. Produces compact clusters of similar size. Tends to work well in practice. |

**Key notes:** Once a merge is made, it cannot be undone. The choice of linkage and distance metric dramatically changes the result. Ward's method assumes Euclidean distance. The dendrogram gives you flexibility: you do not need to know the number of clusters ahead of time.

---

## Deep Learning

### Multi-Layer Perceptron (MLP)

An MLP applies alternating linear transformations and nonlinear activations.

**Forward pass (one hidden layer example):**

$$Z^{[1]} = X W^{[1]} + b^{[1]}$$

$$A^{[1]} = g(Z^{[1]})$$

$$Z^{[2]} = A^{[1]} W^{[2]} + b^{[2]}$$

$$\hat{y} = \sigma(Z^{[2]}) \text{ (for binary classification)}$$

Where $g$ is an activation function (ReLU, tanh, sigmoid).

**Common activation functions and their derivatives:**

| Activation | Function $g(z)$ | Derivative $g'(z)$ |
|---|---|---|
| Sigmoid | $\frac{1}{1 + e^{-z}}$ | $g(z)(1 - g(z))$ |
| Tanh | $\frac{e^z - e^{-z}}{e^z + e^{-z}}$ | $1 - g(z)^2$ |
| ReLU | $\max(0, z)$ | $1$ if $z > 0$, else $0$ |
| Leaky ReLU | $\max(0.01z, z)$ | $1$ if $z > 0$, else $0.01$ |

**Loss functions (output layer):**

| Task | Loss | Formula |
|---|---|---|
| Binary classification | Binary Cross-Entropy | $-\frac{1}{m} \sum [y \log(\hat{y}) + (1-y) \log(1-\hat{y})]$ |
| Multiclass classification | Categorical Cross-Entropy | $-\frac{1}{m} \sum \sum_{c} y_c \log(\hat{y}_c)$ |
| Regression | MSE | $\frac{1}{2m} \sum (\hat{y} - y)^2$ |

**Backpropagation:** Compute gradients from output to input using the chain rule. For layer $l$:

Define the error signal at the output of layer $l$ as $\delta^{[l]} = \frac{\partial J}{\partial Z^{[l]}}$.

**Output layer error (binary cross-entropy + sigmoid):**

$$\delta^{[L]} = \hat{y} - y$$

**Gradient for weights and biases at layer $l$:**

$$\frac{\partial J}{\partial W^{[l]}} = \frac{1}{m} (A^{[l-1]})^T \delta^{[l]}$$

$$\frac{\partial J}{\partial b^{[l]}} = \frac{1}{m} \sum \delta^{[l]}$$

**Backpropagate the error to the previous layer:**

$$\delta^{[l-1]} = \delta^{[l]} (W^{[l]})^T \odot g'(Z^{[l-1]})$$

Where $\odot$ is element-wise multiplication.

**Update rule:**

$$W^{[l]} := W^{[l]} - \alpha \frac{\partial J}{\partial W^{[l]}}$$

$$b^{[l]} := b^{[l]} - \alpha \frac{\partial J}{\partial b^{[l]}}$$

**Key notes:** Proper weight initialization is critical for deep networks. He initialization (for ReLU): $W \sim \mathcal{N}(0, \sqrt{2/n_{in}})$. Xavier/Glorot initialization (for tanh/sigmoid): $W \sim \mathcal{N}(0, \sqrt{2/(n_{in} + n_{out})})$. Always check gradient magnitudes. Use gradient checking during development: compare analytical gradients with finite-difference approximations.

---

### Convolutional Neural Network (CNN)

CNNs use convolution operations instead of full matrix multiplication. This introduces parameter sharing and sparse connectivity.

**2D Convolution (single channel, single filter):**

$$(X * K)_{i,j} = \sum_{p=0}^{k_h-1} \sum_{q=0}^{k_w-1} X_{i+p, j+q} \cdot K_{p,q}$$

Where $K$ is the kernel (filter) of size $k_h \times k_w$.

**Output size after convolution:**

$$H_{out} = \left\lfloor \frac{H_{in} + 2P - k_h}{S} \right\rfloor + 1$$

Where $P$ is padding and $S$ is stride.

**Max Pooling:**

$$Y_{i,j} = \max_{p=0}^{h-1} \max_{q=0}^{w-1} X_{i \cdot S + p,\ j \cdot S + q}$$

**Backpropagation through convolution:** The gradient with respect to the kernel is also a convolution (flipped kernel on the input error). The gradient with respect to the input is a full convolution (padded kernel on the output error).

**Gradient for kernel weights:**

$$\frac{\partial J}{\partial K_{p,q}} = \sum_i \sum_j \frac{\partial J}{\partial Z_{i,j}} \cdot X_{i+p, j+q}$$

**Gradient for the input (backpropagation):**

$$\frac{\partial J}{\partial X_{i,j}} = \sum_p \sum_q \frac{\partial J}{\partial Z_{i-p, j-q}} \cdot K_{p,q}$$

(With appropriate padding and handling of boundaries.)

**Backpropagation through max pooling:** The gradient is passed only through the position that had the maximum value during the forward pass. All other positions receive zero gradient.

**Key notes:** CNNs exploit translational invariance: the same pattern can appear anywhere. Parameter sharing means a $3 \times 3$ kernel uses only 9 parameters regardless of input size. This drastically reduces overfitting. The receptive field grows with depth: a neuron in layer $L$ "sees" a larger region of the original input as you go deeper.

---

### Recurrent Neural Network (RNN)

An RNN processes sequences by maintaining a hidden state that is updated at each time step.

**Forward pass (single time step $t$):**

$$h_t = \tanh(W_{hh} h_{t-1} + W_{xh} x_t + b_h)$$

$$\hat{y}_t = W_{hy} h_t + b_y$$

Where $h_t$ is the hidden state at time $t$, $x_t$ is the input at time $t$, $W_{hh}$ is the recurrent weight (hidden-to-hidden), $W_{xh}$ is the input-to-hidden weight.

**Loss (across all time steps):**

$$J = \sum_{t=1}^{T} J_t(\hat{y}_t, y_t)$$

**Backpropagation Through Time (BPTT):** Unroll the RNN across all time steps and apply backpropagation. The gradient through the recurrent connection involves a product of $t$ Jacobian matrices:

$$\frac{\partial J}{\partial h_0} = \sum_{t=1}^{T} \frac{\partial J_t}{\partial h_t} \cdot \prod_{k=1}^{t-1} \frac{\partial h_{k+1}}{\partial h_k}$$

Where $\frac{\partial h_{k+1}}{\partial h_k} = W_{hh}^T \odot (1 - h_{k+1}^2)$ (due to tanh).

**Why vanishing gradients happen:** The repeated multiplication by $W_{hh}$ means the gradient decays exponentially. If eigenvalues of $W_{hh}$ are less than 1, the gradient vanishes. If greater than 1, it explodes.

**Gradient clipping (for exploding gradients):**

$$g := g \cdot \min\left(1, \frac{\text{threshold}}{\|g\|}\right)$$

Scale the gradient down if its norm exceeds the threshold.

**Key notes:** Standard RNNs cannot capture long-range dependencies (more than about 10 to 20 time steps) due to vanishing gradients. Truncated BPTT limits the unroll length for computational efficiency. Gradient clipping is essential for stable training.

---

### Long Short-Term Memory (LSTM)

LSTMs add gates to control information flow, allowing gradients to flow unchanged across many time steps.

**Forward pass (single time step $t$):**

**Forget gate:** Controls what to discard from the cell state.
$$f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$$

**Input gate:** Controls what new information to store.
$$i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$$

**Candidate cell state:** New information to potentially add.
$$\tilde{c}_t = \tanh(W_c \cdot [h_{t-1}, x_t] + b_c)$$

**Cell state update:** Combine old state (gated by forget gate) and new candidate (gated by input gate).
$$c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$$

**Output gate:** Controls what to output from the cell state.
$$o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)$$

**Hidden state:** Filtered version of the cell state.
$$h_t = o_t \odot \tanh(c_t)$$

**Why LSTMs avoid vanishing gradients:** The cell state $c_t$ has an additive update path. When $f_t \approx 1$ and $i_t \approx 0$, $c_t \approx c_{t-1}$, meaning the gradient can flow unchanged through time. The forget gate learns to keep important long-term information and forget irrelevant short-term information.

**Gradients:** Backpropagation through an LSTM involves the chain rule through each gate. The full derivation is lengthy but follows the same pattern as RNN backpropagation.

**Key notes:** LSTM has four times as many parameters as a vanilla RNN with the same hidden size. The forget gate bias is typically initialized to 1 (or a positive value) to encourage remembering by default. LSTMs can handle sequences of hundreds of time steps, far more than vanilla RNNs.

---

## NLP and Representation Learning

### N-Gram Language Model

An N-gram model estimates the probability of a word given the previous $n-1$ words. It is a count-based model with no gradient-based optimization.

**Chain rule of probability:**

$$P(w_1, w_2, ..., w_T) = \prod_{t=1}^{T} P(w_t \mid w_{t-n+1}, ..., w_{t-1})$$

**Maximum Likelihood Estimate (MLE):**

$$P(w_t \mid w_{t-n+1}^{t-1}) = \frac{\text{count}(w_{t-n+1}, ..., w_{t-1}, w_t)}{\text{count}(w_{t-n+1}, ..., w_{t-1})}$$

**Perplexity (evaluation metric):**

$$\text{PPL} = \exp\left( -\frac{1}{T} \sum_{t=1}^{T} \log P(w_t \mid w_{t-n+1}^{t-1}) \right)$$

Lower perplexity means the model is less surprised by the text (better).

**Smoothing (Laplace / Add-One):**

$$P(w_t \mid w_{t-1}) = \frac{\text{count}(w_{t-1}, w_t) + 1}{\text{count}(w_{t-1}) + V}$$

Where $V$ is the vocabulary size. Smoothing prevents zero probabilities for unseen n-grams.

**Key notes:** The number of possible n-grams grows exponentially with $n$. Most n-grams never appear in the training data (sparsity problem), so smoothing is essential. N-gram models cannot generalize to unseen word combinations. They were the state of the art in NLP before neural methods.

---

### Word2Vec (Skip-Gram with Negative Sampling)

Word2Vec learns dense vector representations of words by predicting context words from a target word (skip-gram) or vice versa (CBOW).

**Skip-Gram objective:** Given a target word $w$, predict surrounding context words within a window of size $c$.

**Softmax formulation (original, computationally expensive):**

$$P(w_{out} \mid w_{in}) = \frac{\exp(v_{w_{out}}^T v_{w_{in}})}{\sum_{w=1}^{V} \exp(v_w^T v_{w_{in}})}$$

**Negative Sampling:** A more efficient approximation. Instead of computing the full softmax over the entire vocabulary, train the model to distinguish the true context word from a few randomly sampled "negative" words.

**Loss (for one target-context pair):**

$$J = -\log \sigma(v_c^T v_t) - \sum_{k=1}^{K} \mathbb{E}_{w \sim P_n(w)} [\log \sigma(-v_w^T v_t)]$$

Where $v_t$ is the target word vector, $v_c$ is the context word vector, $K$ is the number of negative samples (typically 5 to 20), and $P_n(w)$ is the noise distribution (usually unigram frequency raised to the power 3/4).

**Gradient for target vector $v_t$:**

$$\frac{\partial J}{\partial v_t} = (\sigma(v_c^T v_t) - 1) v_c + \sum_{k=1}^{K} (1 - \sigma(-v_w^T v_t)) (-v_w)$$

**Gradient for context vector $v_c$:**

$$\frac{\partial J}{\partial v_c} = (\sigma(v_c^T v_t) - 1) v_t$$

**Update rule (SGD):**

$$v_t := v_t - \alpha \frac{\partial J}{\partial v_t}$$

$$v_c := v_c - \alpha \frac{\partial J}{\partial v_c}$$

(Different learning rates may be used for each.)

**Key notes:** Word2Vec produces two vectors per word: target (input) vectors and context (output) vectors. The final embedding is typically the target vector or the average of both. The famous "king - man + woman = queen" analogy comes from the linear structure of the embedding space. Word2Vec embeddings are static: each word has one vector regardless of context.

---

### Attention Mechanism

Attention computes a weighted sum of values, where the weights reflect how relevant each position is to the current query.

**Scaled Dot-Product Attention:**

Given queries $Q$, keys $K$, and values $V$ (all matrices of shape $m \times d$):

$$\text{Attention}(Q, K, V) = \text{softmax}\left( \frac{QK^T}{\sqrt{d_k}} \right) V$$

**Step by step:**

1. Compute alignment scores: $S = QK^T$ (shape $m \times m$). Each entry $S_{ij}$ measures how relevant position $j$ is to position $i$.
2. Scale: $\frac{S}{\sqrt{d_k}}$ prevents dot products from growing too large in high dimensions, which would push the softmax into regions of tiny gradients.
3. Normalize: $A = \text{softmax}(S / \sqrt{d_k})$ along each row. This gives attention weights that sum to 1.
4. Weight values: $O = A V$. Each output is a weighted combination of all values.

**Why scaling by $\sqrt{d_k}$?** If $q$ and $k$ are independent random vectors with mean 0 and variance 1, their dot product has mean 0 and variance $d_k$. Without scaling, large $d_k$ makes the softmax saturate (near one-hot), which kills gradients.

**Gradients:** The softmax and matrix multiplications are all standard operations. Backpropagation follows the usual rules for matmul and softmax.

**Key notes:** Self-attention means $Q$, $K$, $V$ are all derived from the same input (through different learned linear projections). Cross-attention means $Q$ comes from one source and $K$, $V$ from another. The $O(m^2)$ memory cost from the attention matrix limits sequence length.

---

### Transformer

The Transformer processes sequences using only attention, without any recurrence or convolution. The original architecture has an encoder and decoder. Modern models (GPT) use decoder-only architectures.

**Multi-Head Attention:** Run $h$ attention heads in parallel, then concatenate and project.

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h) W^O$$

$$\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$

Where $W_i^Q, W_i^K, W_i^V$ are learned projection matrices per head, and $W^O$ combines all heads.

**Positional Encoding:** Since attention has no notion of order, position information is added to the input embeddings.

$$PE_{(pos, 2i)} = \sin\left( \frac{pos}{10000^{2i/d}} \right)$$

$$PE_{(pos, 2i+1)} = \cos\left( \frac{pos}{10000^{2i/d}} \right)$$

Where $pos$ is the position and $i$ is the dimension index.

**Feed-Forward Network (per position, same weights across positions):**

$$\text{FFN}(x) = \text{ReLU}(xW_1 + b_1)W_2 + b_2$$

(Or GELU in newer architectures.)

**Layer Normalization:** Normalizes activations across the feature dimension for each sample independently.

$$\text{LayerNorm}(x) = \gamma \odot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta$$

Where $\mu$ and $\sigma^2$ are the mean and variance computed across features, and $\gamma$, $\beta$ are learnable parameters.

**Residual connections:** Add the input of each sublayer to its output before layer normalization.

$$\text{Output} = \text{LayerNorm}(x + \text{Sublayer}(x))$$

**Loss (Transformer Language Model / GPT):** Causal language modeling with cross-entropy.

$$J = -\frac{1}{T} \sum_{t=1}^{T} \log P(w_t \mid w_1, ..., w_{t-1})$$

Where $P(w_t)$ is the softmax output of the final linear layer.

**Causal masking:** To prevent attending to future tokens, the attention scores for positions $j > i$ are set to $-\infty$ before the softmax.

**Key notes:** Training a Transformer from scratch requires massive data and compute. In practice, most people fine-tune pre-trained models. The quadratic complexity in sequence length means scaling to long contexts requires architectural modifications (sparse attention, linear attention, FlashAttention). Layer norm placement differs between architectures (pre-norm in GPT, post-norm in the original paper).

---

## Optimization Algorithms

These algorithms are used to update model parameters once gradients are computed.

### Gradient Descent (Vanilla)

$$W := W - \alpha \frac{\partial J}{\partial W}$$

**Key notes:** Uses the full dataset to compute each gradient step. Can be slow for large datasets. Guaranteed to converge to a local minimum for convex functions with appropriate learning rate schedules.

### Stochastic Gradient Descent (SGD)

$$W := W - \alpha \frac{\partial J}{\partial W}^{(i)}$$

Where the gradient is computed from a single random sample $i$. High variance in updates but can escape shallow local minima.

### Mini-Batch Gradient Descent

$$W := W - \alpha \frac{1}{|B|} \sum_{i \in B} \frac{\partial J}{\partial W}^{(i)}$$

Where $B$ is a mini-batch of samples. Balances gradient accuracy and computational efficiency. This is what most people mean by "SGD" in practice.

### SGD with Momentum

$$v := \beta v + (1 - \beta) \frac{\partial J}{\partial W}$$

$$W := W - \alpha v$$

Momentum accumulates a velocity vector. This smooths oscillations and accelerates convergence in directions with consistent gradients. $\beta$ is typically 0.9.

### Adam (Adaptive Moment Estimation)

Combines momentum and adaptive learning rates per parameter.

$$m := \beta_1 m + (1 - \beta_1) \frac{\partial J}{\partial W} \quad \text{(first moment: mean)}$$

$$v := \beta_2 v + (1 - \beta_2) \left(\frac{\partial J}{\partial W}\right)^2 \quad \text{(second moment: uncentered variance)}$$

$$\hat{m} = \frac{m}{1 - \beta_1^t} \quad \text{(bias correction)}$$

$$\hat{v} = \frac{v}{1 - \beta_2^t}$$

$$W := W - \alpha \frac{\hat{m}}{\sqrt{\hat{v}} + \epsilon}$$

Default values: $\alpha = 0.001$, $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$.

**Key notes:** Adam is the default optimizer for most deep learning tasks. It is less sensitive to the learning rate choice than vanilla SGD. However, Adam can sometimes generalize worse than well-tuned SGD with momentum, especially for image classification tasks.

---

## Quick Reference: Cost Functions by Problem Type

| Problem Type | Cost / Loss Function | Formula |
|---|---|---|
| Regression | Mean Squared Error | $\frac{1}{2m} \sum (\hat{y} - y)^2$ |
| Regression | Mean Absolute Error | $\frac{1}{m} \sum \|\hat{y} - y\|$ |
| Binary Classification | Binary Cross-Entropy | $-\frac{1}{m} \sum [y \log(\hat{y}) + (1-y) \log(1-\hat{y})]$ |
| Multiclass Classification | Categorical Cross-Entropy | $-\frac{1}{m} \sum \sum_c y_c \log(\hat{y}_c)$ |
| SVM Classification | Hinge Loss | $\frac{1}{m} \sum \max(0, 1 - y \cdot f(x))$ |
| Clustering (K-Means) | Within-Cluster SSE | $\sum \sum r_{ik} \|x^{(i)} - \mu_k\|^2$ |
| GMM | Negative Log-Likelihood | $-\sum \log(\sum \pi_k \mathcal{N}(x \mid \mu_k, \Sigma_k))$ |
| PCA | Reconstruction Error | $\frac{1}{m} \sum \|x - WW^T x\|^2$ |
| Word2Vec (NS) | Negative Sampling Loss | $-\log \sigma(v_c^T v_t) - \sum \log \sigma(-v_w^T v_t)$ |
| Language Model | Cross-Entropy (per token) | $-\frac{1}{T} \sum \log P(w_t \mid w_{<t})$ |

---

## Quick Reference: Gradient Update Rules

| Model | Gradient Form | Update Rule | Optimizer |
|---|---|---|---|
| Linear/Logistic Regression | $\frac{1}{m} X^T(\hat{y} - y)$ | $W := W - \alpha \cdot \text{grad}$ | GD / SGD |
| Ridge Regression | $\frac{1}{m} X^T(\hat{y} - y) + \frac{\lambda}{m} W$ | $W := W - \alpha \cdot \text{grad}$ | GD / SGD |
| Lasso Regression | $\frac{1}{m} X^T(\hat{y} - y) + \frac{\lambda}{m} \cdot \text{sign}(W)$ | $W_j := S(W_j - \alpha \cdot \text{grad})$ | Coordinate Descent |
| SVM (Primal) | $\lambda W - \frac{1}{m} \sum_{i \in \text{viol}} y^{(i)} x^{(i)}$ | $W := W - \alpha \cdot \text{grad}$ | GD / SGD |
| MLP (layer $l$) | $\frac{1}{m} (A^{[l-1]})^T \delta^{[l]}$ | $W^{[l]} := W^{[l]} - \alpha \cdot \text{grad}$ | SGD / Adam |
| Decision Tree | N/A (greedy split selection) | Choose (feature, threshold) that maximizes gain | Greedy search |
| Random Forest | N/A (ensemble of trees) | Build $T$ independent trees | Bootstrap + voting |
| Gradient Boosting | Fit tree to negative gradient of loss | $F_t := F_{t-1} + \eta \cdot h_t$ | Sequential trees |
| K-Means | N/A (alternating assignment/update) | $\mu_k = \text{mean of assigned points}$ | EM-style |
| GMM | N/A (EM algorithm) | E-step: compute responsibilities; M-step: update params | EM Algorithm |
| PCA | N/A (eigendecomposition) | $W = \text{top } d \text{ eigenvectors of } X^TX$ | Closed form |
| KNN / Naive Bayes / LDA | N/A (no iterative optimization) | N/A (count-based or closed-form solution) | Direct computation |
| Word2Vec | $\frac{\partial J}{\partial v} = (\sigma(v^T v') - 1) v'$ | $v := v - \alpha \cdot \text{grad}$ | SGD |
| Transformer | Standard backprop through attention + FFN | $W := W - \alpha \cdot \text{grad}$ | Adam / AdamW |
