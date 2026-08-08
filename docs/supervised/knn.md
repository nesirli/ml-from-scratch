# K-Nearest Neighbors (KNN)

KNN is a supervised learning algorithm. It can be used for both classification and regression. It is a non-parametric, instance-based, lazy learner. "Lazy" means it does not learn a model during training. It simply stores the training data and does all the work at prediction time.

## How It Works

The idea behind KNN is simple: similar data points are close to each other. To make a prediction for a new point, the algorithm finds the $k$ closest training points and uses their labels to decide the output.

- **For classification**: take a vote. The predicted class is the most common class among the $k$ neighbors.
- **For regression**: take an average. The predicted value is the mean of the target values of the $k$ neighbors.

## Prediction

Let $N_k(x)$ be the set of $k$ nearest neighbors of a new point $x$.

**Classification (mode):**

$$\hat{y} = \text{mode}\{ y^{(i)} : i \in N_k(x) \}$$

**Regression (mean):**

$$\hat{y} = \frac{1}{k} \sum_{i \in N_k(x)} y^{(i)}$$

Where $y^{(i)}$ is the label of training sample $i$.

## Distance Metric

To find the nearest neighbors, we need to measure distance. The most common choice is Euclidean distance:

$$d(x, x^{(i)}) = \sqrt{\sum_{j=1}^{n} (x_j - x_j^{(i)})^2}$$

- $n$ is the number of features
- $x_j$ is the $j$-th feature of the new point
- $x_j^{(i)}$ is the $j$-th feature of training point $i$

Other distance metrics like Manhattan distance can also be used, but Euclidean is the default.

## Cost Function

KNN has no cost function and no gradient descent. It is a lazy learner. There is nothing to optimize during training. The only "learning" is picking the right $k$ and the right distance metric, which is done through cross-validation.

## Algorithm Steps

### Training (fit)

1. Store the training data $X_{train}$ and labels $y_{train}$.
2. That is it. No computation, no weights to update.

### Prediction (predict)

1. For a new point $x$:
   - Compute the distance from $x$ to every point in $X_{train}$
   - Sort the distances and pick the $k$ smallest ones
   - Get the labels of those $k$ nearest neighbors
2. **If classification**: return the most common label (majority vote).
3. **If regression**: return the mean of the labels.

### Prediction with Probabilities (classification only)

Instead of returning just the class, return the fraction of neighbors for each class:

$$P(\text{class} = c) = \frac{\text{number of neighbors with label } c}{k}$$

## Hyperparameters

- **k** (`n_neighbors`): The number of neighbors to consider. Default is 3 or 5.

A small $k$ (like 1) makes the model sensitive to noise. It can overfit because a single noisy point can change the prediction. A large $k$ makes the model smoother but can underfit because it averages over too many points. The best $k$ is usually found by trying different values and using cross-validation.

## Notes

- **Feature scaling is critical.** KNN uses distances, so features with larger ranges (like salary in thousands) dominate features with smaller ranges (like age in years). Always standardize or normalize features before using KNN. Without scaling, the distance values are meaningless.

- **No training time, slow prediction.** Training is instant because the model just stores the data. But prediction is slow for large datasets. For each new point, the algorithm must compute distances to every training point. The time per prediction is $O(m \cdot n)$, where $m$ is the number of training samples and $n$ is the number of features.

- **Curse of dimensionality.** As the number of features grows, distances become less meaningful. In high-dimensional spaces, all points become roughly the same distance from each other. KNN works best with a small number of features (typically fewer than 20).

- **Choosing $k$.** Odd values of $k$ are preferred for binary classification to avoid ties. A common starting point is $k = \sqrt{m}$, where $m$ is the number of training samples. But the best way to choose $k$ is through cross-validation.

- **Memory usage.** The model must keep the entire training dataset in memory. For very large datasets, this can be a problem.

- **No assumptions about data shape.** KNN makes no assumptions about the underlying distribution of the data. This makes it flexible and able to capture complex decision boundaries. But it also means the model cannot generalize beyond the training data. If a region of the feature space has no training points, the model has no way to make a good prediction there.
