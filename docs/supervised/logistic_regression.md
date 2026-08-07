# Logistic Regression

Logistic regression is a supervised learning algorithm. It predicts the probability that a sample belongs to a class. Even though the name says "regression", it is used for binary classification.

## Model

The model first computes a linear score, then passes it through the sigmoid function to get a probability:

$$z = XW + b$$

$$\hat{y} = \sigma(z) = \frac{1}{1 + e^{-z}}$$

- $X$ is the input matrix with shape $(m, n)$
- $W$ is the weight vector with shape $(n,)$
- $b$ is the bias (a single number)
- $\sigma(z)$ is the sigmoid function that squashes the output between 0 and 1
- $\hat{y}$ is the predicted probability, shape $(m,)$

To get a class label, we apply a threshold (default 0.5):

$$\text{prediction} = \begin{cases} 1 & \text{if } \hat{y} > 0.5 \\ 0 & \text{otherwise} \end{cases}$$

## Cost Function

We use binary cross-entropy (also called log loss). It penalizes wrong predictions more heavily when the model is confident but incorrect:

$$J(W, b) = -\frac{1}{m} \sum_{i=1}^{m} \left[ y^{(i)} \log(\hat{y}^{(i)}) + (1 - y^{(i)}) \log(1 - \hat{y}^{(i)}) \right]$$

- $m$ is the number of training examples
- $y^{(i)}$ is the true label (0 or 1)
- $\hat{y}^{(i)}$ is the predicted probability

When $y=1$, only the $\log(\hat{y})$ term matters. When $y=0$, only the $\log(1 - \hat{y})$ term matters. The closer $\hat{y}$ is to $y$, the smaller the cost.

## Gradients

A nice property of combining the sigmoid with cross-entropy is that the gradient becomes simple:

$$\frac{\partial J}{\partial W} = \frac{1}{m} X^T(\hat{y} - y)$$

$$\frac{\partial J}{\partial b} = \frac{1}{m} \sum_{i=1}^{m} (\hat{y}^{(i)} - y^{(i)})$$

The gradient has the same form as linear regression. The difference is that $\hat{y}$ now includes the sigmoid.

## Numerical Stability

The sigmoid function can overflow for large inputs. A stable version is:

$$\sigma(z) = \begin{cases} \frac{1}{1 + e^{-z}} & \text{if } z \ge 0 \\ \frac{e^z}{1 + e^z} & \text{if } z < 0 \end{cases}$$

Also, to avoid $\log(0)$ in the cost, predictions are clipped to $[10^{-15}, 1 - 10^{-15}]$ before computing the log.

## Parameter Update

We use gradient descent to update the parameters:

$$W := W - \alpha \frac{\partial J}{\partial W}$$

$$b := b - \alpha \frac{\partial J}{\partial b}$$

- $\alpha$ is the learning rate

This step is repeated for a fixed number of iterations. The cost should go down over time.

## Algorithm Steps

1. Initialize $W$ with small random values and $b$ with zero
2. For each iteration:
   a. Compute linear score $z = XW + b$
   b. Compute probabilities $\hat{y} = \sigma(z)$
   c. Compute the cost $J(W, b)$
   d. Compute the gradients $\frac{\partial J}{\partial W}$ and $\frac{\partial J}{\partial b}$
   e. Update $W$ and $b$ using gradient descent
3. To predict a class: return 1 if $\hat{y} > 0.5$, else return 0
4. To predict a probability: return $\hat{y}$ directly

## Hyperparameters

- **lr** ($\alpha$): Learning rate for gradient descent. Default is 0.01
- **n_iter**: Number of gradient descent steps. Default is 3000

## Notes

- The cost function is convex, so gradient descent will always reach the global minimum with a small enough learning rate
- Feature scaling helps gradient descent converge faster
- Weight initialization matters: starting with large weights saturates the sigmoid and slows down learning
- scikit-learn's `LogisticRegression` uses L2 regularization by default. This implementation does not, so it may slightly overfit on clean datasets but can achieve a better fit
- Unlike linear regression, there is no closed-form solution. We must use an iterative optimizer like gradient descent
