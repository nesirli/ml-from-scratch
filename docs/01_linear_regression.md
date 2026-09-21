# Linear Regression

Linear regression is a supervised learning algorithm. It predicts a continuous target variable from one or more input features.

## Model

The model is a linear function. For input features $X$, the prediction is:

$$\hat{y} = XW + b$$

- $X$ is the input matrix with shape $(m, n)$
- $W$ is the weight vector with shape $(n,)$
- $b$ is the bias (a single number)
- $\hat{y}$ is the predicted output with shape $(m,)$

## Cost Function

The cost measures how far the predictions are from the true values. We use mean squared error (MSE):

$$J(W, b) = \frac{1}{2m} \sum_{i=1}^{m} (\hat{y}^{(i)} - y^{(i)})^2$$

- $m$ is the number of training examples
- $y$ is the true target value

The factor $\frac{1}{2}$ is for convenience. It cancels out the 2 that appears when we take the derivative.

## Gradients

To minimize the cost, we compute the partial derivatives with respect to $W$ and $b$:

$$\frac{\partial J}{\partial W} = \frac{1}{m} X^T(\hat{y} - y)$$

$$\frac{\partial J}{\partial b} = \frac{1}{m} \sum_{i=1}^{m} (\hat{y}^{(i)} - y^{(i)})$$

## Parameter Update

We use gradient descent to update the parameters:

$$W := W - \alpha \frac{\partial J}{\partial W}$$

$$b := b - \alpha \frac{\partial J}{\partial b}$$

- $\alpha$ is the learning rate

This step is repeated for a fixed number of iterations. The cost should go down over time.

## Algorithm Steps

1. Initialize $W$ with small random values and $b$ with zero
2. For each iteration:
   a. Compute predictions $\hat{y} = XW + b$
   b. Compute the cost $J(W, b)$
   c. Compute the gradients $\frac{\partial J}{\partial W}$ and $\frac{\partial J}{\partial b}$
   d. Update $W$ and $b$ using gradient descent
3. After training, use the final $W$ and $b$ to make predictions

## Hyperparameters

- **learning_rate** ($\alpha$): Controls the step size in gradient descent. Default is 0.001
- **n_iterations**: Number of gradient descent steps. Default is 10000

## Notes

- The cost function is convex, so gradient descent will always reach the global minimum with a small enough learning rate
- Feature scaling (like standardization) helps gradient descent converge faster
- For comparison, scikit-learn's `LinearRegression` uses the normal equation (a closed-form solution), not gradient descent
