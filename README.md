# ML From Scratch

A collection of machine learning and deep learning algorithms implemented from scratch in Python and NumPy. The goal is to understand how these algorithms work internally, practice mathematical derivations, and prepare for machine learning interviews.

Every core algorithm is implemented with NumPy only. PyTorch is used only for testing and comparison.

## Why This Repository Exists

Reading about algorithms is useful, but implementing them reveals gaps in understanding. This repository focuses on:

- Writing algorithms from scratch using only NumPy
- Deriving gradients, losses, and update rules by hand
- Comparing custom implementations against PyTorch or reference results
- Writing tests for correctness and numerical stability
- Preparing concise interview notes for each algorithm

## Target Audience

This repository is for anyone who:

- Wants to understand ML and DL algorithms at the implementation level
- Is preparing for machine learning interviews
- Prefers learning by building rather than only reading

## Recommended Learning Approach

With roughly three months before an interview, use a hybrid strategy. Go deep on the algorithms that appear most often in interviews. Cover the rest with clean implementations and strong conceptual notes.

### Deep Focus Algorithms

These algorithms should be implemented fully from scratch, with hand-derived gradients, thorough tests, and detailed interview notes.

1. Linear Regression
2. Logistic Regression
3. Decision Tree
4. Random Forest
5. Gradient Boosting
6. K-Means
7. PCA
8. MLP with Backpropagation
9. CNN
10. Transformer

### Breadth Coverage Algorithms

These algorithms should be implemented cleanly and understood conceptually. Optimization fine-tuning and exhaustive edge cases are less critical here.

- K-Nearest Neighbors
- Naive Bayes
- Support Vector Machine
- Hierarchical Clustering
- Gaussian Mixture Model
- RNN and LSTM
- Word2Vec
- Attention Variants
- Transformer Language Model

## Repository Structure

```
ml-from-scratch/
├── README.md
├── pyproject.toml
├── uv.lock
├── .github/
│   └── workflows/
│       └── test.yml
├── src/
│   ├── supervised/
│   │   ├── linear_regression/
│   │   ├── logistic_regression/
│   │   ├── knn/
│   │   ├── naive_bayes/
│   │   ├── decision_tree/
│   │   ├── random_forest/
│   │   ├── svm/
│   │   └── gradient_boosting/
│   ├── unsupervised/
│   │   ├── kmeans/
│   │   ├── pca/
│   │   ├── gaussian_mixture/
│   │   └── hierarchical_clustering/
│   ├── deep_learning/
│   │   ├── mlp/
│   │   ├── cnn/
│   │   ├── rnn/
│   │   ├── lstm/
│   │   ├── word2vec/
│   │   ├── attention/
│   │   └── transformer/
│   └── nlp/
│       ├── tokenizer/
│       ├── ngram_lm/
│       ├── word_embeddings/
│       └── transformer_lm/
├── tests/
├── notebooks/
└── docs/
    └── interview_notes/
```

Each algorithm directory follows the same template described below.

## Per-Algorithm Template

Every algorithm directory contains the following files:

- `algorithm.py`: Pure NumPy implementation
- `demo.py`: A runnable example with a toy or real dataset
- `test_algorithm.py`: Pytest tests for correctness and edge cases
- `pytorch_check.py`: Optional comparison against PyTorch for neural network models
- `README.md`: Mathematical notes, complexity analysis, and common interview questions

This structure keeps the codebase consistent and easy to review before an interview.

## Prerequisites

- Python 3.10 or newer
- uv for dependency management
- Git

## Setup

Clone the repository:

```bash
git clone https://github.com/nesirli/ml-from-scratch.git
cd ml-from-scratch
```

Install dependencies using uv:

```bash
uv sync
```

Activate the virtual environment:

```bash
source .venv/bin/activate
```

Run the tests:

```bash
uv run pytest
```

## Running a Demo

Each algorithm has its own demo. For example, to run the Linear Regression demo:

```bash
uv run python src/supervised/linear_regression/demo.py
```

## Running Tests

Run all tests:

```bash
uv run pytest
```

Run tests for a specific algorithm:

```bash
uv run pytest tests/supervised/test_linear_regression.py
```

## Continuous Integration

GitHub Actions runs the full test suite on every pull request and push to the main branch. The workflow uses uv for fast dependency installation.

## Technology Choices

- **Python**: The standard language for machine learning development and interviews.
- **NumPy**: Used for all core algorithm implementations. This forces explicit handling of linear algebra, broadcasting, and numerical stability.
- **PyTorch**: Used only for testing and verifying custom implementations. No TensorFlow is used in this repository.
- **pytest**: Used for unit tests and correctness checks.
- **uv**: Used for fast dependency locking and environment management.
- **GitHub Actions**: Used for continuous integration.

## Learning Roadmap

The algorithms are ordered from basic to advanced. It is recommended to follow this order, but you can skip topics you already know well.

### Phase 1: Supervised Foundations

1. Linear Regression (ordinary least squares and gradient descent)
2. Logistic Regression (binary and multiclass)
3. K-Nearest Neighbors
4. Naive Bayes
5. Decision Tree
6. Random Forest
7. Support Vector Machine
8. Gradient Boosting

### Phase 2: Unsupervised Learning

9. K-Means Clustering
10. Principal Component Analysis
11. Gaussian Mixture Model
12. Hierarchical Clustering

### Phase 3: Deep Learning

13. Multi-Layer Perceptron (backpropagation from scratch)
14. Convolutional Neural Network
15. Recurrent Neural Network
16. Long Short-Term Memory Network

### Phase 4: NLP and Sequence Modeling

17. Tokenization
18. N-Gram Language Model
19. Word2Vec
20. Attention Mechanism
21. Transformer
22. Transformer Language Model

## Suggested Weekly Schedule

Assuming three months of preparation, a possible schedule is:

- **Weeks 1 to 2**: Linear Regression, Logistic Regression, KNN, Naive Bayes
- **Weeks 3 to 4**: Decision Tree, Random Forest, Gradient Boosting
- **Weeks 5 to 6**: K-Means, PCA, SVM
- **Weeks 7 to 8**: MLP and Backpropagation
- **Weeks 9 to 10**: CNN, RNN, LSTM
- **Weeks 11 to 12**: Word2Vec, Attention, Transformer
- **Final Week**: Review interview notes, rerun tests, and practice explaining algorithms out loud

This schedule assumes roughly 5 to 10 hours of focused work per week.

## How to Use This Repository for Interview Prep

1. Read the README for an algorithm.
2. Try to derive the loss function and gradients on paper.
3. Implement the algorithm in NumPy without looking at the solution.
4. Compare your implementation with the reference version in this repository.
5. Run the tests and fix any issues.
6. Review the interview questions and try to answer them out loud.
7. Explain the algorithm to yourself or someone else in simple terms.

## Interview Notes Format

Each algorithm README contains:

- A short description of what the algorithm does
- The mathematical formulation
- The training procedure
- Time and space complexity
- Common hyperparameters
- Strengths and weaknesses
- Common interview questions with concise answers

## Contributing

This is a personal learning repository, but suggestions and improvements are welcome. If you find a bug or want to add a reference implementation, please open an issue or pull request.

## License

MIT License. See LICENSE for details.
