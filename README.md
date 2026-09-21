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

## Repository Structure

```
ml-from-scratch/
├── README.md
├── pyproject.toml
├── uv.lock
├── algorithms/
│   ├── 01_linear_regression.py
│   ├── 02_logistic_regression.py
│   ├── 03_knn.py
│   ├── 04_naive_bayes.py
│   ├── 05_decision_tree.py
│   ├── 06_random_forest.py
│   ├── 07_svm.py
│   ├── 08_gradient_boosting.py
│   ├── 09_kmeans.py
│   ├── 10_pca.py
│   ├── 11_gaussian_mixture.py
│   ├── 12_hierarchical_clustering.py
│   ├── 13_mlp.py
│   ├── 14_cnn.py
│   ├── 15_rnn.py
│   ├── 16_lstm.py
│   ├── 17_tokenization.py
│   ├── 18_ngram_lm.py
│   ├── 19_word2vec.py
│   ├── 20_attention.py
│   ├── 21_transformer.py
│   └── 22_transformer_lm.py
└── docs/
    ├── implementations.md
    ├── metrics.md
    ├── model-classification.md
    └── supervised/
        ├── knn.md
        ├── linear_regression.md
        └── logistic_regression.md
```

Every algorithm lives in a single numbered file. Each file contains the pure NumPy implementation at the top and a runnable demo guarded by `if __name__ == "__main__":`.

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

## Running a Demo

Each algorithm file is runnable. For example, to run the Linear Regression demo:

```bash
uv run python algorithms/01_linear_regression.py
```

## Technology Choices

- **Python**: The standard language for machine learning development and interviews.
- **NumPy**: Used for all core algorithm implementations. This forces explicit handling of linear algebra, broadcasting, and numerical stability.
- **PyTorch**: Used only for testing and verifying custom implementations. No TensorFlow is used in this repository.
- **pytest**: Used for unit tests and correctness checks.
- **uv**: Used for fast dependency locking and environment management.
- **GitHub Actions**: Used for continuous integration.

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
