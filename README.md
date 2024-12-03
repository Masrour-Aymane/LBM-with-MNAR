# **Learning from Missing Data with the Binary Latent Block Model (LBM-MNAR)**

This repository provides the implementation of the **Binary Latent Block Model (LBM)** with integration for **Missing Not At Random (MNAR)** mechanisms. The repository includes the code and notebooks necessary to reproduce the results presented in the article *"Learning from Missing Data Using the Binary Latent Block Model"*.

## **Overview**
The Binary Latent Block Model (LBM) is a co-clustering approach designed for binary data matrices. This extension addresses the challenges posed by missing entries that are **informative**, i.e., not missing completely at random (MCAR). The repository:
- Implements the **Variational Expectation-Maximization (VEM)** algorithm for efficient parameter inference.
- Provides reproducible results for a real-world case study using the **French Parliamentary Voting Data**.

## **Features**
1. **MNAR Missingness Handling**:
   - Explicit modeling of missingness as a function of both observed and unobserved data.
   - A robust framework for extracting latent structures in the presence of informative missingness.

2. **Variational Inference**:
   - Efficient approximation of the posterior distribution using a factorized variational approach.
   - Support for entropy and ELBO computation with second-order Taylor approximations.
     
3. **Applications**:
   - Analysis of the French Parliament voting dataset to uncover voting patterns and interpret non-voter behavior.

## **Contents**
- **Code**: Python scripts implementing the LBM-MNAR framework using the PyTorch library.
- **Notebooks**: Jupyter notebooks for reproducing results on the French Parliamentary Voting dataset.
- **Figures**: Visualizations of co-clustering results, non-voters' behavior, and classification performance.
- **Report & slides** : Review of the work done in the article *"Learning from Missing Data Using the Binary Latent Block Model"*.

## **Usage**
1. Clone the repository:
   ```bash
   git clone https://github.com/<your-repo>/lbm-mnar.git
   cd lbm-mnar
