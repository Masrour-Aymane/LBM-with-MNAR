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

## **Unsupervised Learning**

This project was completed as part of the master's course on **Unsupervised Learning**, done by *Aymane Masrour* and *Ndiaye Samara*. In this course project, the primary goals were:
- Replication of the key findings from the research paper *"Learning from Missing Data with the Binary Latent Block Model"*.
- Providing a comprehensive computational overview of the Latent Block Model (LBM) tailored for Missing Not At Random (MNAR) scenarios.

## **Contents**
- **Code**: Python scripts implementing the LBM-MNAR framework using the PyTorch library.
- **Notebook**: `LBM_MNAR_review.ipynb` for reproducing results on the French Parliamentary Voting dataset.
- **Figures**: Visualizations of co-clustering results, non-voters' behavior, and classification performance.
- **Report & Slides**: Review of the work done in the article *"Learning from Missing Data Using the Binary Latent Block Model"*.

## **How to Apply the LBM with MNAR Missing Data on the French Parliament 2018 Dataset**

**The use of at least one GPU is necessary to run the model on the French parliament dataset.**

The French parliament 2018 dataset is available in the folder *data_parliament*.

---

### Pytorch installation

The model is implemented with pytorch.
To install pytorch we refer the reader to the [Pytorch website](https://pytorch.org/get-started/locally/)

With conda:
```bash
conda install pytorch torchvision cudatoolkit=10.2 -c pytorch
```

With pip:
```bash
pip install torch torchvision
```
### Other requirements

With conda:
```bash
conda install numpy
conda install -c anaconda scipy
conda install -c conda-forge matplotlib
conda install -c conda-forge argparse

```

With pip:
```bash
pip install numpy scipy matplotlib argparse
```


## Usage 1

- run the notebook `LBM_MNAR_review.ipynb`

or

## Usage 2
- use the script *run_on_dataset_parliament.py*:
```bash
python run_on_dataset_parliament.py
```
The default number of row classes is 3 and column classes is 5.



To run with a GPU use the argument *device* and specify the cuda index of desired gpu (often 0):
```bash
python run_on_dataset_parliament.py --device=0
```

To run with higher number of classes, use the arguments *nb_row_classes* and *nb_col_classes* as:
```bash
python run_on_dataset_parliament.py --nb_row_classes=3 --nb_col_classes=5
```

With higher number of classes, the memory of your GPU may overflow. In that case, you can use a second GPU with the argument *device2* (index cuda needs to be specify):

```bash
python run_on_dataset_parliament.py --device=0 --device2=1 --nb_row_classes=3 --nb_col_classes=8
```

## License
[MIT]

The work is based on the Github from the article: 
https://github.com/gfrisch/LBM-MNAR
