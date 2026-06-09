# Technical Report: Titanic Survival Classification Engine
**Prepared by:** Saad Yasser Al-Haqan  
**Course:** Capstone Project Assignment  

## 1. Executive Summary & Objective
This project implements an end-to-end machine learning pipeline to resolve a binary classification problem using the historical Titanic passenger registry. The core objective is to analyze multiple demographic and socio-economic variables to predict individual survival probabilities. This dataset serves as an excellent benchmark for examining data imputation techniques and analyzing algorithmic behavior across different paradigms.

## 2. Statistical Analysis of the Dataset
The underlying dataset comprises 891 individual observations. Preliminary exploratory data analysis revealed a significant class imbalance: approximately 61% of the records represent casualties, whereas only 38% represent survivors. 

Because of this asymmetric class distribution, relying solely on standard Accuracy can yield overly optimistic and misleading results. Therefore, this evaluation prioritizes the F1-Score metric—the harmonic mean of precision and recall—alongside the standard classification accuracy to ensure a rigorous and balanced model assessment.

## 3. Traditional Machine Learning Benchmarks
Three diverse classification algorithms were trained and evaluated on the preprocessed feature space. The empirical performance metrics are detailed below:

* **Logistic Regression (LR):** Achieved an Accuracy of 80% with an F1-Score of 0.72. This linear classifier establishes a reliable baseline.
* **K-Nearest Neighbors (KNN):** Achieved an Accuracy of 81% with an F1-Score of 0.74 by evaluating spatial proximity within the normalized feature space.
* **Random Forest (RF):** Achieved an Accuracy of 83% with an F1-Score of 0.76. 

The ensemble-based Random Forest architecture demonstrated the highest predictive performance among the baseline classifiers.

## 4. Deep Learning Architecture (Artificial Neural Network)
To explore non-linear high-dimensional decision boundaries, a multi-layer Perceptron was constructed via the TensorFlow Keras Sequential API. The network topology consists of the following components:
1.  **Input/First Hidden Layer:** 32 nodes utilizing the Rectified Linear Unit (ReLU) activation function.
2.  **Regularization Layer:** A Dropout layer with a rate of 0.2 to mitigate the risk of overfitting by randomly disabling 20% of neurons during training.
3.  **Second Hidden Layer:** 16 nodes utilizing the ReLU activation function.
4.  **Output Layer:** A single node utilizing a Sigmoid activation function to map the network output into a valid probability distribution between 0 and 1.

The network was compiled using the Adam optimizer and Binary Crossentropy loss, and trained over 15 training epochs.

## 5. Comparative Evaluation & Analytical Synthesis
The Deep Learning model achieved an Accuracy of 80% and an F1-Score of 0.72 upon evaluation against the hold-out test set. While its performance closely mirrors the linear baseline established by Logistic Regression, it underperformed relative to the Random Forest ensemble. 

This behavior reinforces a fundamental principle in data science: for small-scale tabular datasets (such as this 891-row registry), complex deep neural networks often suffer from data scarcity and are easily outperformed by tree-based ensemble methods, which are computationally faster and highly efficient at capturing structural boundaries without extensive optimization.

## 6. Project Conclusions
* Imputation methodology directly governs downstream predictive validity; selecting robust statistical descriptors (like the median) prevents distribution distortion.
* Model complexity does not automatically translate to superior performance. Tree-based ensembles remain highly effective for structured tabular environments.