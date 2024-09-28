# COVID-19 Infection Prediction using Physical Signs

This project applies machine learning algorithms to predict COVID-19 infection based on physical signs. We implemented three widely-used classification models—**Decision Tree**, **Random Forest**, and **Support Vector Machine (SVM)**—using a dataset of 10,000 instances sourced from Kaggle. These models were trained and tested using the **Scikit-learn** library to achieve an impressive prediction accuracy of 99%. The paper is [paper](https://www.spiedigitallibrary.org/conference-proceedings-of-spie/12287/122872C/COVID-19-infection-prediction-using-physical-signs/10.1117/12.2640976.short#_=_)

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Algorithms Used](#algorithms-used)
- [Results](#results)

## Introduction
The goal of this project is to predict whether a patient is COVID-19 positive or negative based on three physical signs from the dataset. This solution could potentially serve as a fast preliminary diagnostic tool, enhancing the detection accuracy in settings with limited testing resources.

By training and optimizing machine learning models, we aim to reduce false negatives and increase overall testing efficiency, contributing to more accurate early diagnosis.

## Dataset
The dataset contains 10,000 entries with three key features corresponding to physical signs:

These features are used to predict the target label: **COVID-19 Positive** or **Negative**.

The dataset can be downloaded from [Kaggle](https://www.kaggle.com/).

## Algorithms Used
We employed three machine learning algorithms for this classification task:

1. **Decision Tree**  
   A simple yet powerful model that uses a tree-like structure to make decisions based on input features.

2. **Random Forest**  
   An ensemble model consisting of multiple decision trees to improve accuracy and reduce overfitting.

3. **Support Vector Machine (SVM)**  
   A robust classifier that finds the optimal hyperplane to separate the data into different classes.

### Tools Used
- **Python**: Main programming language for the project
- **Scikit-learn**: Used to build and evaluate machine learning models
- **Numpy**: For data manipulation and mathematical operations
- **Pandas**: For handling datasets (optional)

### Results

We achieved a 99% accuracy in predicting whether a patient is COVID-19 positive or negative. The 
Random Forest model performed the best among the three, with the SVM model following closely behind. All models were evaluated using
 cross-validation and metrics such as accuracy, precision, recall, and F1-score.
  
