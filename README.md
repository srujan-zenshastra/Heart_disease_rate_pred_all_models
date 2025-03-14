# Heart Disease Rate Prediction

## Overview
This project focuses on predicting heart disease rates using multiple machine learning models. The dataset used for this project contains various health-related features that aid in determining the likelihood of heart disease.

## Dataset
The dataset used in this project can be accessed at the following link:
[Kaggle Heart Disease Prediction Dataset](https://www.kaggle.com/datasets/rishidamarla/heart-disease-prediction)

## Installation and Dependencies
To run this project, install the required dependencies using the following command:
```bash
pip install -r requirements.txt
```
If a requirements file is not provided, ensure the following dependencies are installed:
- Python (>=3.7)
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn
- LightGBM
- Jupyter Notebook

## Usage
1. Clone the repository or download the project files.
2. Open the Jupyter Notebook (`heart_dis_pred-all_models.ipynb`).
3. Run the cells sequentially to:
   - Load and preprocess the dataset.
   - Train and evaluate various machine learning models.
   - Compare model performance using appropriate metrics.

## Models Implemented
The following models have been implemented:
- **Logistic Regression**
- **Support Vector Machine (SVM)**
- **Decision Tree Classifier**
- **Gradient Boosting**
- **LightGBM (LGBM)**
- **MultiLayer Perceptron (MLP)**
- **Random Forest Classifier**
- **Voting Classifier (Hard & Soft Voting)**
- **Stacking Classifier (Layered Learning)**

## Model Evaluation
The models' performance is evaluated using:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- ROC Curve (if applicable)

## Results
Each model's results are compared to determine the most effective approach for heart disease prediction. Ensemble methods, such as voting and stacking classifiers, are tested to see if they outperform individual models.
