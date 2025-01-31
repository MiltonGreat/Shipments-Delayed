# Shipments Delayed Prediction

### Project Overview

This project aims to predict the likelihood of shipment delays using machine learning models based on historical shipment data. By leveraging features such as shipment type, origin, destination, and geospatial data, the model determines the risk of delay for each shipment. The solution includes data preprocessing, model training, hyperparameter tuning, and evaluation using multiple machine learning algorithms.

### Dataset

The dataset consists of historical shipment data with the following important features:

- Shipment Details: Shipment type, origin, destination, pickup/drop-off latitude/longitude.
- Target Variable: is_delayed (1 = delayed, 0 = on-time).

### Project Steps

1. Data Preprocessing: Includes handling missing values, feature scaling, and encoding categorical variables.
2. Multiple Models Evaluated: Logistic Regression, Decision Tree, Random Forest, XGBoost, LightGBM, and SVM.
3. Hyperparameter Tuning: Use GridSearchCV for finding the best hyperparameters.
4. SMOTE: Use SMOTE (Synthetic Minority Over-sampling Technique) to balance the dataset.
5. Model Performance Evaluation: Evaluates models using metrics like accuracy, ROC-AUC, precision, recall, and F1-score.

### Key Results

- Best Model: RandomForestClassifier
- Accuracy: 99.84%
- AUC-ROC: 99.98%

### Key Findings

- The RandomForestClassifier achieved exceptionally high accuracy and ROC-AUC scores, performing well on both majority and minority classes.
- Cross-validation was applied to ensure robustness and reduce overfitting risk.
- SMOTE effectively improved model performance on the minority class (delayed shipments).

### Potential Improvements

- Further Validation: Although the model performs exceptionally well, it's important to validate it further on a truly unseen dataset (perhaps from a different time period or location).

- Model Interpretability: Use tools like SHAP or LIME to explain the model’s predictions and understand which features contribute the most to predicting delays.

- Regularization: Introduce stronger regularization (e.g., increase min_samples_split, min_samples_leaf, or alpha) to prevent potential overfitting.

- Ensemble Techniques: Consider combining the outputs of multiple models (RandomForest, XGBoost, and LightGBM) using techniques like stacking to improve performance further.

### Source

Dataset: [Delayed Shipment Dataset on Kaggle](https://www.kaggle.com/datasets/omnamahshivai/dataset-delayed-shipments-outd-avt2-t3-sep052019)
