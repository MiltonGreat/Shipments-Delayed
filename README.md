# Bank Customer Segmentation Using Clustering Techniques

## Project Overview

This project aims to predict the likelihood of shipment delays using a machine learning model based on historical shipment data. By using features such as shipment type, origin, destination, and more, the model determines the risk of delay for each shipment. The solution includes data preprocessing, model training, hyperparameter tuning, and evaluation using various machine learning algorithms like Random Forest, XGBoost, and LightGBM.

## Dataset

The dataset consists of historical shipment data with the following important features:

- Shipment Details: Shipment type, origin, destination, pickup/drop-off latitude/longitude.
- Target Variable: is_delayed (1 = delayed, 0 = on-time).

## Project Steps

1. Data Preprocessing: Includes handling missing values, feature scaling, and encoding categorical variables.
2. Multiple Models Evaluated: Logistic Regression, Decision Tree, Random Forest, XGBoost, LightGBM, and SVM.
3. Hyperparameter Tuning: Utilizes RandomizedSearchCV and GridSearchCV for finding the best hyperparameters.
4. SMOTE: Balances the dataset by oversampling minority classes.
5. Model Performance Evaluation: Evaluates models using metrics like accuracy, ROC-AUC, precision, recall, and F1-score.

## Key Results

Model Performance

- From the output, it is evident that the models, particularly RandomForest and XGBoost, are performing exceptionally well with extremely high accuracy and ROC-AUC scores.

Classification Report

- The classification report shows that the model is performing near perfectly on both the majority class (on-time shipments) and the minority class (delayed shipments). This is an important achievement considering class imbalance is often an issue in shipment delay prediction.
Risk of Overfitting

- Given the training accuracy of 1.0, there is a potential risk of overfitting, especially since the test accuracy is also extremely high. However, based on the high test performance, it seems the model is generalizing well. Cross-validation was used to ensure the robustness of the model, which helps mitigate the risk of overfitting.
Use of SMOTE

- SMOTE (Synthetic Minority Over-sampling Technique) was applied to balance the dataset. This is a crucial step as it helps improve the performance of the model on the minority class (is_delayed = 1).

### Potential Improvements

- Further Validation: Although the model performs exceptionally well, it's important to validate it further on a truly unseen dataset (perhaps from a different time period or location).

- Model Interpretability: Use tools like SHAP or LIME to explain the model’s predictions and understand which features contribute the most to predicting delays.

- Regularization: Introduce stronger regularization (e.g., increase min_samples_split, min_samples_leaf, or alpha) to prevent potential overfitting.

- Ensemble Techniques: Consider combining the outputs of multiple models (RandomForest, XGBoost, and LightGBM) using techniques like stacking to improve performance further.

### Source

https://www.kaggle.com/datasets/omnamahshivai/dataset-delayed-shipments-outd-avt2-t3-sep052019
