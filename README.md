# 🚱 Water-Potability-Classification 🚰

## Table of Content

## Project Overview
The Water Potability Classification project focuses on building a machine learning model to determine whether water is safe for consumption based on key physicochemical properties. The primary goal of this project is developing a reliable and efficient classification system that can assist in identifying potable water sources. By providing an automated solution, this project benefits consumers by enabling quicker and more accurate assessments of water quality, empowering communities, water management agencies, and environmental researchers to make informed decisions about water safety.

## Dataset Overview
The *Water Quality* is a publicly accessible dataset provided by Aditya Kadiwal on [Kaggle](https://www.kaggle.com/datasets/adityakadiwal/water-potability), designed to classify water potability based on various chemical and physical parameters. 

| **Feature**         | **Data Type** | **Description**                                                                 |
|----------------------|---------------|---------------------------------------------------------------------------------|
| `ph`                | Float         | pH value of water (range: 0 to 14).                                            |
| `Hardness`          | Float         | Capacity of water to precipitate soap, measured in mg/L.                       |
| `Solids`            | Float         | Total dissolved solids in water, measured in ppm.                              |
| `Chloramines`       | Float         | Amount of Chloramines in water, measured in ppm.                               |
| `Sulfate`           | Float         | Amount of Sulfates dissolved in water, measured in mg/L.                       |
| `Conductivity`      | Float         | Electrical conductivity of water, measured in μS/cm.                           |
| `Organic_carbon`    | Float         | Amount of organic carbon in water, measured in ppm.                            |
| `Trihalomethanes`   | Float         | Amount of Trihalomethanes in water, measured in μg/L.                          |
| `Turbidity`         | Float         | Measure of the light-emitting property of water, measured in NTU.              |
| `Potability`        | Integer       | Indicates if water is safe for human consumption: 1 = Potable, 0 = Not potable.| 

## Depedencies
Below are the Python libraries used in this project, along with their functionalities:  

- **pandas**: For data manipulation and analysis.  
- **matplotlib.pyplot**: For creating static visualizations.  
- **seaborn**: For creating informative and attractive statistical graphics.  
- **numpy**: For numerical operations and handling arrays.  
- **scikit-learn**:  
  - `train_test_split`: To split data into training and testing sets.  
  - `RandomForestClassifier`, `DecisionTreeClassifier`, `GaussianNB`, `LogisticRegression`: For implementing machine learning models.  
  - `accuracy_score`, `classification_report`, `confusion_matrix`, `roc_curve`, `auc`, `precision_recall_curve`: For model evaluation and performance metrics.  
  - `SelectKBest`, `f_classif`, `chi2`: For feature selection.  
  - `MinMaxScaler`: For feature scaling.  
  - `GridSearchCV`, `RandomizedSearchCV`: For hyperparameter tuning.  
  - `learning_curve`: For plotting learning curves.  
  - `BaseEstimator`, `ClassifierMixin`: For creating custom classifiers.  
- **imblearn**:  
  - `SMOTE`: For handling imbalanced datasets via oversampling.  
- **xgboost**:  
  - `XGBClassifier`: For implementing gradient boosting algorithms.  
- **mlxtend**:  
  - `SequentialFeatureSelector (SFS)`: For sequential feature selection.  
- **sklearn.feature_selection**:  
  - `RFE`: For recursive feature elimination.  
- **lime**:  
  - `lime_tabular`: For generating interpretable explanations of machine learning model predictions using the LIME (Local Interpretable Model-Agnostic Explanations) framework.  

These libraries collectively enable data preprocessing, model building, evaluation, optimization, and interpretability for the water potability classification project.

## **Methodology**  

The methodology for the *Water Potability Classification* project is divided into the following steps:  

1. **Library Import**  
   Necessary Python libraries are imported to facilitate data manipulation, visualization, preprocessing, modeling, evaluation, and interpretability.

2. **Exploratory Data Analysis (EDA)**  
   Initial analysis is conducted to understand the data distribution, identify missing values, and explore relationships between features and the target variable (`Potability`). Visualizations like histograms, boxplots, and correlation heatmaps are used for insights.  

3. **Preprocessing and Transformation**  
   - **Data Cleaning**: Handle missing values, outliers, and inconsistencies in the dataset.  
   - **Data Transformation**: Normalize numerical features to ensure uniform scaling.  
   - **Feature Selection**: Apply statistical methods such as Chi-square (Chi2), Sequential Feature Selector (SFS), and Recursive Feature Elimination (RFE) to identify important features.  
   - **Resampling**: Use SMOTE (Synthetic Minority Oversampling Technique) to address class imbalance.  
   - **Feature Selection on Resampled Data**: Reapply feature selection techniques after balancing the dataset.  

4. **Training**  
   Train models using five different machine learning algorithms, including Random Forest, Logistic Regression, XGBoost, Decision Tree, and Naive Bayes. For each algorithm, experiments are conducted using:  
   - Original data.  
   - Resampled data.  
   - Different feature selection techniques to compare performance.  

5. **Evaluation of the Best Model**  
   Evaluate model performance using metrics such as accuracy, precision, recall, F1-score, ROC-AUC, and confusion matrix. The best model is selected based on its performance across these metrics.  

6. **Model Interpretation with LIME**  
   Use the LIME (Local Interpretable Model-Agnostic Explanations) framework to interpret and explain the predictions of the best-performing model. LIME provides feature importance for individual predictions, enhancing transparency and trust in the model.  

7. **Error Analysis**  
   Analyze misclassified samples to understand patterns in prediction errors, identify potential limitations, and recommend improvements.  

8. **User Scenarios**  
   Provide use cases and scenarios where the model can be deployed, such as:  
   - Monitoring water quality in real-time systems.  
   - Assisting water management agencies in decision-making.  
   - Providing rapid water safety assessments for communities and researchers.  

This structured approach ensures the project achieves its goals of creating an accurate, interpretable, and user-oriented water potability classification system.
