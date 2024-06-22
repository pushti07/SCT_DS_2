# SCT_DS_2

## Task Overview

This is the second task of my internship at Skillcraft Technology. The task involves building various machine learning models to predict the survival of passengers on the Titanic based on the Titanic dataset. The analysis includes data preprocessing, exploratory data analysis (EDA), and model building using different algorithms.

## Dataset

The dataset used for this project is the Titanic dataset, which is divided into two parts:

- `train.csv`: Contains the training data with survival labels.
- `test.csv`: Contains the test data without survival labels.

## Data Analysis and Preprocessing

1. **Data Loading and Initial Inspection**:
   - Loaded the train and test datasets using `pandas`.
   - Displayed the first few rows of both datasets to understand their structure.
   - Checked the shape of the datasets to understand the number of samples and features.
   - Inspected the data types and missing values using `.info()` and `.isnull().sum()` methods.
   - Identified and removed duplicate entries.
   - Provided summary statistics using `.describe()`.

2. **Exploratory Data Analysis (EDA)**:
   - Analyzed the gender distribution of passengers.
   - Plotted bar charts to visualize the number of male and female passengers.
   - Calculated and visualized the survival rate by gender using bar charts.
   - Analyzed the survival rate by passenger class (`Pclass`) using bar charts.
   - Plotted histograms to analyze the age distribution of survivors and non-survivors.
   - Analyzed the impact of siblings/spouses aboard (`SibSp`) on survival rate.
   - Analyzed the impact of port of embarkation (`Embarked`) on survival rate using a pie chart.
   
3. **Data Cleaning and Feature Engineering**:
   - Dropped irrelevant features such as `Ticket`, `Cabin`, and `Name`.
   - Filled missing values in the `Age` column with the median age.
   - Filled missing values in the `Embarked` column using forward fill method.
   - Converted categorical variables (`Sex` and `Embarked`) into numerical values using label encoding.

## Machine Learning Models

1. **Data Splitting**:
   - Split the data into training and testing sets using `train_test_split` from `sklearn.model_selection`.

2. **Model Building**:
   - Built and trained multiple machine learning models including:
     - Logistic Regression
     - Support Vector Machines (SVM)
     - K-Nearest Neighbors (KNN)
     - Naive Bayes
     - Decision Tree

3. **Model Evaluation**:
   - Evaluated the models using accuracy score and confusion matrix.
   - Compared the performance of different models based on their accuracy scores.

## Results

The performance of the models is summarized in the following table:

| Model                   | Accuracy Score |
|-------------------------|----------------|
| Logistic Regression     | 0.75           |
| Support Vector Machines | 0.66           |
| Naive Bayes             | 0.76           |
| K-Nearest Neighbors     | 0.66           |
| Decision Tree           | 0.74           |

Based on the accuracy scores, the Naive Bayes model performed the best with an accuracy of 0.76.

## Conclusion

In this project, I successfully built and evaluated multiple machine learning models to predict the survival of passengers on the Titanic. The Naive Bayes model achieved the highest accuracy. This analysis provided valuable insights into the factors affecting survival and demonstrated the application of various machine learning algorithms.


This project showcases my ability to perform data analysis, preprocess data, and build machine learning models, fulfilling the requirements of the second task for my internship at Skillcraft Technology.
