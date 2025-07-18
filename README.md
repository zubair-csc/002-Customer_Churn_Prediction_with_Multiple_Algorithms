Customer Churn Prediction with Multiple Algorithms
Overview
This project implements a comprehensive customer churn prediction model using multiple machine learning algorithms: Random Forest, XGBoost, and SVM. The analysis includes data preprocessing, exploratory data analysis (EDA), model training, evaluation with ROC analysis, feature importance analysis, and business metrics to quantify the financial impact of churn predictions. The code is designed to run in Google Colab and is stored in the notebook 002-Customer-Churn-Prediction-with-Multiple-Algorithms.ipynb.

The goal of this project is to predict customer churn for a telecommunications company using a synthetic dataset. The notebook performs the following tasks:

Data Generation: Creates a synthetic dataset with 10,000 customer records, including features like age, tenure, monthly charges, service usage, and churn status.
Exploratory Data Analysis (EDA): Visualizes churn distribution, feature correlations, and relationships between key features and churn.
Data Preprocessing: Applies feature scaling and train-test splitting with stratification to maintain consistent churn rates.
Model Training: Trains three models (Random Forest, XGBoost, SVM) and evaluates them using accuracy, precision, recall, F1-score, and ROC-AUC.
ROC Analysis: Generates ROC curves and calculates AUC for model comparison.
Feature Importance: Analyzes feature importance for tree-based models (Random Forest and XGBoost).
Business Metrics: Quantifies revenue impact, retention costs, and ROI for each model's predictions.
Recommendations: Provides actionable business insights based on model results and key churn indicators.

Dataset
The dataset is synthetically generated with the following features:

Demographics: Age
Account Information: Tenure, monthly charges, total charges
Services: Internet service, phone service, multiple lines, online security, online backup, device protection, tech support, streaming TV, streaming movies
Contract and Payment: Month-to-month contract, one-year contract, two-year contract, paperless billing, auto-pay
Customer Service: Number of support calls
Target Variable: Churn (binary: 0 = No Churn, 1 = Churn)

The dataset contains 10,000 records with a realistic churn rate (approximately 20-30%). For real-world applications, replace the synthetic data generation with your own dataset.
Requirements
The notebook is designed to run in Google Colab, which includes all necessary libraries. The required Python packages are:

pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost

These are pre-installed in Google Colab. If running locally, install them using:
pip install pandas numpy matplotlib seaborn scikit-learn xgboost

How to Run

Open in Google Colab:
Navigate to Google Colab.
Upload the notebook 002-Customer-Churn-Prediction-with-Multiple-Algorithms.ipynb or open it directly from GitHub via Colab's GitHub integration.


Run the Notebook:
Execute the cells sequentially. The notebook includes comments and section headers for clarity.
The synthetic dataset is generated automatically. To use your own dataset, modify the data loading section (replace the generate_customer_data function with your data import logic, e.g., loading from a CSV file).


View Results:
The notebook generates visualizations (churn distribution, ROC curves, confusion matrices, feature importance) and prints model performance metrics and business insights.


Save and Export:
Save the notebook in Colab: File > Download > Download .ipynb.
Optionally, save outputs (e.g., plots) using Colab's file download options.



Repository Structure
002-Customer_Churn_Prediction_with_Multiple_Algorithms/
├── 002-Customer-Churn-Prediction-with-Multiple-Algorithms.ipynb
├── README.md

Results

Model Performance: The notebook compares Random Forest, XGBoost, and SVM based on accuracy, precision, recall, F1-score, and ROC-AUC. Cross-validation ensures robust evaluation.
ROC Analysis: ROC curves and AUC scores are plotted for all models, with Random Forest typically achieving the highest AUC.
Feature Importance: Key churn indicators include tenure, monthly charges, support calls, and contract type.
Business Impact:
Calculates prevented churn value, retention campaign costs, and ROI.
Provides net revenue impact for each model, helping prioritize retention strategies.


Recommendations:
Focus retention on new customers (<1 year tenure).
Target high monthly charge customers with proactive offers.
Implement automated alerts for customers with frequent support calls.
Promote longer-term contracts to reduce churn.
