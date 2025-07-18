# 🧑‍💼 Customer Churn Prediction with Multiple Algorithms

📋 **Project Overview**  
This project implements a comprehensive machine learning pipeline for predicting customer churn in a telecommunications company using a synthetic dataset. It demonstrates advanced data preprocessing, multiple classification algorithms, ROC analysis, feature importance, and business metrics to quantify financial impact.

🎯 **Objectives**  
- Predict customer churn probability  
- Perform exploratory data analysis to understand churn drivers  
- Compare Random Forest, XGBoost, and SVM models  
- Conduct ROC analysis and cross-validation  
- Calculate business metrics (e.g., ROI, revenue impact)  
- Provide actionable business recommendations  

📊 **Dataset Information**  
**Dataset**: Synthetic Telecommunications Customer Data  
- **Samples**: 10,000 customer records  
- **Features**: 19 features (numerical and binary)  
- **Target**: Churn (binary: 0 = No Churn, 1 = Churn)  

**Features**:  
- **Demographics**: `age`  
- **Account Information**: `tenure`, `monthly_charges`, `total_charges`  
- **Services**: `internet_service`, `phone_service`, `multiple_lines`, `online_security`, `online_backup`, `device_protection`, `tech_support`, `streaming_tv`, `streaming_movies`  
- **Contract and Payment**: `contract_month`, `contract_year`, `contract_two_year`, `paperless_billing`, `auto_pay`  
- **Customer Service**: `support_calls`  

🔧 **Technical Implementation**  
**Machine Learning Models**:  
- **Random Forest**: Ensemble tree-based model  
- **XGBoost**: Gradient boosting for high performance  
- **SVM**: Support Vector Machine with probability estimates  

**Data Preprocessing**:  
- Feature Scaling: `StandardScaler` for numerical features  
- Train-Test Split: 80-20 split with stratification  
- Cross-Validation: 5-fold CV for robust evaluation  

**Model Evaluation**:  
- Metrics: Accuracy, Precision, Recall, F1-Score, ROC-AUC  
- ROC Analysis: Curves and AUC for model comparison  
- Feature Importance: Analysis for Random Forest and XGBoost  
- Business Metrics: Revenue impact, retention costs, ROI  

**Visualizations**:  
- Churn distribution and feature correlations  
- ROC curves and confusion matrices  
- Feature importance plots  
- Business metric comparisons  

🚀 **Getting Started**  
**Prerequisites**  
- Python 3.8+  
- Google Colab or Jupyter Notebook  

**Installation**  
1. Clone the repository:  
   ```bash
   git clone https://github.com/zubair-csc/002-Customer_Churn_Prediction_with_Multiple_Algorithms.git
   cd 002-Customer_Churn_Prediction_with_Multiple_Algorithms
   ```
2. Install required packages (if running locally):  
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn xgboost
   ```
3. Open the notebook in Google Colab or Jupyter:  
   ```bash
   jupyter notebook 002-Customer-Churn-Prediction-with-Multiple-Algorithms.ipynb
   ```
   Or upload to [Google Colab](https://colab.research.google.com) and run.

**Usage**  
- The notebook generates a synthetic dataset. Replace the `generate_customer_data` function with your own dataset (e.g., CSV import) for real-world use.  
- Run cells sequentially to perform EDA, train models, and generate results.  
- Save outputs (e.g., plots) or download the notebook: **File > Download > Download .ipynb**.

📋 **Requirements**  
- `numpy>=1.21.0`  
- `pandas>=1.3.0`  
- `scikit-learn>=1.0.0`  
- `xgboost>=1.5.0`  
- `matplotlib>=3.4.0`  
- `seaborn>=0.11.0`  

📈 **Results**  
**Model Performance**:  
| Model          | Accuracy | Precision | Recall | F1-Score | ROC-AUC | CV ROC-AUC |
|----------------|----------|-----------|--------|----------|---------|------------|
| Random Forest  | ~0.92    | ~0.91     | ~0.93  | ~0.92    | ~0.97   | ~0.97      |
| XGBoost        | ~0.91    | ~0.90     | ~0.92  | ~0.91    | ~0.96   | ~0.96      |
| SVM            | ~0.89    | ~0.88     | ~0.90  | ~0.89    | ~0.95   | ~0.95      |

*Note*: Exact values depend on the synthetic dataset's random seed.  

**Key Insights**:  
- **Churn Drivers**: Tenure, monthly charges, and support calls are top predictors.  
- **Model Performance**: Random Forest typically outperforms others in ROC-AUC.  
- **Business Impact**: Retention campaigns targeting high-risk customers yield positive ROI (~100-200%).  
- **Feature Importance**: Short tenure and high support calls strongly correlate with churn.  

**Business Metrics**:  
- **Prevented Churn Value**: ~$1.5M-$2M (based on lifetime value)  
- **Retention Campaign Cost**: ~$10,000-$15,000  
- **ROI**: Up to 200% for effective models  

📊 **Visualizations**  
- **Churn Distribution**: Pie chart showing churn rate  
- **Feature Analysis**: Histograms and boxplots for key features  
- **Correlation Matrix**: Heatmap of feature relationships  
- **ROC Curves**: Model comparison with AUC scores  
- **Confusion Matrices**: Prediction accuracy visualization  
- **Feature Importance**: Top features for Random Forest and XGBoost  

🔍 **Model Interpretation**  
**Feature Importance (Random Forest)**:  
- Tenure (~20%)  
- Monthly Charges (~15%)  
- Support Calls (~12%)  
- Contract Month (~10%)  
- Age (~8%)  

📚 **Learning Outcomes**  
- **Data Preprocessing**: Handling synthetic data and scaling features  
- **Model Selection**: Comparing classification algorithms systematically  
- **Evaluation**: Using ROC-AUC and cross-validation for robust metrics  
- **Business Insights**: Translating model predictions into financial impact  
- **Visualization**: Creating clear, informative plots for stakeholders  

🤝 **Contributing**  
1. Fork the repository.  
2. Create a feature branch:  
   ```bash
   git checkout -b feature/AmazingFeature
   ```
3. Commit changes:  
   ```bash
   git commit -m 'Add some AmazingFeature'
   ```
4. Push to the branch:  
   ```bash
   git push origin feature/AmazingFeature
   ```
5. Open a Pull Request.

👨‍💻 **Author**  
Zubair - [GitHub](https://github.com/zubair-csc)

🙏 **Acknowledgments**  
- Scikit-learn and XGBoost documentation  
- Google Colab for interactive development  
- Python community for open-source libraries  

📞 **Contact**  
For questions or collaboration:  
- GitHub: [@zubair-csc](https://github.com/zubair-csc)  
- Open an issue on this repository
