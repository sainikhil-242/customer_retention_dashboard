# customer_retention_dashboard

This project uses machine learning to predict whether a telecom customer is likely to churn based on various features. It includes model training (Logistic Regression, XGBoost, Random Forest), Streamlit UI for predictions, and a Power BI dashboard for interactive visualization.

🧠 PROBLEM STATEMENT


The objective of this project is to predict customer churn in a telecom dataset using machine learning techniques. A churned customer is one who has stopped using the service. Predicting churn can help the company implement proactive customer retention strategies.


🛠️ APPROACH


DATA PREPROCESSING


Loaded Dataset: WA_Fn-UseC_-Telco-Customer-Churn.csv.

Dropped customerID column.

Converted TotalCharges to numeric; imputed missing values using median.

Binary encoded columns: Partner, Dependents, PhoneService, PaperlessBilling.

Replaced 'No internet service' and 'No phone service' with 'No' in relevant columns.

Applied One-Hot Encoding for categorical features.

Scaled numerical features: tenure, MonthlyCharges, TotalCharges using StandardScaler.


EXPLORATORY DATA ANALYSIS (EDA)


Visualized churn distribution, correlation heatmap, and monthly charges vs churn using Seaborn and Matplotlib.

Handling Class Imbalance


Applied SMOTE (Synthetic Minority Over-sampling Technique) to balance classes in the target variable Churn.

Model Training & Tuning

   
Trained three models with hyperparameter tuning using RandomizedSearchCV:

Model	Notes


Logistic Regression	Baseline model
Random Forest	Tree-based ensemble with hyperparameter tuning
XGBoost	Gradient boosting model with tuning 

EVALUATION METRICS


Each model was evaluated using:

Accuracy – Overall correctness of the model

Precision – How many predicted churns were correct

Recall – How many actual churns were identified

F1-Score – Harmonic mean of precision and recall



📊RESULTS


![custo_metrics](https://github.com/user-attachments/assets/2cde8867-e484-4384-ab63-dd2f1a9faf16)



Random Forest gave the best F1-Score and was selected as the best-performing model.



🚀 DEPLOYMENT


Built an interactive frontend using Streamlit.

Allows users to:

Upload customer CSV files.

Select prediction model (Logistic Regression, XGBoost, or Random Forest).

View churn prediction results.

Download predictions and model metrics.

📊 VISUALISATION


Embedded a Power BI dashboard to visualize customer behavior.

Provided .pbix file for download in the app.

🧾 MODEL FILES & OUTPUTS


File	Description
logistic_model.pkl	Trained Logistic Regression model
xgb_model.pkl	Trained XGBoost model
rf_model.pkl	Trained Random Forest model
train_columns.pkl	Feature column list for inference
model_metrics.json	Stores metrics (Accuracy, F1, etc.) for UI
customer_retention_dashboard.pbix	Power BI dashboard source file






STEPS TO RUN THE CODE

Create folder structure like this and place the files 
📁 Folder Structure
![custo_folder_struct](https://github.com/user-attachments/assets/fc3c233d-c397-42eb-9181-0d7643b668a5)



* Open Pycharm and open the project folder in it.
  
* Create a vitual environment for it inside pycharm (Ensure the python interpreter version as 3.11 as because of model coding version must be compatible for streamlit app.

* 
INSTALL DEPENDENCIES USING BELOW CODE:


pip install -r requirements.txt
 
TRAIN THE MODELS



Run the training script notebook in Google Colab to preprocess data and save trained models.

python customer_.ipynb

This will save models (.pkl files) download it to your desktop  and place it in the( models/)folder.


🚀 RUN THE STREAMLIT APP

Open terminal in pycharm and run thr code using streamlit run app.py


* Upload a CSV file with customer data (same structure as training data).

* Select a model: Logistic Regression, XGBoost, or Random Forest.

* View churn predictions in the browser and download results as CSV.

  ![Screenshot 2025-06-20 163353](https://github.com/user-attachments/assets/1c1a03e9-590d-4f11-93d2-eaf2e92a1ee5)
![Screenshot 2025-06-20 163329](https://github.com/user-attachments/assets/9860f6cb-b64a-4d91-a46d-f7602daf3386)




✅ REQUIREMENTS


Python 3.8+,(3.11 preffered)

Streamlit

Scikit-learn

XGBoost

Imbalanced-learn

Pandas, NumPy, Seaborn, Matplotlib, Joblib

RandomSearchCV


📌 TO DO


Improve accuracy for Random Forest model



VISUALS AND FINDINGS


![customer_retention](https://github.com/user-attachments/assets/8fc40293-2124-450f-896a-fe01818deffb)


FINDINGS


Retention is Strong: A high retention rate of 73.42% shows positive customer loyalty.

Contract Length Matters: Customers with longer contracts (2 years) tend to churn less than those with month-to-month agreements.

Fiber Optic Service Leads: Majority of customers use Fiber Optic internet, which might be contributing to higher overall charges.

Online Backup and StreamingTV impact churn indirectly:

Users with Online Backup are a significant portion (43.9%).

A near-even split exists in StreamingTV usage.

Tenure Matters: Average customer tenure of 32.42 months suggests that long-term users are common and may be more loyal.



STORY TELLING



The dashboard tells a compelling story of how contract type, internet service, and digital add-ons like Online Backup and StreamingTV influence customer retention. Most customers are on fiber optic internet, many of whom use month-to-month contracts—a risk factor for churn. However, customers who commit to longer-term contracts are more likely to stay. The dashboard empowers decision-makers to target month-to-month customers for upselling into longer plans, leverage value-added services like Online Backup, and prioritize engagement with high-value customer segments.


RECOMMENDATIONS:


Promote Online Backup & StreamingTV:

* Bundling these services may increase stickiness.

Churn Prevention Programs:

* Target customers on month-to-month contracts with tailored offers.
  
* Offer discounts or perks for 1- or 2-year plans.
