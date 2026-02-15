# E-Commerce Sales Forecasting Engine

A Machine Learning based predictive tool designed to forecast future sales trends using historical transactional data.

## 📊 Overview
This project implements a **Multiple Linear Regression** model to predict quarterly sales performance. It analyzes the relationship between marketing expenditure, website traffic, and discount percentages to provide data-driven business insights.

## 🛠️ Tech Stack
* **Language:** Python 3.x
* **Data Analysis:** Pandas, NumPy
* **Machine Learning:** Scikit-Learn (Linear Regression, Train-Test Split)
* **Visualization (Optional):** Matplotlib / Seaborn
* **Tools:** Jupyter Notebook / VS Code

## 📈 Key Features
* **Multi-Variable Analysis:** Considers multiple factors (Marketing spend, Visitors, etc.) simultaneously for more accurate forecasting.
* **Data Preprocessing:** Handles missing values and categorical data encoding using NumPy and Pandas.
* **Model Evaluation:** Performance is measured using **Mean Absolute Error (MAE)** and **R-squared (R²)** scores.
* **Predictive Capability:** Includes a forecasting function to predict sales for future months based on new budget inputs.

## 📋 How It Works
1. **Data Ingestion:** Loads historical sales records into a Pandas DataFrame.
2. **Feature Engineering:** Selects the most relevant features (X) and the target variable (Sales - y).
3. **Training:** Splits data into 80% training and 20% testing sets to ensure model reliability.
4. **Prediction:** The model learns the coefficients and generates a forecast for the upcoming quarter.

## 💻 How to Run
1. Ensure you have Python and the required libraries installed:
   ```bash
   pip install pandas scikit-learn numpy
