# 📊 SalesForecastPro: Sales Forecasting with SARIMA

**SalesForecastPro** is a Streamlit web app that leverages machine learning to forecast future sales using SARIMA models. By analyzing historical sales data and incorporating external economic indicators, this tool helps businesses make informed decisions around **inventory**, **marketing**, and **pricing strategies**.

---

## 🧠 Problem / Opportunity

Accurate sales forecasting is critical for businesses to manage operations efficiently. Many organizations still rely on static rule-based systems or simple averages, which fail to capture complex trends and seasonality.

**Challenges with current forecasting methods:**
- Low adaptability to sudden market changes
- Limited use of external influencing factors
- Lack of explainability and actionable insights

---

## 💡 Why This Matters

An advanced forecasting model provides:
- Proactive **inventory planning** and reduced stockouts
- Better **pricing strategies** based on predicted demand
- Improved **marketing campaign timing**
- Enhanced **customer satisfaction** with product availability

---

## 👥 Stakeholders & Beneficiaries

- **Product Managers** – for roadmap and pricing decisions  
- **Sales & Marketing Teams** – to align promotions with demand  
- **Inventory Managers** – for stock level optimization  
- **Finance Teams** – for accurate revenue forecasting  
- **Customers** – for consistent product availability

---

## 🔧 Solution Overview

This project uses a **SARIMA** model, deployed via a **Streamlit app**, to forecast future sales.

### 🔄 Workflow:
1. **Data Ingestion** – Load historical sales and external market data
2. **Feature Engineering** – Market Spend,Economic Index,Competitor Price Index and lag features
3. **Modeling** – SARIMA for time-series forecasting
4. **Visualization** – Interactive Streamlit interface to explore forecasts
5. **Deployment** – Shareable and accessible forecasting tool

---

## 📈 Success Metrics

- RMSE / MAPE < industry benchmark (<10%)
- Clear visualizations and interactive dashboard
- Usability by non-technical business users
- Number of business use cases supported

---
Dataset:

## Overview  
The dataset includes historical sales data enriched with lag features and external factors relevant to forecasting. The key columns are:

**Date** – Date of the observation

**Sales** – Actual sales for the day

**Marketing_Spend** – Marketing expenditure on that date

**Economic_Index** – An indicator representing macroeconomic conditions

**Competitor_Price_Index** – Index capturing competitor pricing trends

**Sales_t-1** – Sales on the previous day (lag-1)

**Sales_t-2** – Sales two days prior (lag-2)

## 🚀 Run Locally

To run this app on your local machine:

```bash
# Clone the repo
git clone https://github.com/your-username/SalesForecastPro.git
cd SalesTrendsAI

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py

## **Contributors - Team 6 Members**
Nayana Hegde
Esther Abel
Aishwarya Jadeja
Dhrushi Padma
Haarika Atluri

---
## **Acknowledgment**
We appreciate the effort of the team and our Prof. Bala for this project, for making this project possible.

---
## Thank You!
If you would like more details, you can visit our GitHub repository.


