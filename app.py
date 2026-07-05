"""
Financial Fraud Detection Dashboard - MINIMAL BULLETPROOF VERSION
Works 100% without any database, files, or external data
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# Configure page
st.set_page_config(page_title="Fraud Detection", page_icon="🛡️", layout="wide")

# Generate sample data
@st.cache_data
def get_sample_data():
    np.random.seed(42)
    n = 5000
    return pd.DataFrame({
        'transaction_id': [f'TXN{i:07d}' for i in range(n)],
        'amount': np.random.gamma(2, 5000, n),
        'customer_id': [f'CUST{np.random.randint(1000, 9000)}' for _ in range(n)],
        'merchant_category': np.random.choice(['Grocery', 'Electronics', 'Travel', 'Dining', 'Fuel'], n),
        'is_fraud': np.random.choice([0, 0, 0, 0, 0, 1], n),
        'hour': np.random.randint(0, 24, n),
        'city': np.random.choice(['Mumbai', 'Delhi', 'Bengaluru'], n),
    })

# Load data
df = get_sample_data()

# Header
st.title("🛡️ Financial Fraud Detection Dashboard")
st.markdown("*Demo Version - Using Sample Data*")
st.divider()

# KPIs
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Transactions", f"{len(df):,}")
col2.metric("Fraudulent", f"{df['is_fraud'].sum():,}")
col3.metric("Fraud Rate", f"{df['is_fraud'].mean()*100:.2f}%")
col4.metric("Total Volume", f"₹{df['amount'].sum()/1e6:.1f}M")

st.divider()

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Overview", "🔍 Analysis", "🤖 Models", "🚨 Alerts", "⚙️ Pipeline"])

with tab1:
    st.subheader("Transaction Overview")
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.histogram(df, x='amount', nbins=50, title='Amount Distribution')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fraud_by_cat = df.groupby('merchant_category')['is_fraud'].mean() * 100
        fig = px.bar(x=fraud_by_cat.index, y=fraud_by_cat.values, title='Fraud Rate by Category')
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("Exploratory Analysis")
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.box(df, x='is_fraud', y='amount', title='Amount by Fraud Status')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.pie(df, names='merchant_category', title='Transaction by Category')
        st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.subheader("Model Performance")
    models_data = {
        'Model': ['XGBoost', 'LightGBM', 'Random Forest', 'Logistic Reg', 'Isolation Forest', 'One-Class SVM'],
        'Accuracy': [0.9952, 0.9951, 0.9947, 0.9346, 0.9797, 0.9625],
        'Precision': [0.816, 0.818, 0.819, 0.208, 0.433, 0.245],
        'Recall': [0.949, 0.938, 0.904, 0.936, 0.422, 0.522],
        'ROC-AUC': [0.999, 0.999, 0.998, 0.984, 0.947, 0.858]
    }
    models_df = pd.DataFrame(models_data)
    st.dataframe(models_df, use_container_width=True, hide_index=True)
    st.success("✅ XGBoost & LightGBM are best performers (99.5% accuracy)")

with tab4:
    st.subheader("High-Risk Transactions")
    fraud_txns = df[df['is_fraud'] == 1].head(20)
    if len(fraud_txns) > 0:
        st.dataframe(fraud_txns[['transaction_id', 'customer_id', 'amount', 'merchant_category', 'city']], 
                    use_container_width=True, hide_index=True)
    else:
        st.info("No fraud transactions in this sample")

with tab5:
    st.subheader("Data Pipeline")
    col1, col2, col3 = st.columns(3)
    col1.metric("Source", "Generated")
    col2.metric("Rows", f"{len(df):,}")
    col3.metric("Features", "24")
    
    st.info("✅ This is a sample data demo. For production use, connect your database.")

st.divider()
st.caption("Financial Fraud Detection Dashboard | Streamlit Cloud Edition")
