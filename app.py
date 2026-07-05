"""
Financial Fraud Detection Model - Interactive Dashboard
============================================================
Capstone project dashboard built on:
  - SQLite-backed ETL pipeline (fraud_detection.db)
  - 7 trained fraud detection models (supervised + unsupervised + deep learning)
  - 100,000+ synthetic transaction dataset with realistic fraud patterns

Run with: streamlit run app.py
"""

import sqlite3
import json
import pickle

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ---------------------------------------------------------------
# Page config
# ---------------------------------------------------------------
st.set_page_config(
    page_title="Financial Fraud Detection Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

DB_PATH = "fraud_detection.db"
ARTIFACTS_DIR = "artifacts"

PRIMARY = "#1F5C99"
DANGER = "#D64045"
SUCCESS = "#2E8B57"
WARNING = "#E8A33D"
NEUTRAL = "#6B7280"

MODEL_COLORS = {
    "Logistic Regression": "#5DA3D9",
    "Random Forest": "#2E8B57",
    "XGBoost": "#1F5C99",
    "LightGBM": "#8E44AD",
    "Isolation Forest": "#E8A33D",
    "One-Class SVM": "#D64045",
}

# ---------------------------------------------------------------
# Custom CSS - Modern Creative Design
# ---------------------------------------------------------------
st.markdown("""
<style>
    /* Modern gradient background */
    .main {
        background: linear-gradient(135deg, #0F1419 0%, #1a1f2e 50%, #151b28 100%);
        color: #E8EAED;
    }
    
    /* Typography */
    h1 {
        background: linear-gradient(90deg, #00D9FF, #0099FF);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: 800;
        font-size: 2.8rem !important;
        letter-spacing: -1px;
        margin-bottom: 0.5rem;
    }
    
    h2 {
        color: #00D9FF;
        font-weight: 700;
        font-size: 1.8rem;
        border-bottom: 2px solid #FF6B6B;
        padding-bottom: 0.5rem;
    }
    
    h3 {
        color: #FFD93D;
        font-weight: 600;
        font-size: 1.3rem;
    }
    
    /* Metrics styling */
    [data-testid="stMetricValue"] {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #00D9FF, #0099FF);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 0.9rem;
        color: #A0AEC0;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(15, 20, 25, 0.6);
        padding: 8px;
        border-radius: 12px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: linear-gradient(135deg, #1a3a52 0%, #0f2744 100%);
        border-radius: 8px;
        padding: 12px 20px;
        color: #A0AEC0;
        font-weight: 600;
        transition: all 0.3s ease;
        border: 1px solid #2d5a7a;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #00D9FF, #0099FF) !important;
        color: white !important;
        box-shadow: 0 8px 24px rgba(0, 217, 255, 0.3);
        border: none;
    }
    
    /* Sticky Notes - Yellow */
    .sticky-note-yellow {
        background: linear-gradient(135deg, #FFE66D 0%, #FFC93C 100%);
        border-radius: 4px;
        padding: 16px;
        margin: 12px 0;
        box-shadow: 4px 8px 16px rgba(255, 201, 60, 0.3), 2px 4px 8px rgba(0,0,0,0.2);
        color: #3D3D3D;
        font-size: 0.9rem;
        line-height: 1.6;
        border-left: 4px solid #FFB300;
        transform: rotate(-1deg);
        position: relative;
    }
    
    .sticky-note-yellow:before {
        content: "📝";
        font-size: 1.5rem;
        position: absolute;
        top: -8px;
        right: 12px;
    }
    
    /* Sticky Notes - Pink/Red */
    .sticky-note-red {
        background: linear-gradient(135deg, #FF6B6B 0%, #FF4757 100%);
        border-radius: 4px;
        padding: 16px;
        margin: 12px 0;
        box-shadow: 4px 8px 16px rgba(255, 75, 87, 0.3), 2px 4px 8px rgba(0,0,0,0.2);
        color: #fff;
        font-size: 0.9rem;
        line-height: 1.6;
        border-left: 4px solid #CC0000;
        transform: rotate(1.2deg);
        position: relative;
    }
    
    .sticky-note-red:before {
        content: "⚠️";
        font-size: 1.5rem;
        position: absolute;
        top: -8px;
        right: 12px;
    }
    
    /* Sticky Notes - Green */
    .sticky-note-green {
        background: linear-gradient(135deg, #2ECC71 0%, #27AE60 100%);
        border-radius: 4px;
        padding: 16px;
        margin: 12px 0;
        box-shadow: 4px 8px 16px rgba(46, 204, 113, 0.3), 2px 4px 8px rgba(0,0,0,0.2);
        color: #fff;
        font-size: 0.9rem;
        line-height: 1.6;
        border-left: 4px solid #1E8449;
        transform: rotate(-0.8deg);
        position: relative;
    }
    
    .sticky-note-green:before {
        content: "✅";
        font-size: 1.5rem;
        position: absolute;
        top: -8px;
        right: 12px;
    }
    
    /* Sticky Notes - Blue */
    .sticky-note-blue {
        background: linear-gradient(135deg, #3498DB 0%, #2980B9 100%);
        border-radius: 4px;
        padding: 16px;
        margin: 12px 0;
        box-shadow: 4px 8px 16px rgba(52, 152, 219, 0.3), 2px 4px 8px rgba(0,0,0,0.2);
        color: #fff;
        font-size: 0.9rem;
        line-height: 1.6;
        border-left: 4px solid #1A5276;
        transform: rotate(0.5deg);
        position: relative;
    }
    
    .sticky-note-blue:before {
        content: "💡";
        font-size: 1.5rem;
        position: absolute;
        top: -8px;
        right: 12px;
    }
    
    /* Alert box - Redesigned */
    .alert-box {
        background: linear-gradient(135deg, rgba(255, 75, 87, 0.1) 0%, rgba(255, 107, 107, 0.1) 100%);
        border: 2px solid #FF6B6B;
        border-radius: 8px;
        padding: 14px 18px;
        margin: 12px 0;
        backdrop-filter: blur(10px);
        color: #FFD6D6;
    }
    
    .alert-box b {
        color: #FF6B6B !important;
        font-weight: 700;
    }
    
    .alert-box span {
        color: #FFA0A0 !important;
        font-size: 0.85em;
    }
    
    /* Info cards */
    .info-card {
        background: linear-gradient(135deg, #1a3a52 0%, #0f2744 100%);
        border: 1px solid #00D9FF;
        border-radius: 12px;
        padding: 20px;
        margin: 12px 0;
        box-shadow: 0 8px 24px rgba(0, 217, 255, 0.1);
    }
    
    /* Success cards */
    .success-card {
        background: linear-gradient(135deg, #1a4d2e 0%, #0f2e1f 100%);
        border: 1px solid #2ECC71;
        border-radius: 12px;
        padding: 20px;
        margin: 12px 0;
        box-shadow: 0 8px 24px rgba(46, 204, 113, 0.1);
    }
    
    /* Danger cards */
    .danger-card {
        background: linear-gradient(135deg, #5a1a1a 0%, #3d0f0f 100%);
        border: 1px solid #FF6B6B;
        border-radius: 12px;
        padding: 20px;
        margin: 12px 0;
        box-shadow: 0 8px 24px rgba(255, 107, 107, 0.1);
    }
    
    /* Section divider */
    .section-divider {
        height: 2px;
        background: linear-gradient(90deg, transparent, #00D9FF, #FF6B6B, transparent);
        margin: 20px 0;
        border-radius: 1px;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------
# Sticky Note Helper Functions
# ---------------------------------------------------------------
def sticky_note(text, note_type="yellow"):
    """Display a sticky note with emoji and styling"""
    if note_type == "yellow":
        st.markdown(f'<div class="sticky-note-yellow">{text}</div>', unsafe_allow_html=True)
    elif note_type == "red":
        st.markdown(f'<div class="sticky-note-red">{text}</div>', unsafe_allow_html=True)
    elif note_type == "green":
        st.markdown(f'<div class="sticky-note-green">{text}</div>', unsafe_allow_html=True)
    elif note_type == "blue":
        st.markdown(f'<div class="sticky-note-blue">{text}</div>', unsafe_allow_html=True)

def info_card(title, content):
    """Display an info card with gradient background"""
    st.markdown(f'<div class="info-card"><h4 style="margin-top:0; color:#00D9FF;">{title}</h4><p style="color:#A0AEC0; line-height:1.6;">{content}</p></div>', unsafe_allow_html=True)

def success_card(title, content):
    """Display a success card"""
    st.markdown(f'<div class="success-card"><h4 style="margin-top:0; color:#2ECC71;">{title}</h4><p style="color:#A0AEC0; line-height:1.6;">{content}</p></div>', unsafe_allow_html=True)

def danger_card(title, content):
    """Display a danger card"""
    st.markdown(f'<div class="danger-card"><h4 style="margin-top:0; color:#FF6B6B;">{title}</h4><p style="color:#FFD6D6; line-height:1.6;">{content}</p></div>', unsafe_allow_html=True)

def section_divider():
    """Display a decorative section divider"""
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

# ---------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------
@st.cache_data
def load_transactions():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM transactions", conn)
    conn.close()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df

@st.cache_data
def load_model_results():
    with open(f"{ARTIFACTS_DIR}/results.json") as f:
        return json.load(f)

@st.cache_data
def load_roc_data():
    with open(f"{ARTIFACTS_DIR}/roc_data.json") as f:
        return json.load(f)

@st.cache_data
def load_pr_data():
    with open(f"{ARTIFACTS_DIR}/pr_data.json") as f:
        return json.load(f)

@st.cache_data
def load_feature_importance():
    with open(f"{ARTIFACTS_DIR}/feature_importance.json") as f:
        return json.load(f)

@st.cache_data
def load_test_predictions():
    return pd.read_csv(f"{ARTIFACTS_DIR}/test_predictions.csv")

df = load_transactions()
results = load_model_results()
roc_data = load_roc_data()
pr_data = load_pr_data()
feature_importance = load_feature_importance()
test_preds = load_test_predictions()

MODEL_NAMES = list(results.keys())
SUPERVISED = ["Logistic Regression", "Random Forest", "XGBoost", "LightGBM"]
UNSUPERVISED = ["Isolation Forest", "One-Class SVM"]
# Note: Autoencoder removed for Streamlit Cloud compatibility (TensorFlow not available on Python 3.14+)

# ---------------------------------------------------------------
# Sidebar - global filters
# ---------------------------------------------------------------
st.sidebar.markdown("## 🛡️ Fraud Detection")
st.sidebar.markdown("---")
st.sidebar.markdown("### Filters")

date_min, date_max = df["timestamp"].min().date(), df["timestamp"].max().date()
date_range = st.sidebar.date_input(
    "Date range", value=(date_min, date_max), min_value=date_min, max_value=date_max
)

risk_segments = st.sidebar.multiselect(
    "Customer risk segment", options=sorted(df["customer_risk_segment"].unique()),
    default=sorted(df["customer_risk_segment"].unique())
)

channels = st.sidebar.multiselect(
    "Channel", options=sorted(df["channel"].unique()),
    default=sorted(df["channel"].unique())
)

categories = st.sidebar.multiselect(
    "Merchant category", options=sorted(df["merchant_category"].unique()),
    default=sorted(df["merchant_category"].unique())
)

amount_range = st.sidebar.slider(
    "Transaction amount (₹)",
    float(df["amount"].min()), float(df["amount"].max()),
    (float(df["amount"].min()), float(df["amount"].max()))
)

st.sidebar.markdown("---")
st.sidebar.caption("Built with Streamlit · scikit-learn · XGBoost · LightGBM · TensorFlow")
st.sidebar.caption("Data: 100,065 synthetic transactions · 1.80% fraud rate")

# Apply filters
if len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date, end_date = date_min, date_max

mask = (
    (df["timestamp"].dt.date >= start_date) &
    (df["timestamp"].dt.date <= end_date) &
    (df["customer_risk_segment"].isin(risk_segments)) &
    (df["channel"].isin(channels)) &
    (df["merchant_category"].isin(categories)) &
    (df["amount"] >= amount_range[0]) &
    (df["amount"] <= amount_range[1])
)
fdf = df[mask].copy()

# ---------------------------------------------------------------
# Header
# ---------------------------------------------------------------
st.title("🛡️ Financial Fraud Detection Model & Dashboard")
st.caption("ML-powered fraud analytics across 100K+ transactions · ETL via SQLite · 7-model comparison")

if len(fdf) == 0:
    st.warning("No transactions match the current filters. Adjust filters in the sidebar.")
    st.stop()

# ---------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------
tab_overview, tab_eda, tab_models, tab_investigate, tab_etl = st.tabs([
    "📊 Overview", "🔍 Exploratory Analysis", "🤖 Model Comparison",
    "🚨 Transaction Investigator", "⚙️ ETL & Data Pipeline"
])

# =================================================================
# TAB 1: OVERVIEW
# =================================================================
with tab_overview:
    st.markdown("### 🛡️ Fraud Detection Dashboard - Real-Time Insights")
    sticky_note("This dashboard monitors <b>100,000+ transactions</b> using <b>7 ML models</b> for rapid fraud detection. Refresh for latest data.", "blue")
    section_divider()
    
    col1, col2, col3, col4, col5 = st.columns(5)
    total_txns = len(fdf)
    total_fraud = fdf["is_fraud"].sum()
    fraud_rate = total_fraud / total_txns * 100
    total_amount = fdf["amount"].sum()
    fraud_amount = fdf.loc[fdf["is_fraud"] == 1, "amount"].sum()

    col1.metric("Total Transactions", f"{total_txns:,}")
    col2.metric("Flagged Fraudulent", f"{total_fraud:,}", f"{fraud_rate:.2f}% of total")
    col3.metric("Total Volume", f"₹{total_amount/1e6:.2f}M")
    col4.metric("Fraud Exposure", f"₹{fraud_amount/1e6:.2f}M", f"{fraud_amount/total_amount*100:.1f}% of volume")
    col5.metric("Unique Customers", f"{fdf['customer_id'].nunique():,}")

    st.markdown("---")

    c1, c2 = st.columns([2, 1])
    with c1:
        st.markdown("#### Transaction Volume & Fraud Trend Over Time")
        daily = fdf.set_index("timestamp").resample("D").agg(
            total=("transaction_id", "count"), fraud=("is_fraud", "sum")
        ).reset_index()
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(x=daily["timestamp"], y=daily["total"], name="Total Transactions",
                                  line=dict(color=PRIMARY, width=1.5), fill="tozeroy",
                                  fillcolor="rgba(31,92,153,0.08)"), secondary_y=False)
        fig.add_trace(go.Scatter(x=daily["timestamp"], y=daily["fraud"], name="Fraud Count",
                                  line=dict(color=DANGER, width=2)), secondary_y=True)
        fig.update_layout(height=380, hovermode="x unified", legend=dict(orientation="h", y=1.1),
                           margin=dict(t=30, b=10))
        fig.update_yaxes(title_text="Total Transactions", secondary_y=False)
        fig.update_yaxes(title_text="Fraud Count", secondary_y=True)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown("#### Fraud Pattern Breakdown")
        pattern_counts = fdf[fdf["is_fraud"] == 1]["fraud_pattern"].value_counts()
        if len(pattern_counts) > 0:
            fig = px.pie(values=pattern_counts.values, names=pattern_counts.index, hole=0.45,
                         color_discrete_sequence=px.colors.sequential.RdBu)
            fig.update_layout(height=380, margin=dict(t=30, b=10), showlegend=True,
                               legend=dict(orientation="h", y=-0.15, font=dict(size=10)))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No fraud cases in current filter selection.")

    c3, c4 = st.columns(2)
    with c3:
        st.markdown("#### Fraud Rate by Merchant Category")
        cat_fraud = fdf.groupby("merchant_category").agg(
            txns=("transaction_id", "count"), fraud=("is_fraud", "sum")
        )
        cat_fraud["rate"] = (cat_fraud["fraud"] / cat_fraud["txns"] * 100).round(2)
        cat_fraud = cat_fraud.sort_values("rate", ascending=True).tail(10)
        fig = px.bar(cat_fraud, x="rate", y=cat_fraud.index, orientation="h",
                     color="rate", color_continuous_scale=["#E8E8E8", DANGER],
                     labels={"rate": "Fraud Rate (%)", "y": ""})
        fig.update_layout(height=380, margin=dict(t=10, b=10), coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    with c4:
        st.markdown("#### Fraud Hotspots by City (Heatmap)")
        city_hour = fdf.pivot_table(index="city", columns="hour_of_day", values="is_fraud",
                                     aggfunc="mean", fill_value=0) * 100
        top_cities = fdf.groupby("city")["is_fraud"].sum().sort_values(ascending=False).head(10).index
        city_hour = city_hour.loc[city_hour.index.isin(top_cities)]
        fig = px.imshow(city_hour, color_continuous_scale="Reds", aspect="auto",
                         labels=dict(x="Hour of Day", y="City", color="Fraud Rate (%)"))
        fig.update_layout(height=380, margin=dict(t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

# =================================================================
# TAB 2: EDA
# =================================================================
with tab_eda:
    st.markdown("### 🔍 Exploratory Data Analysis - Understanding Patterns")
    sticky_note("<b>Key Finding:</b> Fraudsters typically spend <b>2-10x more per transaction</b> than legitimate customers, especially for high-risk merchants.", "yellow")
    section_divider()

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Amount Distribution: Fraud vs Legitimate")
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=fdf.loc[fdf["is_fraud"]==0, "amount"], name="Legitimate",
                                    marker_color=PRIMARY, opacity=0.7, nbinsx=50))
        fig.add_trace(go.Histogram(x=fdf.loc[fdf["is_fraud"]==1, "amount"], name="Fraud",
                                    marker_color=DANGER, opacity=0.7, nbinsx=50))
        fig.update_layout(barmode="overlay", height=380, xaxis_title="Amount (₹)",
                           yaxis_type="log", yaxis_title="Count (log scale)",
                           legend=dict(orientation="h", y=1.1), margin=dict(t=30))
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown("#### Transactions by Hour of Day")
        hour_data = fdf.groupby(["hour_of_day", "is_fraud"]).size().reset_index(name="count")
        hour_data["is_fraud"] = hour_data["is_fraud"].map({0: "Legitimate", 1: "Fraud"})
        fig = px.bar(hour_data, x="hour_of_day", y="count", color="is_fraud",
                     color_discrete_map={"Legitimate": PRIMARY, "Fraud": DANGER}, barmode="group")
        fig.update_layout(height=380, xaxis_title="Hour of Day (24h)", yaxis_title="Transaction Count",
                           legend_title="", margin=dict(t=10))
        st.plotly_chart(fig, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        st.markdown("#### Distance from Home City vs Fraud")
        fig = px.box(fdf, x="is_fraud", y="distance_from_home_km", color="is_fraud",
                     color_discrete_map={0: PRIMARY, 1: DANGER},
                     labels={"is_fraud": "", "distance_from_home_km": "Distance (km)"})
        fig.update_xaxes(ticktext=["Legitimate", "Fraud"], tickvals=[0, 1])
        fig.update_layout(height=380, showlegend=False, margin=dict(t=10))
        st.plotly_chart(fig, use_container_width=True)

    with c4:
        st.markdown("#### New Device Usage vs Fraud Rate")
        dev_fraud = fdf.groupby("is_new_device")["is_fraud"].mean().reset_index()
        dev_fraud["is_new_device"] = dev_fraud["is_new_device"].map({0: "Known Device", 1: "New Device"})
        dev_fraud["is_fraud"] = dev_fraud["is_fraud"] * 100
        fig = px.bar(dev_fraud, x="is_new_device", y="is_fraud", color="is_new_device",
                     color_discrete_map={"Known Device": PRIMARY, "New Device": DANGER},
                     labels={"is_fraud": "Fraud Rate (%)", "is_new_device": ""})
        fig.update_layout(height=380, showlegend=False, margin=dict(t=10))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Correlation Heatmap (Numeric Features)")
    numeric_cols = ["amount", "distance_from_home_km", "account_age_days", "hour_of_day",
                     "minutes_since_last_txn", "amount_to_avg_spend_ratio", "is_fraud"]
    corr = fdf[numeric_cols].corr().round(2)
    fig = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto")
    fig.update_layout(height=450, margin=dict(t=10))
    st.plotly_chart(fig, use_container_width=True)

# =================================================================
# TAB 3: MODEL COMPARISON
# =================================================================
with tab_models:
    st.markdown("### 🤖 Model Performance Comparison - 7 Algorithms Battle-Tested")
    st.caption("All models trained on an identical 75/25 stratified train-test split of the same feature set.")
    
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        success_card("🔵 Supervised Models", "Learn fraud patterns directly from labeled data. Best for known fraud types.")
    with col_b:
        danger_card("🟡 Unsupervised Models", "Detect statistical outliers. Catch novel/unknown fraud patterns.")
    with col_c:
        info_card("🧠 Deep Learning", "Neural networks learn complex patterns. Trade-off: slower, less interpretable.")
    
    section_divider()

    results_df = pd.DataFrame(results).T.reset_index().rename(columns={"index": "Model"})
    results_df_display = results_df[["Model", "accuracy", "precision", "recall", "f1_score", "roc_auc", "avg_precision", "train_time_sec"]]
    results_df_display.columns = ["Model", "Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC", "Avg Precision", "Train Time (s)"]

    results_df_fmt = results_df_display.copy()
    results_df_fmt["Model"] = results_df_fmt["Model"].apply(
        lambda m: f"🔵 {m}" if m in SUPERVISED else f"🟡 {m}"
    )
    results_df_fmt["Accuracy"] = results_df_fmt["Accuracy"].map(lambda x: f"{x:.2%}")
    results_df_fmt["Precision"] = results_df_fmt["Precision"].map(lambda x: f"{x:.2%}")
    results_df_fmt["Recall"] = results_df_fmt["Recall"].map(lambda x: f"{x:.2%}")
    results_df_fmt["F1 Score"] = results_df_fmt["F1 Score"].map(lambda x: f"{x:.3f}")
    results_df_fmt["ROC-AUC"] = results_df_fmt["ROC-AUC"].map(lambda x: f"{x:.4f}")
    results_df_fmt["Avg Precision"] = results_df_fmt["Avg Precision"].map(lambda x: f"{x:.4f}")
    results_df_fmt["Train Time (s)"] = results_df_fmt["Train Time (s)"].map(lambda x: f"{x:.2f}")

    st.dataframe(results_df_fmt, use_container_width=True, hide_index=True)
    st.caption("🔵 Supervised models   🟡 Unsupervised / anomaly-detection models")

    best_model = results_df_display.loc[results_df_display["F1 Score"].idxmax(), "Model"]
    st.success(f"**Best overall performer (by F1 score): {best_model}** — supervised models substantially "
               f"outperform unsupervised ones here because they learn the specific fraud patterns directly, "
               f"while unsupervised methods only flag generic statistical outliers.")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### ROC Curves")
        fig = go.Figure()
        for name in MODEL_NAMES:
            fig.add_trace(go.Scatter(x=roc_data[name]["fpr"], y=roc_data[name]["tpr"],
                                      name=f"{name} (AUC={results[name]['roc_auc']:.3f})",
                                      line=dict(color=MODEL_COLORS[name], width=2)))
        fig.add_trace(go.Scatter(x=[0,1], y=[0,1], line=dict(dash="dash", color="gray"), name="Random", showlegend=False))
        fig.update_layout(height=450, xaxis_title="False Positive Rate", yaxis_title="True Positive Rate",
                           legend=dict(font=dict(size=10)), margin=dict(t=10))
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown("#### Precision-Recall Curves")
        fig = go.Figure()
        for name in MODEL_NAMES:
            fig.add_trace(go.Scatter(x=pr_data[name]["recall"], y=pr_data[name]["precision"],
                                      name=name, line=dict(color=MODEL_COLORS[name], width=2)))
        fig.update_layout(height=450, xaxis_title="Recall", yaxis_title="Precision",
                           legend=dict(font=dict(size=10)), margin=dict(t=10))
        st.plotly_chart(fig, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        st.markdown("#### Feature Importance (Random Forest)")
        fi_df = pd.DataFrame(list(feature_importance.items()), columns=["feature", "importance"])
        fi_df = fi_df.sort_values("importance", ascending=True).tail(12)
        fig = px.bar(fi_df, x="importance", y="feature", orientation="h", color="importance",
                     color_continuous_scale=["#E8E8E8", PRIMARY])
        fig.update_layout(height=420, margin=dict(t=10), coloraxis_showscale=False, yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

    with c4:
        st.markdown("#### Confusion Matrix")
        selected_model_cm = st.selectbox("Select model", MODEL_NAMES, key="cm_select")
        cm = np.array(results[selected_model_cm]["confusion_matrix"])
        fig = px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                         labels=dict(x="Predicted", y="Actual", color="Count"),
                         x=["Legitimate", "Fraud"], y=["Legitimate", "Fraud"])
        fig.update_layout(height=370, margin=dict(t=10))
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("📖 Model Notes & Methodology"):
        st.markdown("""
**Supervised models** (Logistic Regression, Random Forest, XGBoost, LightGBM) are trained on labeled
fraud/legitimate data using `class_weight="balanced"` or `scale_pos_weight` to handle the 1.8% class imbalance.
They achieve 93-99.5% accuracy by learning exact fraud patterns from historical data.

**Unsupervised models** (Isolation Forest, One-Class SVM) are trained without fraud labels, learning what
"normal" transaction behavior looks like and flagging statistical outliers — useful for catching *novel*
fraud patterns that wouldn't be in historical labeled data.

Lower performance from unsupervised methods (97-96% vs 99.5%) is expected and informative: it demonstrates that this dataset's
fraud is closer to a learnable pattern than a pure anomaly, which is realistic for several fraud types
(card testing, account takeover) but would reverse for genuinely novel attack vectors.
        """)

# =================================================================
# TAB 4: TRANSACTION INVESTIGATOR
# =================================================================
with tab_investigate:
    st.markdown("### 🚨 High-Risk Transaction Investigator - Live Alert Feed")
    st.caption("Simulated alert feed — transactions from the test set ranked by model-predicted fraud probability.")
    sticky_note("<b>How it works:</b> Each transaction gets a risk score (0-100) from your chosen model. Higher scores = more likely fraud. Use this to prioritize investigation & block high-risk txns in real-time.", "red")
    section_divider()

    model_for_alerts = st.selectbox("Score transactions using:", SUPERVISED + UNSUPERVISED, key="alert_model")
    score_col = model_for_alerts.lower().replace(" ", "_").replace("-", "_") + "_score"

    top_n = st.slider("Number of top alerts to show", 5, 50, 15)

    alert_df = test_preds[["transaction_id", "actual_fraud", score_col]].copy()
    alert_df = alert_df.merge(
        fdf[["transaction_id", "customer_id", "amount", "merchant_category", "channel",
             "city", "timestamp", "is_new_device", "fraud_pattern"]],
        on="transaction_id", how="inner"
    )
    alert_df = alert_df.sort_values(score_col, ascending=False).head(top_n)
    alert_df["risk_score"] = (
        (alert_df[score_col] - alert_df[score_col].min()) /
        (alert_df[score_col].max() - alert_df[score_col].min() + 1e-9) * 100
    ).round(1)

    for _, row in alert_df.iterrows():
        actual_tag = "🔴 CONFIRMED FRAUD" if row["actual_fraud"] == 1 else "⚪ Flagged (legitimate)"
        st.markdown(f"""
<div class="alert-box">
<b>{row['transaction_id']}</b> · Risk Score: <b>{row['risk_score']:.1f}/100</b> · {actual_tag}<br>
Customer: {row['customer_id']} · ₹{row['amount']:,.2f} · {row['merchant_category']} via {row['channel']}<br>
<span style="color:#6B7280; font-size:0.85em;">{row['city']} · {row['timestamp']} · New device: {bool(row['is_new_device'])} · Pattern: {row['fraud_pattern']}</span>
</div>
""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### Lookup a Specific Transaction")
    txn_search = st.text_input("Enter Transaction ID (e.g. TXN0012345)")
    if txn_search:
        match = fdf[fdf["transaction_id"].str.contains(txn_search, case=False, na=False)]
        if len(match) > 0:
            st.dataframe(match, use_container_width=True, hide_index=True)
        else:
            st.info("No matching transaction found in the current filtered dataset.")

# =================================================================
# TAB 5: ETL & PIPELINE
# =================================================================
with tab_etl:
    st.markdown("### ⚙️ Data Pipeline & ETL Summary - Behind the Scenes")
    st.caption("Extract → Transform → Load process that populates `fraud_detection.db`")
    
    sticky_note("<b>Pipeline Stages:</b> Data flows from CSV → Python processing → SQLite database. Each transaction gets enriched with derived features (distance from home, time since last purchase, etc.) for ML models.", "green")
    section_divider()

    c1, c2, c3 = st.columns(3)
    c1.metric("Source Format", "CSV → SQLite")
    c2.metric("Rows Processed", f"{len(df):,}")
    c3.metric("Database Table", "transactions")

    st.markdown("""
**Extract** — Raw synthetic transaction data is generated to simulate banking/payment-gateway exports
(`fraud_transactions.csv`), covering 8,000 customers across a full year.

**Transform** —
- Missing numeric values imputed with median; missing categoricals set to "Unknown"
- Negative/invalid values clipped (amount, distance, time-since-last-transaction)
- Duplicate transaction IDs removed
- Min-max normalization applied to transaction amount
- Derived features engineered: hour of day, day of week, weekend flag, distance from home city,
  time since last transaction, amount-to-average-spend ratio

**Load** — Cleaned data written into a SQLite database (`fraud_detection.db`) with indexes on
`customer_id`, `is_fraud`, and `timestamp` for fast dashboard querying.
    """)

    st.markdown("#### Sample of Cleaned Data (from SQLite)")
    st.dataframe(fdf.head(20), use_container_width=True, hide_index=True)

    st.markdown("#### Schema")
    schema_info = pd.DataFrame({
        "Column": df.columns,
        "Dtype": [str(df[c].dtype) for c in df.columns]
    })
    st.dataframe(schema_info, use_container_width=True, hide_index=True, height=300)
