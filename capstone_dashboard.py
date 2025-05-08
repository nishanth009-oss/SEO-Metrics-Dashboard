# live_dashboard.py

import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.stats import ttest_ind
import statsmodels.api as sm
from prophet import Prophet
from googleapiclient.discovery import build
from oauth2client.service_account import ServiceAccountCredentials

# Set page config
st.set_page_config(page_title="🔎 SEO Analytics Dashboard", layout="wide")

# GSC credentials
KEY_FILE_LOCATION = "searchconsole-455516-f95aab4b66a5.json"
SCOPES = ['https://www.googleapis.com/auth/webmasters.readonly']
SITE_URL = 'https://locusit.se/' 

# Data fetching function
@st.cache_data(ttl=3600)
def get_search_console_data(start_date, end_date):
    credentials = ServiceAccountCredentials.from_json_keyfile_name(KEY_FILE_LOCATION, SCOPES)
    service = build('searchconsole', 'v1', credentials=credentials)

    request = {
        'startDate': start_date,
        'endDate': end_date,
        'dimensions': ['query', 'page', 'country', 'device', 'date'],
        'rowLimit': 5000
    }
    response = service.searchanalytics().query(siteUrl=SITE_URL, body=request).execute()
    rows = response.get('rows', [])

    data = []
    for row in rows:
        keys = row['keys']
        data.append({
            'query': keys[0],
            'page': keys[1],
            'country': keys[2],
            'device': keys[3],
            'date': keys[4],
            'clicks': row.get('clicks', 0),
            'impressions': row.get('impressions', 0),
            'ctr': row.get('ctr', 0),
            'position': row.get('position', 0)
        })

    return pd.DataFrame(data)

# Sidebar - Date selection
st.sidebar.header("📅 Date Range")
start_date = st.sidebar.date_input("Start Date", datetime.date.today() - datetime.timedelta(days=90))
end_date = st.sidebar.date_input("End Date", datetime.date.today())

# Load Data
st.header("🔎 SEO Performance Dashboard")
st.markdown("Fetching live data from Google Search Console...")

df = get_search_console_data(start_date.isoformat(), end_date.isoformat())
df['date'] = pd.to_datetime(df['date'])

# Keyword Summary
keyword_summary = df.groupby('query').agg({
    'clicks': 'sum',
    'impressions': 'sum',
    'ctr': 'mean',
    'position': 'mean'
}).reset_index()

# Tabs Layout
tab1, tab2, tab3, tab4, tab5 = st.tabs(["CTR vs Position", "A/B Testing", "Keyword Clustering", "Time Series", "Forecasting"])

with tab1:
    st.subheader("🔍 CTR vs Position")
    fig = px.scatter(keyword_summary, x='position', y='ctr', color='query', size='impressions',
                     title="CTR vs Google Position", labels={"position": "Google Position", "ctr": "Click Through Rate"})
    fig.update_layout(xaxis_autorange="reversed")
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("🧪 A/B Testing - CTR Mobile vs Desktop")
    mobile = df[df['device'] == 'mobile']['ctr']
    desktop = df[df['device'] == 'desktop']['ctr']
    t_stat, p_val = ttest_ind(mobile, desktop, nan_policy='omit')

    st.write(f"*T-Statistic:* {t_stat:.2f}, *P-Value:* {p_val:.3f}")

    fig = px.box(df, x='device', y='ctr', color='device',
                 title="CTR by Device (Mobile vs Desktop)")
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.subheader("📊 Keyword Clustering")
    features = keyword_summary[['ctr', 'position']]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)

    kmeans = KMeans(n_clusters=3, random_state=42)
    keyword_summary['cluster'] = kmeans.fit_predict(X_scaled)

    fig = px.scatter(keyword_summary, x='position', y='ctr', color='cluster',
                     title="Keyword Clusters Based on CTR & Position", hover_data=['query'])
    fig.update_layout(xaxis_autorange="reversed")
    st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.subheader("📈 Time Series Analysis of Clicks")
    clicks_ts = df.groupby('date')['clicks'].sum()
    decomposition = sm.tsa.seasonal_decompose(clicks_ts, model='additive', period=7)

    for col in ['observed', 'trend', 'seasonal', 'resid']:
        fig = px.line(x=clicks_ts.index, y=getattr(decomposition, col), labels={"x": "Date", "y": col.capitalize()})
        st.plotly_chart(fig, use_container_width=True)

with tab5:
    st.subheader("🔮 Predictive Analytics (Clicks Forecast)")
    prophet_df = clicks_ts.reset_index().rename(columns={"date": "ds", "clicks": "y"})
    model = Prophet(daily_seasonality=True)
    model.fit(prophet_df)
    future = model.make_future_dataframe(periods=14)
    forecast = model.predict(future)

    fig = px.line(forecast, x='ds', y='yhat', title="Predicted Clicks for Next 2 Weeks")
    fig.add_scatter(x=forecast['ds'], y=forecast['yhat_lower'], mode='lines', name='Lower Bound')
    fig.add_scatter(x=forecast['ds'], y=forecast['yhat_upper'], mode='lines', name='Upper Bound')
    st.plotly_chart(fig, use_container_width=True)

st.success("✅ Dashboard updated successfully!")

# 📈 Keyword Ranking Analysis
st.header("🏆 Keyword Ranking Analysis")

# Sorting keywords by best average position (lower is better)
top_keywords = keyword_summary.sort_values(by='position', ascending=True).head(20)

# Display as a table
st.subheader("Top 20 Keywords by Best Average Google Position")
st.dataframe(top_keywords[['query', 'position', 'clicks', 'impressions']])

# Plotting
fig = px.bar(top_keywords, x='query', y='position',
             title="Top 20 Keywords - Google Ranking Position",
             labels={"position": "Average Google Position", "query": "Keyword"},
             orientation='v')
fig.update_layout(xaxis_tickangle=-45, xaxis_title="Keyword", yaxis_title="Position", yaxis_autorange="reversed")
st.plotly_chart(fig, use_container_width=True)

# 📈 Keyword Ranking Tracking
st.header("🏅 Keyword Ranking Tracking Over Time")

# Select Top 10 Keywords overall
top_10_keywords = keyword_summary.sort_values('clicks', ascending=False).head(10)['query'].tolist()

# Filter dataset
tracking_df = df[df['query'].isin(top_10_keywords)]

# Plot
fig = px.line(tracking_df, x='date', y='position', color='query',
              title="Top 10 Keywords - Position Tracking Over Time",
              labels={"position": "Google Position", "date": "Date"})
fig.update_layout(yaxis_autorange="reversed")  # Position 1 should be at top
st.plotly_chart(fig, use_container_width=True)

# 📊 Advanced Segmentation
st.header("🌎 Advanced Segmentation Analysis")

# User selects segmentation
segment_choice = st.selectbox("Select Segment:", ["country", "device", "page"])

# Group and Aggregate
segment_summary = df.groupby(segment_choice).agg({
    'clicks': 'sum',
    'impressions': 'sum',
    'ctr': 'mean'
}).reset_index()

# Plot
fig = px.bar(segment_summary, x=segment_choice, y='clicks',
             title=f"Clicks by {segment_choice.capitalize()}",
             labels={segment_choice: segment_choice.capitalize(), "clicks": "Total Clicks"})
st.plotly_chart(fig, use_container_width=True)

# Table
st.dataframe(segment_summary)

# 🧠 Feature Engineering: CTR Buckets
st.header("🧠 CTR Behavior Based on Impressions")

# Create Buckets
df['impression_bucket'] = pd.cut(df['impressions'],
                                 bins=[0, 100, 1000, 10000, np.inf],
                                 labels=['Low (0-100)', 'Medium (100-1K)', 'High (1K-10K)', 'Very High (10K+)'])

# Group
bucket_summary = df.groupby('impression_bucket').agg({
    'clicks': 'sum',
    'impressions': 'sum',
    'ctr': 'mean'
}).reset_index()

# Plot
fig = px.bar(bucket_summary, x='impression_bucket', y='ctr',
             title="Average CTR by Impression Bucket",
             labels={"ctr": "Average CTR", "impression_bucket": "Impression Range"})
st.plotly_chart(fig, use_container_width=True)

# ⚡ Traffic Drop Alerts
st.header("⚡ Traffic Drop Detection")

# Create weekly CTR
df['week'] = df['date'].dt.isocalendar().week
weekly_ctr = df.groupby('week')['ctr'].mean().reset_index()

# Calculate % change
weekly_ctr['pct_change'] = weekly_ctr['ctr'].pct_change() * 100

# Find Alert Conditions
alerts = weekly_ctr[weekly_ctr['pct_change'] <= -30]  # Drop >30%

# Show Alerts
if not alerts.empty:
    st.error("🚨 ALERT: Major CTR drop detected!")
    st.dataframe(alerts)
else:
    st.success("✅ No major CTR drops detected!")