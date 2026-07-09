import streamlit as st
import plotly.express as px
import pandas as pd
import geopandas as gpd
import os
import json

from data_processor import load_and_process_new_data

st.set_page_config(page_title="Dengue Outbreak Monitor - VN", layout="wide")

st.title("🦟 Dengue Outbreak Monitor - Việt Nam")
st.markdown("Monitor historical cases, predict hotspots, and export GIS data for the new 34-province structure.")

# Sidebar
st.sidebar.header("Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload new patient data (Excel)", type=['xlsx'])

@st.cache_data
def load_historical():
    if os.path.exists('../data/processed/historical.csv'):
        df = pd.read_csv('../data/processed/historical.csv')
        df['date'] = pd.to_datetime(df['date'])
        return df
    return pd.DataFrame()

df = load_historical()

if uploaded_file:
    st.sidebar.success("File uploaded! Processing...")
    with st.spinner("Parsing data and mapping provinces..."):
        new_data = load_and_process_new_data(uploaded_file, '../data/population/vnm_admpop_adm1_2024.csv')
        
    if not new_data.empty:
        if not df.empty:
            df = pd.concat([df, new_data], ignore_index=True)
            # drop duplicates just in case
            df = df.drop_duplicates(subset=['date', 'province'], keep='last')
        else:
            df = new_data
        
        # Save updated historical
        df.to_csv('../data/processed/historical.csv', index=False)
        st.sidebar.success("Data processed & saved!")

if not df.empty:
    # Summary stats
    total_cases = df['cases'].sum()
    latest_month = df['date'].max()
    latest_data = df[df['date'] == latest_month]
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Cases (All Time)", f"{total_cases:,.0f}")
    col2.metric(f"Cases in {latest_month.strftime('%b %Y')}", f"{latest_data['cases'].sum():,.0f}")
    col3.metric("High Risk Provinces", len(latest_data[latest_data['incidence_rate'] > 20]))
    
    # Map
    st.subheader("Geographical Distribution (Latest Data)")
    
    # We can use Plotly Express to plot the geojson if available
    geojson_path = '../data/population/vietnam_34_provinces_2025.geojson'
    if os.path.exists(geojson_path):
        with open(geojson_path, encoding='utf-8') as f:
            vn_geojson = json.load(f)
            
        fig_map = px.choropleth(
            latest_data,
            geojson=vn_geojson,
            locations='province',
            featureidkey="properties.adm1_name1",
            color='incidence_rate',
            hover_name='province',
            hover_data=['cases', 'population'],
            color_continuous_scale="Reds",
            title="Dengue Incidence Rate (per 100k) by Province"
        )
        fig_map.update_geos(fitbounds="locations", visible=False)
        st.plotly_chart(fig_map, use_container_width=True)
    
    # Charts
    st.subheader("Temporal Trends")
    trend_df = df.groupby('date')['cases'].sum().reset_index()
    fig_trend = px.line(trend_df, x='date', y='cases', title="Monthly Dengue Cases (National)", markers=True)
    st.plotly_chart(fig_trend, use_container_width=True)
    
    st.subheader("Top Affected Provinces")
    top_provs = latest_data.sort_values(by='cases', ascending=False).head(10)
    fig_bar = px.bar(top_provs, x='province', y='cases', color='incidence_rate', color_continuous_scale='Reds')
    st.plotly_chart(fig_bar, use_container_width=True)
    
    # Download
    st.sidebar.header("Export for ArcGIS")
    map_export_path = '../outputs/maps/dengue_risk_map.geojson'
    if os.path.exists(map_export_path):
        with open(map_export_path, 'rb') as f:
            st.sidebar.download_button("Download GeoJSON", f, "dengue_map.geojson")
else:
    st.info("No data available. Please upload a patient dataset.")
