import streamlit as st
import plotly.express as px
import pandas as pd
import os
import json
import sys
from pathlib import Path

# Make sure this works whether Streamlit is launched as
# `streamlit run src/dashboard.py` (cwd = project root) or
# `streamlit run dashboard.py` from inside src/.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import paths
from data_processor import load_and_process_new_data

st.set_page_config(page_title="Dengue Outbreak Monitor - VN", layout="wide")

st.title("🦟 Dengue Outbreak Monitor - Việt Nam")
st.markdown("Monitor historical cases, predict hotspots, and export GIS data for the new 34-province structure.")

# Sidebar
st.sidebar.header("Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload new patient data (Excel)", type=['xlsx'])


@st.cache_data
def load_historical():
    if paths.HISTORICAL_CSV.exists():
        df = pd.read_csv(paths.HISTORICAL_CSV)
        df['date'] = pd.to_datetime(df['date'])
        return df
    return pd.DataFrame()


@st.cache_data
def load_predictions():
    pred_path = paths.PROCESSED_DIR / 'latest_predictions.csv'
    if pred_path.exists():
        return pd.read_csv(pred_path)
    return pd.DataFrame()


df = load_historical()

if uploaded_file:
    st.sidebar.success("File uploaded! Processing...")
    with st.spinner("Parsing data and mapping provinces..."):
        new_data = load_and_process_new_data(uploaded_file, paths.POP_DATA_FILE)

    if not new_data.empty:
        if not df.empty:
            df = pd.concat([df, new_data], ignore_index=True)
            # drop duplicates just in case
            df = df.drop_duplicates(subset=['date', 'province'], keep='last')
        else:
            df = new_data

        # Save updated historical
        paths.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(paths.HISTORICAL_CSV, index=False)
        st.sidebar.success("Data processed & saved!")
        st.sidebar.info("Re-run `python src/main.py` to retrain the model and refresh predictions on this new data.")

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

    if paths.BOUNDARY_FILE.exists():
        with open(paths.BOUNDARY_FILE, encoding='utf-8') as f:
            vn_geojson = json.load(f)

        # Provinces with 0 reported cases in the latest month have no row at
        # all in latest_data (aggregation only produces a row when cases > 0),
        # so previously they were simply absent from the map instead of
        # showing as 0. Fill in every province from the boundary file so the
        # whole country always renders, not just the subset with a nonzero row.
        all_provinces = [feat['properties']['adm1_name1'] for feat in vn_geojson['features']]
        map_data = pd.DataFrame({'province': all_provinces}).merge(latest_data, on='province', how='left')
        map_data['cases'] = map_data['cases'].fillna(0)
        map_data['incidence_rate'] = map_data['incidence_rate'].fillna(0)
        map_data['population'] = map_data['population'].fillna(0)

        # Using px.choropleth (the "geo" chart type) draws a vector land/country
        # base layer that, depending on the Plotly.js/browser version, can end
        # up rendered on TOP of the choropleth data layer instead of behind it -
        # which looks exactly like "the map/province layer sits above the dengue
        # layer" and makes the data appear as a flat, uncolored 0 everywhere.
        # Switching to choropleth_mapbox avoids this entirely: it uses a raster
        # tile basemap instead of a competing vector land layer, so there's no
        # z-order fight - the data colors always sit directly on top.
        # Compute a center point from all boundary coordinates for the initial view
        lats, lons = [], []
        for feat in vn_geojson['features']:
            geom = feat['geometry']
            coords = geom['coordinates']
            def flatten(c):
                if isinstance(c[0], (float, int)):
                    yield c
                else:
                    for sub in c:
                        yield from flatten(sub)
            for lon, lat in flatten(coords):
                lons.append(lon)
                lats.append(lat)
        center = {"lat": sum(lats) / len(lats), "lon": sum(lons) / len(lons)}

        fig_map = px.choropleth_mapbox(
            map_data,
            geojson=vn_geojson,
            locations='province',
            featureidkey="properties.adm1_name1",
            color='incidence_rate',
            hover_name='province',
            hover_data=['cases', 'population'],
            color_continuous_scale="Reds",
            mapbox_style="carto-positron",  # free tile style, no token needed
            center=center,
            zoom=4.3,
            opacity=0.85,
            title="Dengue Incidence Rate (per 100k) by Province"
        )
        fig_map.update_layout(height=800, margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig_map, use_container_width=True)
    else:
        st.warning(f"Boundary file not found at {paths.BOUNDARY_FILE} - map skipped.")

    # Charts
    st.subheader("Temporal Trends")
    trend_df = df.groupby('date')['cases'].sum().reset_index()
    fig_trend = px.line(trend_df, x='date', y='cases', title="Monthly Dengue Cases (National)", markers=True)
    st.plotly_chart(fig_trend, use_container_width=True)

    st.subheader("Top Affected Provinces")
    top_provs = latest_data.sort_values(by='cases', ascending=False).head(10)
    fig_bar = px.bar(top_provs, x='province', y='cases', color='incidence_rate', color_continuous_scale='Reds')
    st.plotly_chart(fig_bar, use_container_width=True)

    # Predictions - previously computed by main.py but never shown anywhere
    predictions = load_predictions()
    if not predictions.empty:
        st.subheader("📈 Next Month's Forecast")
        pred_sorted = predictions.sort_values(by='predicted_cases', ascending=False).head(10)
        fig_pred = px.bar(
            pred_sorted, x='province', y='predicted_cases',
            title="Top 10 Provinces - Predicted Cases (Next Month)",
            color='predicted_cases', color_continuous_scale='Oranges'
        )
        st.plotly_chart(fig_pred, use_container_width=True)
    else:
        st.info("No forecast available yet - run `python src/main.py` to train the model and generate predictions.")

    # Download
    st.sidebar.header("Export for ArcGIS")
    if paths.MAP_OUTPUT_FILE.exists():
        with open(paths.MAP_OUTPUT_FILE, 'rb') as f:
            st.sidebar.download_button("Download GeoJSON", f, "dengue_map.geojson")
else:
    st.info("No data available. Please upload a patient dataset.")