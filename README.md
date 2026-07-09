1. Data Processing (src/data_processor.py)
NOTE

-Addressed the complexities in patient data locations and historical boundary definitions.

-Address Standardization: Uses fuzzy matching logic to extract the province dynamically from unstructured addresses in Diachi.
-Administrative Mapping: Uses a Python dictionary representing the newly approved National Assembly decree to automatically map the 63 pre-2025 provinces to the new 34 administrative units.
-Data Joining: Correctly merges the updated provincial cases with the 2024 HDX population dataset to calculate real incidence_rate metrics (cases per 100,000 population).

2. Machine Learning (src/model.py)
"todo-Employs powerful time-series feature engineering before forecasting"
-Feature Engineering: Derives cyclic temporal features (month_sin, month_cos) to capture the seasonality of Dengue outbreaks, alongside historical lags (cases_lag_1, cases_lag_2).
-XGBoost Algorithm: Trains an XGBRegressor on historical data and predicts the subsequent month's cases.
-During testing, the model successfully compiled and trained, demonstrating an RMSE of ~31.83.
3. GIS Mapping (src/gis_mapper.py)
-Automatically links the predicted and historical dataset metrics with the spatial vnm_admin1.geojson boundaries using the adm1_name1 properties field.
-Categorizes outbreak risk into discrete levels (Low, Medium, High, Very High) for easier visual interpretation.
-Exports a fully-compliant .geojson file (outputs/maps/dengue_risk_map.geojson) configured for direct, seamless upload to the ArcGIS Map Viewer.
4. Interactive Dashboard (src/dashboard.py)
An interactive web application built with Streamlit and Plotly that allows you to:
-Upload new Excel data directly.
-View real-time aggregated metrics (Total cases, High-Risk counts).
-Visualize a thematic choropleth map directly in the browser (if bounds allow).
-Download the processed dengue_map.geojson for advanced ArcGIS mapping via a convenient click button.
