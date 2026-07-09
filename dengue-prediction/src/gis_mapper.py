import geopandas as gpd
import pandas as pd
import os

def create_map_data(agg_data, boundary_file='../data/population/vietnam_34_provinces_2025.geojson', output_file='../outputs/maps/dengue_risk_map.geojson'):
    print(f"Loading boundary GeoJSON from {boundary_file}...")
    
    try:
        gdf = gpd.read_file(boundary_file)
    except Exception as e:
        print(f"Error loading boundaries: {e}")
        return None
    
    # We want to map the latest data (last available month per province) for the map
    if 'date' in agg_data.columns:
        latest_date = agg_data['date'].max()
        map_df = agg_data[agg_data['date'] == latest_date].copy()
    else:
        map_df = agg_data.copy()
        
    print(f"Merging map data for {len(map_df)} provinces...")
    
    # Merge with boundaries
    # The geojson has properties like 'adm1_name' and 'adm1_name1' (Vietnamese names)
    # We match on the english name 'adm1_name' if our provinces match them, 
    # but our provinces are Vietnamese names from MERGE_MAP (e.g. 'Hà Nội', 'TP. Hồ Chí Minh')
    # So we should join on 'adm1_name1' which contains the Vietnamese names in the geojson.
    
    merged = gdf.merge(map_df, left_on='adm1_name1', right_on='province', how='left')
    
    # Fill missing values with 0
    merged['cases'] = merged['cases'].fillna(0)
    if 'incidence_rate' in merged.columns:
        merged['incidence_rate'] = merged['incidence_rate'].fillna(0)
    if 'predicted_cases' in merged.columns:
        merged['predicted_cases'] = merged['predicted_cases'].fillna(0)
        
    # Calculate risk level based on incidence rate
    if 'incidence_rate' in merged.columns:
        merged['risk_level'] = pd.cut(
            merged['incidence_rate'], 
            bins=[-1, 5, 20, 50, float('inf')], 
            labels=['Low', 'Medium', 'High', 'Very High']
        ).astype(str) # convert to string for geojson export
    
    # Ensure outputs dir exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Export to GeoJSON
    merged.to_file(output_file, driver='GeoJSON')
    print(f"✅ GeoJSON ready for ArcGIS Map Viewer at: {output_file}")
    
    return merged
