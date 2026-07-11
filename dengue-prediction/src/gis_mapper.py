import os
import geopandas as gpd
import pandas as pd
import logging

try:
    from . import paths
except ImportError:
    import paths

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def create_map_data(agg_data, boundary_file=None, output_file=None):
    if boundary_file is None:
        boundary_file = paths.BOUNDARY_FILE
    if output_file is None:
        output_file = paths.MAP_OUTPUT_FILE

    logger.info(f"Loading boundary GeoJSON from {boundary_file}...")

    try:
        gdf = gpd.read_file(boundary_file)
    except Exception as e:
        logger.error(f"Error loading boundaries: {e}")
        return None

    # We want to map the latest data (last available month per province) for the map
    if 'date' in agg_data.columns:
        latest_date = agg_data['date'].max()
        map_df = agg_data[agg_data['date'] == latest_date].copy()
    else:
        map_df = agg_data.copy()

    logger.info(f"Merging map data for {len(map_df)} provinces...")

    if 'adm1_name1' not in gdf.columns:
        logger.error(
            f"Expected join key 'adm1_name1' not found in boundary file. "
            f"Available columns: {list(gdf.columns)}"
        )
        return None

    # Merge with boundaries
    # The geojson has properties like 'adm1_name' and 'adm1_name1' (Vietnamese names)
    # We match on 'adm1_name1' which contains the Vietnamese names in the geojson,
    # since our provinces are Vietnamese names from MERGE_MAP (e.g. 'Hà Nội', 'TP. Hồ Chí Minh')
    merged = gdf.merge(map_df, left_on='adm1_name1', right_on='province', how='left')

    # Warn about boundary provinces with no matching data at all (vs. genuinely 0 cases)
    unmatched = merged[merged['province'].isna()]['adm1_name1'].unique()
    if len(unmatched) > 0:
        logger.warning(f"No case data found for these boundary provinces: {list(unmatched)}")

    # Fill missing values with 0
    if 'cases' in merged.columns:
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
        ).astype(str)  # convert to string for geojson export

    # Ensure outputs dir exists (works whether output_file is a str or Path)
    os.makedirs(os.path.dirname(str(output_file)), exist_ok=True)

    # Export to GeoJSON
    merged.to_file(output_file, driver='GeoJSON')
    logger.info(f"✅ GeoJSON ready for ArcGIS Map Viewer at: {output_file}")

    return merged