import os
import sys
import pandas as pd
import subprocess

# Set stdout to UTF-8
sys.stdout.reconfigure(encoding='utf-8')

from src.data_processor import load_and_process_new_data
from src.model import train_model, predict_future
from src.gis_mapper import create_map_data

def main():
    print("🚀 Starting Dengue Outbreak Processing Pipeline...")
    
    # 1. Process Raw Data
    raw_file = 'data/raw/historical_patients.xlsx'
    pop_file = 'data/population/vnm_admpop_adm1_2024.csv'
    
    if not os.path.exists(raw_file):
        print(f"❌ Raw file {raw_file} not found. Please place the patient data there.")
        return
        
    print("\n--- 1. Processing Data ---")
    new_data = load_and_process_new_data(raw_file, pop_file)
    
    if new_data.empty:
        print("❌ No data processed.")
        return
        
    print(f"✅ Processed {len(new_data)} aggregated records.")
    
    # Save to processed folder
    os.makedirs('data/processed', exist_ok=True)
    hist_path = 'data/processed/historical.csv'
    new_data.to_csv(hist_path, index=False)
    
    # 2. Train Model and Predict
    print("\n--- 2. Training Predictive Model ---")
    model = train_model(new_data)
    
    if model:
        print("\n--- 3. Generating Predictions ---")
        # Predict for next month based on latest month's data
        latest_date = new_data['date'].max()
        latest_data_subset = new_data[new_data['date'] == latest_date]
        
        predictions = predict_future(model, latest_data_subset)
        
        # Merge predictions back for mapping
        map_df = pd.merge(
            latest_data_subset, 
            predictions[['province', 'predicted_cases']], 
            on='province', 
            how='left'
        )
    else:
        map_df = new_data
        
    # 3. GIS Mapping
    print("\n--- 4. Creating GIS GeoJSON ---")
    create_map_data(map_df, boundary_file='data/population/vietnam_34_provinces_2025.geojson')
    
    print("\n🎉 Pipeline Complete! To run the dashboard, use:")
    print("streamlit run src/dashboard.py")
    
if __name__ == "__main__":
    main()
