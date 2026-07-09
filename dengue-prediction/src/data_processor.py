import pandas as pd
import numpy as np
import re
from datetime import datetime

# Mapping from 63 old provinces (and their variants) to 34 new province names
MERGE_MAP = {
    'tuyên quang': 'Tuyên Quang', 'hà giang': 'Tuyên Quang',
    'lào cai': 'Lào Cai', 'yên bái': 'Lào Cai',
    'thái nguyên': 'Thái Nguyên', 'bắc kạn': 'Thái Nguyên',
    'hải phòng': 'Hải Phòng', 'hải dương': 'Hải Phòng',
    'hồ chí minh': 'TP. Hồ Chí Minh', 'bà rịa - vũng tàu': 'TP. Hồ Chí Minh', 'bình dương': 'TP. Hồ Chí Minh',
    'phú thọ': 'Phú Thọ', 'vĩnh phúc': 'Phú Thọ', 'hòa bình': 'Phú Thọ',
    'bắc ninh': 'Bắc Ninh', 'bắc giang': 'Bắc Ninh',
    'hưng yên': 'Hưng Yên', 'thái bình': 'Hưng Yên',
    'ninh bình': 'Ninh Bình', 'hà nam': 'Ninh Bình', 'nam định': 'Ninh Bình',
    'quảng trị': 'Quảng Trị', 'quảng bình': 'Quảng Trị',
    'đà nẵng': 'Đà Nẵng', 'quảng nam': 'Đà Nẵng',
    'quảng ngãi': 'Quảng Ngãi', 'kon tum': 'Quảng Ngãi',
    'gia lai': 'Gia Lai', 'bình định': 'Gia Lai',
    'khánh hòa': 'Khánh Hòa', 'ninh thuận': 'Khánh Hòa',
    'lâm đồng': 'Lâm Đồng', 'đắk nông': 'Lâm Đồng', 'bình thuận': 'Lâm Đồng',
    'đắk lắk': 'Đắk Lắk', 'phú yên': 'Đắk Lắk',
    'đồng nai': 'Đồng Nai', 'bình phước': 'Đồng Nai',
    'tây ninh': 'Tây Ninh', 'long an': 'Tây Ninh',
    'cần thơ': 'Cần Thơ', 'sóc trăng': 'Cần Thơ', 'hậu giang': 'Cần Thơ',
    'vĩnh long': 'Vĩnh Long', 'bến tre': 'Vĩnh Long', 'trà vinh': 'Vĩnh Long',
    'đồng tháp': 'Đồng Tháp', 'tiền giang': 'Đồng Tháp',
    'cà mau': 'Cà Mau', 'bạc liêu': 'Cà Mau',
    'an giang': 'An Giang', 'kiên giang': 'An Giang',
    'hà nội': 'Hà Nội', 'huế': 'Huế', 'thừa thiên huế': 'Huế',
    'lai châu': 'Lai Châu', 'điện biên': 'Điện Biên', 'sơn la': 'Sơn La',
    'lạng sơn': 'Lạng Sơn', 'quảng ninh': 'Quảng Ninh', 'thanh hóa': 'Thanh Hóa',
    'nghệ an': 'Nghệ An', 'hà tĩnh': 'Hà Tĩnh', 'cao bằng': 'Cao Bằng'
}

def extract_province(address_str):
    if pd.isna(address_str):
        return None
    address_lower = address_str.lower()
    
    # Handle known variations
    address_lower = address_lower.replace('thành phố hồ chí minh', 'hồ chí minh')
    address_lower = address_lower.replace('tp hồ chí minh', 'hồ chí minh')
    address_lower = address_lower.replace('t.p hồ chí minh', 'hồ chí minh')
    
    # Find match from our mapping keys
    for old_prov in sorted(MERGE_MAP.keys(), key=len, reverse=True):
        if old_prov in address_lower:
            return MERGE_MAP[old_prov]
    
    return None

def process_patient_data(file_path):
    print(f"Loading patient data from {file_path}...")
    xls = pd.ExcelFile(file_path)
    
    all_cases = []
    
    for sheet in xls.sheet_names:
        # Only process sheets that look like dengue data (avoid ICD code sheet)
        if 'sot' in sheet.lower() or 'sốt' in sheet.lower():
            df = pd.read_excel(file_path, sheet_name=sheet)
            
            # Map columns to standard names
            col_map = {
                'NgayVao': 'date',
                'NgayRa': 'discharge_date',
                'Diachi': 'address',
                'NamSinh': 'birth_year',
                'Maicd': 'icd_code',
                'TenBenhNhan': 'patient_name'
            }
            df = df.rename(columns=col_map)
            
            # Only keep rows with valid admission date
            df = df.dropna(subset=['date'])
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            df = df.dropna(subset=['date'])
            
            # Extract new province name
            df['province'] = df['address'].apply(extract_province)
            
            all_cases.append(df)
            
    if not all_cases:
        return pd.DataFrame()
        
    combined_df = pd.concat(all_cases, ignore_index=True)
    
    # Filter only Dengue cases (A97 related or A90) based on ICD
    if 'icd_code' in combined_df.columns:
        combined_df['icd_code'] = combined_df['icd_code'].astype(str)
        combined_df = combined_df[combined_df['icd_code'].str.contains('A90|A97', na=False)]
    
    # Aggregate by province and month
    combined_df.set_index('date', inplace=True)
    agg = combined_df.groupby(['province', pd.Grouper(freq='M')]).size().reset_index(name='cases')
    
    return agg

def load_population_data(pop_file_path):
    print(f"Loading population data from {pop_file_path}...")
    pop_df = pd.read_csv(pop_file_path)
    
    # Population df has ADM1_EN, T_TL (total population). We map it to new 34 provinces.
    pop_df['ADM1_EN_lower'] = pop_df['ADM1_EN'].str.lower()
    
    # Handle known variations in english dataset
    pop_df['ADM1_EN_lower'] = pop_df['ADM1_EN_lower'].replace('ho chi minh city', 'hồ chí minh')
    pop_df['ADM1_EN_lower'] = pop_df['ADM1_EN_lower'].replace('hai phong city', 'hải phòng')
    pop_df['ADM1_EN_lower'] = pop_df['ADM1_EN_lower'].replace('can tho city', 'cần thơ')
    pop_df['ADM1_EN_lower'] = pop_df['ADM1_EN_lower'].replace('da nang city', 'đà nẵng')
    
    # Map old to new
    def map_eng_to_new(name):
        for k, v in MERGE_MAP.items():
            # simple ascii comparison
            import unicodedata
            k_ascii = unicodedata.normalize('NFKD', k).encode('ASCII', 'ignore').decode('utf-8')
            if k_ascii == name or k == name:
                return v
        return None
        
    pop_df['province'] = pop_df['ADM1_EN_lower'].apply(map_eng_to_new)
    
    # Sum population for the merged 34 provinces
    pop_agg = pop_df.groupby('province')['T_TL'].sum().reset_index(name='population')
    
    return pop_agg

def load_and_process_new_data(patient_file, pop_file):
    cases_df = process_patient_data(patient_file)
    pop_df = load_population_data(pop_file)
    
    # Join cases with population
    if cases_df.empty:
        return cases_df
        
    merged = pd.merge(cases_df, pop_df, on='province', how='left')
    
    # Calculate incidence rate per 100,000 people
    merged['incidence_rate'] = (merged['cases'] / merged['population']) * 100000
    merged['incidence_rate'] = merged['incidence_rate'].fillna(0)
    
    # Add time features
    merged['year'] = merged['date'].dt.year
    merged['month'] = merged['date'].dt.month
    
    return merged

if __name__ == "__main__":
    # Test script
    patient_f = '../data/raw/historical_patients.xlsx'
    pop_f = '../data/population/vnm_admpop_adm1_2024.csv'
    res = load_and_process_new_data(patient_f, pop_f)
    print("Sample output:")
    print(res.head())
    res.to_csv('../data/processed/historical.csv', index=False)
