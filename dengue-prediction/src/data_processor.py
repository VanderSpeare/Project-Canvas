import pandas as pd
import numpy as np
import re
import logging
from datetime import datetime

try:
    from . import paths
except ImportError:
    import paths

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

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

REQUIRED_COLUMNS = {'NgayVao', 'Diachi'}


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
    logger.info(f"Loading patient data from {file_path}...")
    xls = pd.ExcelFile(file_path)

    all_cases = []

    for sheet in xls.sheet_names:
        # Only process sheets that look like dengue data (avoid ICD code sheet)
        if 'sot' in sheet.lower() or 'sốt' in sheet.lower():
            df = pd.read_excel(file_path, sheet_name=sheet)

            missing = REQUIRED_COLUMNS - set(df.columns)
            if missing:
                logger.warning(
                    f"Sheet '{sheet}' is missing expected column(s) {missing} - skipping this sheet."
                )
                continue

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

            # Surface unmapped addresses instead of silently dropping them later
            unmapped_mask = df['province'].isna()
            n_unmapped = int(unmapped_mask.sum())
            if n_unmapped > 0:
                sample = df.loc[unmapped_mask, 'address'].dropna().unique()[:5]
                logger.warning(
                    f"Sheet '{sheet}': {n_unmapped} of {len(df)} rows had an address that "
                    f"didn't match any known province and will be excluded from case counts. "
                    f"Example unmatched addresses: {list(sample)}"
                )

            all_cases.append(df)

    if not all_cases:
        return pd.DataFrame()

    combined_df = pd.concat(all_cases, ignore_index=True)

    # Filter only Dengue cases (A97 related or A90) based on ICD
    if 'icd_code' in combined_df.columns:
        combined_df['icd_code'] = combined_df['icd_code'].astype(str)
        combined_df = combined_df[combined_df['icd_code'].str.contains('A90|A97', na=False)]

    # Drop rows where province couldn't be determined (they can't be aggregated meaningfully)
    combined_df = combined_df.dropna(subset=['province'])

    # Aggregate by province and month
    combined_df.set_index('date', inplace=True)
    agg = combined_df.groupby(['province', pd.Grouper(freq='ME')]).size().reset_index(name='cases')

    return agg


def load_population_data(pop_file_path):
    logger.info(f"Loading population data from {pop_file_path}...")
    pop_df = pd.read_csv(pop_file_path)

    # Population df has ADM1_EN, T_TL (total population). We map it to new 34 provinces.
    pop_df['ADM1_EN_lower'] = pop_df['ADM1_EN'].str.lower()

    # Map old to new using robust normalization
    import unicodedata

    def strip_accents(s):
        s = s.replace('đ', 'd').replace('Đ', 'd')
        return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')

    def map_eng_to_new(name):
        name_clean = strip_accents(name).replace(' city', '').replace(' province', '').strip()
        for k, v in MERGE_MAP.items():
            k_clean = strip_accents(k).replace(' city', '').replace(' province', '').strip()
            if k_clean == name_clean or k == name_clean:
                return v
        return None

    pop_df['province'] = pop_df['ADM1_EN_lower'].apply(map_eng_to_new)

    # Check if any provinces failed to map
    unmapped = pop_df[pop_df['province'].isna()]['ADM1_EN'].unique()
    if len(unmapped) > 0:
        logger.warning(f"The following provinces could not be mapped to 34 structure: {list(unmapped)}")

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

    # Warn if any province present in the case data has no population match
    # (previously this silently produced incidence_rate = 0 via fillna with no signal)
    missing_pop = merged[merged['population'].isna()]['province'].unique()
    if len(missing_pop) > 0:
        logger.warning(f"No population data found for provinces: {list(missing_pop)}. Incidence rate set to 0 for these.")

    # Calculate incidence rate per 100,000 people
    merged['incidence_rate'] = (merged['cases'] / merged['population']) * 100000
    merged['incidence_rate'] = merged['incidence_rate'].fillna(0)

    # Add time features
    merged['year'] = merged['date'].dt.year
    merged['month'] = merged['date'].dt.month

    return merged


if __name__ == "__main__":
    # Test script
    res = load_and_process_new_data(paths.RAW_DATA_FILE, paths.POP_DATA_FILE)
    print("Sample output:")
    print(res.head())
    paths.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    res.to_csv(paths.HISTORICAL_CSV, index=False)