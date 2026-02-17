#!/usr/bin/env python3
"""
Configuration module for PDF to PNG conversion using Snowflake
"""

import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent
SNOWFLAKE_DIR = BASE_DIR.parent / 'snowflake'
SNOWFLAKE_CONFIG_PATH = SNOWFLAKE_DIR / 'config.cfg'

# Default output paths
DEFAULT_BASE_OUTPUT = Path(os.environ.get("PDR_CONVERT_OUTPUT", "/hspshare/converted_images"))
DEFAULT_LOG_FOLDER = DEFAULT_BASE_OUTPUT / 'logs'

# PDF2Image settings
PDF2IMAGE_POPPLER_PATH = os.environ.get("PDF2IMAGE_POPPLER_PATH", None)
MAX_TOTAL_PIXELS = int(os.environ.get("PDR_MAX_PIXELS", 300_000_000))

# Logging
LOG_FILENAME = "conversion_log.csv"
CASES_CSV_FILENAME = "logs/cases_needing_conversion.csv"

# Processing settings
RANDOM_SEED = 42
DEFAULT_WORKERS = 6

# Snowflake table mappings (update these based on your actual Snowflake schema)
SNOWFLAKE_TABLES = {
    'files': 'PROD_EDW.DATA_STORE.Files',
    'file_entity_map': 'PROD_EDW.DATA_STORE.FileEntityMap', 
    'documents': 'PROD_EDW.DATA_STORE.documents',
    'pif_ml_open': 'PROD_EDW.DATA_STORE.PIF_ML_OPEN'
}

# PDR Reason IDs
PDR_REASON_IDS = [156, 157, 239]