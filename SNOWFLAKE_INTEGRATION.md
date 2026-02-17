# PDF to PNG Converter with Snowflake Integration

## Overview

The `pdf2png.py` script has been successfully updated to utilize the Snowflake connection from `sf_connect.py`. This integration allows the system to fetch document metadata from Snowflake instead of SQL Server.

## Key Changes Made

### 1. Updated Dependencies
- Added Snowflake connector libraries to `requirements.txt`:
  - `snowflake-connector-python>=3.0.0`
  - `snowflake-sqlalchemy>=1.4.0`
  - `requests>=2.28.0`

### 2. Updated Database Connection
- **Before**: Connected to SQL Server using `pyodbc` and connection string
- **After**: Uses `sf_connect.py` module for OAuth-authenticated Snowflake connection
- Automatically handles token refresh and proper role/warehouse/schema setup

### 3. Updated SQL Query
- Modified SQL syntax for Snowflake compatibility:
  - Changed `STRING_AGG()` to `LISTAGG()`
  - Updated schema references from `HSP_RPT.dbo.*` to `PROD_EDW.DATA_STORE.*`
  - Fixed case sensitivity and JOIN syntax

### 4. Enhanced Path Handling
- Improved module import path resolution
- Added working directory management for config file access
- Better error handling for connection issues

## Usage in pdr_clean Environment

### Activate Environment
```bash
conda activate pdr_clean
```

### Basic Usage
```bash
cd /home/rmir/PDR/PDR_PDF

# Process all available PDR cases
python pdf2png.py --workers 6 --output /hspshare/converted_images

# Process only recent files (last 7 days)
python pdf2png.py --since-days 7

# Process files since a specific date
python pdf2png.py --since 2026-02-01

# Process specific cases from CSV file
python pdf2png.py --cases-csv logs/cases_needing_conversion.csv
```

### Test Integration
```bash
# Test Snowflake connection
python test_integration.py
```

## Features Preserved

- **Incremental Processing**: Tracks already processed documents per case
- **Parallel Processing**: Multi-worker support for faster conversion
- **Progress Logging**: Detailed conversion logs with status tracking
- **Flexible Filtering**: Date-based and case-specific filtering options
- **Image Optimization**: Handles large images with pixel limit controls

## Configuration

### Snowflake Connection
- Configuration managed via `/home/rmir/PDR/snowflake/config.cfg`
- OAuth authentication with automatic token refresh
- Pre-configured for Alliance Health Snowflake environment

### Output Settings
- **Default Output**: `/hspshare/converted_images`
- **Log Files**: Stored in output directory under `logs/`
- **Conversion Log**: `conversion_log.csv` tracks processing history

## Key Functions

### `fetch_metadata_from_snowflake()`
- Establishes Snowflake connection using OAuth
- Executes optimized query for PDR document metadata
- Returns pandas DataFrame with document information

### `sf_connect.connect_to_snowflake()`
- Handles OAuth token management
- Establishes connection with proper role/warehouse/schema
- Provides error handling and connection validation

## Benefits of Integration

1. **Centralized Authentication**: OAuth token management in one place
2. **Better Security**: No hardcoded credentials, token refresh handling
3. **Scalability**: Snowflake's cloud-native performance
4. **Consistency**: Unified connection pattern across data science tools
5. **Maintainability**: Separation of concerns between connection and processing logic

## Next Steps

1. **Production Testing**: Run small batch tests to validate data accuracy
2. **Performance Monitoring**: Compare processing speeds vs. SQL Server
3. **Error Handling**: Monitor for any connection timeout or token expiry issues
4. **Documentation**: Update any downstream processes that depend on this tool