# PDR PDF Processing

This module contains tools for processing PDF files in the PDR (Prior Drug Request) system.

## Files

- `pdf2png.py` - Incremental parallel converter that converts PDF files to PNG images
- `img_llm.py` - Image processing with LLM integration
- `pdf_img.ipynb` - Jupyter notebook for PDF image processing experiments

## pdf2png.py

A robust, incremental PDF to PNG converter with the following features:

### Key Features
- **Incremental Processing**: Tracks processed documents per case to avoid reprocessing
- **Parallel Processing**: Uses multiprocessing for efficient conversion of multiple cases
- **Smart Image Combining**: Combines multiple document images into single combined PNG files
- **Memory Management**: Splits large combined images into parts to respect memory limits
- **Database Integration**: Fetches document metadata from SQL Server database
- **Advisory Locking**: Prevents race conditions in multi-worker environments

### Usage

```bash
# Basic usage with 6 workers
python pdf2png.py --workers 6 --output /hspshare/converted_images

# Process only recent files (last 1 day)
python pdf2png.py --since-days 1

# Process files since a specific date
python pdf2png.py --since 2026-02-01

# Specify custom log folder
python pdf2png.py --workers 6 --output /hspshare/converted_images --log-folder /hspshare/converted_images/logs
```

### Configuration

The script uses several environment variables:
- `PDR_CONVERT_OUTPUT`: Default output directory
- `PDR_MAX_PIXELS`: Maximum total pixels for combined images (default: 300M)
- `PDF2IMAGE_POPPLER_PATH`: Path to Poppler binaries for PDF conversion

### Requirements

- Python 3.7+
- PIL (Pillow)
- pdf2image
- pandas
- sqlalchemy
- pyodbc (for SQL Server connectivity)

## Setup

1. Install required dependencies:
```bash
pip install pillow pdf2image pandas sqlalchemy pyodbc
```

2. Ensure Poppler is installed for PDF processing:
```bash
# On Ubuntu/Debian
sudo apt-get install poppler-utils

# On macOS
brew install poppler
```

3. Configure database connection string in the script if needed.

## Output Structure

The converter creates the following structure:
```
output_folder/
├── CASE_NUMBER_1/
│   └── CASE_NUMBER_1_combined.png
├── CASE_NUMBER_2/
│   ├── CASE_NUMBER_2_combined.png
│   └── CASE_NUMBER_2_combined_part2.png
└── logs/
    └── conversion_log.csv
```

## Logging

The script maintains a detailed conversion log (`conversion_log.csv`) that tracks:
- Cases processed
- Files converted per case
- Processing status and errors
- Output file paths
- Timestamp information