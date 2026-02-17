#!/usr/bin/env python3
"""
pdf2png.py

Incremental parallel converter that treats CASE_NUMBER as the unique case id and
tracks processed DOCUMENTNUMBER per-case using the conversion_log in the designated log folder.

Usage:
    python pdf2png.py --workers 6 --output /hspshare/converted_images --log-folder /hspshare/converted_images/logs


    python pdf2png.py --workers 6 --output /hspshare/converted_images
    # or process only 1 day of new files:
    python pdf2png.py --since-days 1
    # or process since a fixed date:
    python pdf2png.py --since 2026-02-01

"""
from __future__ import annotations

import argparse
import logging
import os
import json
from datetime import datetime, timedelta, date, timezone
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Set
import multiprocessing as mp
import sys
from pathlib import Path

# Add parent directory to path to import sf_connect
current_dir = Path(__file__).parent.absolute()
snowflake_dir = current_dir.parent / 'snowflake'
sys.path.insert(0, str(snowflake_dir))

import numpy as np
import pandas as pd
from PIL import Image
from pdf2image import convert_from_path
from sqlalchemy import create_engine, text
import sf_connect

# POSIX advisory locking (only available on Unix). If you're on Windows use `portalocker` instead.
try:
    import fcntl
except ImportError:
    fcntl = None

from PIL import Image, ImageFile

# allow loading truncated images (some PNGs are slightly truncated but loadable)
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Maximum allowed total pixels for a combined image before splitting into parts.
# Default: 150 million pixels (~150e6). Tune as needed via env var PDR_MAX_PIXELS.
MAX_TOTAL_PIXELS = int(os.environ.get("PDR_MAX_PIXELS", 300_000_000))

# ---------------------------
# Configuration
# ---------------------------
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Snowflake connection will be established via sf_connect module
# CONN_STR is now dynamically generated from Snowflake connection
CONN_STR = None  # Will be set in fetch_metadata_from_snowflake()

DEFAULT_BASE_OUTPUT = Path(os.environ.get("PDR_CONVERT_OUTPUT", "/hspshare/converted_images"))
DEFAULT_LOG_FOLDER = DEFAULT_BASE_OUTPUT / 'logs'
PDF2IMAGE_POPPLER_PATH = os.environ.get("PDF2IMAGE_POPPLER_PATH", None)
LOG_FILENAME = "conversion_log.csv"  # stored in log folder
CASES_CSV_FILENAME = "/hspshare/converted_images/logs/cases_needing_conversion.csv"  # CSV with case numbers to process

logging.basicConfig(
    level=logging.INFO,  # Keep root logger at INFO level
    format="%(asctime)s | %(levelname)-7s | %(processName)s | %(message)s"
)

# Set our own logger to DEBUG for detailed logging
logger = logging.getLogger("pdr_converter_casekey_dateonly")
logger.setLevel(logging.DEBUG)

# Suppress verbose PIL TIFF debugging messages
logging.getLogger('PIL').setLevel(logging.INFO)
logging.getLogger('PIL.TiffImagePlugin').setLevel(logging.INFO)


# ---------------------------
# SQL Query (DOCUMENT_DATE_RECEIVED column name intentionally set)
# ---------------------------
SQL_QUERY = """
 WITH files AS (
  SELECT FileID, TrackingNumber, DateOpened, ReasonID
  FROM  PROD_EDW.DATA_STORE.DS_HSP_FILES -- dbo.Files
  WHERE 
     ReasonID IN (156, 157, 239)
),

doc_mappings AS (
  SELECT fem.FileId, fem.EntityId::NUMBER AS DocumentID
  FROM   PROD_EDW.DATA_STORE.DS_HSP_FILEENTITYMAP fem 
  JOIN files f ON fem.FileId = f.FileID
  WHERE fem.EntityType = 'WIN'
),

claim_mappings AS (
  -- aggregate claim IDs per FileId (comma separated)
  SELECT fem.FileId,
         LISTAGG(fem.EntityId, ', ') WITHIN GROUP (ORDER BY fem.EntityId) AS ClaimIds
  FROM PROD_EDW.DATA_STORE.DS_HSP_FILEENTITYMAP fem
  JOIN files f ON fem.FileId = f.FileID
  WHERE fem.EntityType = 'CLM'
  GROUP BY fem.FileId
),

docs AS (
  SELECT DocumentID, DocumentNumber, Location
  FROM PROD_EDW.DATA_STORE.DS_HSP_DOCUMENTS -- dbo.documents
  WHERE DocumentID IN (SELECT DocumentID FROM doc_mappings)
)

SELECT
  f.TrackingNumber AS casenumber,
  d.DocumentNumber,
  d.DocumentID,
  -- swap "domain" for actual hostname and append DocumentNumber
  REPLACE(d.Location, 'domain', 'ccah-alliance.org') || d.DocumentNumber AS FullPath,
  cm.ClaimIds,
  f.DateOpened,
  f.ReasonID
FROM files f
JOIN doc_mappings dm ON f.FileID = dm.FileId
LEFT JOIN claim_mappings cm ON f.FileID = cm.FileId
JOIN docs d ON dm.DocumentID = d.DocumentID
 Join PROD_EDW.DATASCIENCE.PIF_LIST_ALL a on a.casenumber = f.TrackingNumber
and a.status ='OPEN'
ORDER BY d.DocumentNumber;
"""


# ---------------------------
# Case filtering utilities
# ---------------------------
def load_cases_needing_conversion(cases_csv_path: Path) -> Set[str]:
    """
    Load case numbers from the CSV file (from case_number column).
    Returns set of case numbers that need conversion.
    """
    if not cases_csv_path.exists():
        logger.warning("Cases CSV file not found at %s. Processing all cases from database.", cases_csv_path)
        return set()
    
    try:
        df_cases = pd.read_csv(cases_csv_path)
        if df_cases.empty:
            logger.warning("Cases CSV file is empty at %s", cases_csv_path)
            return set()
        
        # Look for case_number column specifically
        if 'case_number' not in df_cases.columns:
            logger.error("CSV file at %s does not have 'case_number' column. Available columns: %s", 
                        cases_csv_path, list(df_cases.columns))
            return set()
        
        # Get case numbers from case_number column
        case_numbers = df_cases['case_number'].astype(str).str.strip()
        case_numbers = case_numbers[case_numbers != ''].dropna()
        
        cases_set = set(case_numbers)
        logger.info("Loaded %d case numbers from %s (case_number column)", len(cases_set), cases_csv_path)
        return cases_set
    
    except Exception as e:
        logger.exception("Failed to read cases CSV file at %s: %s", cases_csv_path, e)
        return set()


# ---------------------------
# DB utilities
# ---------------------------
def fetch_metadata_from_snowflake(case_numbers_filter: Set[str] = None) -> pd.DataFrame:
    """Fetch metadata using Snowflake connection via sf_connect module"""
    logger.info("Connecting to Snowflake...")
    
    # Store current working directory and change to snowflake directory for config access
    import os
    original_cwd = os.getcwd()
    snowflake_dir = Path(__file__).parent.parent / 'snowflake'
    os.chdir(str(snowflake_dir))
    
    try:
        # Get Snowflake connection
        conn = sf_connect.connect_to_snowflake()
        if not conn:
            logger.error("Failed to establish Snowflake connection")
            return pd.DataFrame()
        
        try:
            cursor = conn.cursor()
            logger.info("Running query to fetch document metadata from Snowflake...")
            
            # Modify query to include case number filtering if provided
            if case_numbers_filter:
                # Create a comma-separated list of quoted case numbers for SQL IN clause
                case_list = "', '".join(case_numbers_filter)
                filtered_query = SQL_QUERY.replace(
                    "WHERE \n     ReasonID IN (156, 157, 239)",
                    f"WHERE \n     ReasonID IN (156, 157, 239)\n     AND TrackingNumber IN ('{case_list}')"
                )
                logger.info(f"Filtering query to {len(case_numbers_filter)} specific case numbers")
            else:
                filtered_query = SQL_QUERY + " LIMIT 1000"  # Safety limit if no case filter
                logger.info("No case filter provided, adding safety LIMIT 1000")
            
            # Execute query
            cursor.execute(filtered_query)
            
            # Fetch results
            columns = [desc[0] for desc in cursor.description]
            rows = cursor.fetchall()
            
            # Create DataFrame
            df = pd.DataFrame(rows, columns=columns)
            logger.info(f"Retrieved {len(df)} rows from Snowflake")
            
            if df.empty:
                logger.warning("Query returned no rows.")
                return pd.DataFrame()
            
            # Normalize column names
            df.columns = df.columns.str.upper()
            
            # Additional filtering in Python as safety (should be minimal now)
            if case_numbers_filter and not df.empty:
                initial_count = len(df)
                df['CASENUMBER'] = df['CASENUMBER'].astype(str).str.strip()
                df = df[df['CASENUMBER'].isin(case_numbers_filter)]
                logger.info("Additional Python filtering: %d rows after SQL filter, %d final rows", initial_count, len(df))
            
        except Exception as e:
            logger.exception(f"Error executing Snowflake query: {e}")
            return pd.DataFrame()
        finally:
            conn.close()
            logger.info("Snowflake connection closed.")
    
    finally:
        # Restore original working directory
        os.chdir(original_cwd)
    
    return df

def fetch_metadata(conn_str: str, case_numbers_filter: Set[str] = None) -> pd.DataFrame:
    logger.info("Connecting to database...")
    engine = create_engine(conn_str)
    try:
        with engine.connect() as conn:
            logger.info("Running query to fetch document metadata...")
            df = pd.read_sql_query(SQL_QUERY, conn)
    finally:
        engine.dispose()
        logger.info("Database connection disposed.")
    if df is None or df.shape[0] == 0:
        logger.warning("Query returned no rows.")
        return pd.DataFrame()
    
    # normalize column names
    df.columns = df.columns.str.upper()

    # Filter by case numbers if provided
    if case_numbers_filter:
        initial_count = len(df)
        df['CASE_NUMBER'] = df['CASENUMBER'].astype(str).str.strip()
        df = df[df['CASENUMBER'].isin(case_numbers_filter)]
        logger.info("Filtered to %d rows from %d based on cases CSV file", len(df), initial_count)

    # Convert DOCUMENT_DATE_RECEIVED to date-only (drop time component)
    if "DOCUMENT_DATE_RECEIVED" in df.columns:
        # convert to datetime then to date
        df["DOCUMENT_DATE_RECEIVED"] = pd.to_datetime(df["DOCUMENT_DATE_RECEIVED"], errors="coerce").dt.date

    return df


# ---------------------------
# Log-load helpers (per-case)
# ---------------------------
def load_previous_log(log_folder: Path) -> pd.DataFrame:
    log_path = log_folder / LOG_FILENAME
    if not log_path.exists():
        logger.info("No previous conversion_log found at %s (first run).", log_path)
        return pd.DataFrame()
    try:
        df_log = pd.read_csv(log_path, dtype=str)
        logger.info("Loaded previous conversion log (%d rows) from %s", len(df_log), log_path)
        # If prev log has DOCUMENT_DATE_RECEIVED, normalize it to date-only as well (if present)
        df_log.columns = df_log.columns.str.upper()
        if "DOCUMENT_DATE_RECEIVED" in df_log.columns:
            try:
                df_log["DOCUMENT_DATE_RECEIVED"] = pd.to_datetime(df_log["DOCUMENT_DATE_RECEIVED"], errors="coerce").dt.date
            except Exception:
                # ignore; keep as string
                pass
        return df_log
    except Exception:
        logger.exception("Failed to read previous conversion log at %s. Ignoring and treating as first run.", log_path)
        return pd.DataFrame()


def build_processed_by_case(prev_log: pd.DataFrame) -> Dict[str, Set[str]]:
    """
    Build mapping: { casenumber (str) : set of DOCUMENTNUMBER strings processed for that case }.
    The log is expected to have 'casenumber' column and 'processed_files' column (JSON list of dicts with 'file' key).
    """
    processed: Dict[str, Set[str]] = {}
    if prev_log.empty:
        return processed
    # normalize column names for safety
    prev_log.columns = prev_log.columns.str.upper()
    if "CASENUMBER" not in prev_log.columns or "PROCESSED_FILES" not in prev_log.columns:
        logger.warning("Previous log doesn't have 'casenumber' and/or 'processed_files' columns. No per-case processed mapping will be created.")
        return processed

    for _, row in prev_log.iterrows():
        case = str(row.get("CASENUMBER", "")).strip()
        if not case:
            continue
        entry = row.get("PROCESSED_FILES", "")
        if not entry:
            continue
        # parse JSON, fallback to eval
        try:
            items = json.loads(entry)
        except Exception:
            try:
                items = eval(entry)
            except Exception:
                items = []
        if not isinstance(items, list):
            # if it's a dict, try to extract the 'file' key
            if isinstance(items, dict) and "file" in items:
                items = [items]
            else:
                items = []
        s = processed.setdefault(case, set())
        for it in items:
            if isinstance(it, dict) and "file" in it:
                s.add(str(it["file"]))
            elif isinstance(it, str):
                s.add(it)
    logger.info("Built processed-by-case map for %d cases from previous log.", len(processed))
    return processed


def compute_since_cutoff(prev_log: pd.DataFrame, since_iso: Optional[str], since_days: Optional[int]) -> Optional[date]:
    """
    Determine cutoff as a date (not datetime). Priority:
      - explicit since_iso if provided (expects YYYY-MM-DD)
      - explicit since_days if provided (date.today() - days)
      - else derive from prev_log max DOCUMENT_DATE_RECEIVED if available
      - else None (process everything)
    """
    if since_iso:
        try:
            # parse only date portion
            parsed = datetime.fromisoformat(since_iso)
            return parsed.date()
        except Exception:
            # try parsing as date-only string
            try:
                parsed_date = datetime.strptime(since_iso, "%Y-%m-%d").date()
                return parsed_date
            except Exception:
                logger.warning("Invalid --since value '%s' ignored. Expected YYYY-MM-DD.", since_iso)

    if since_days is not None:
        return (datetime.now().date() - timedelta(days=since_days))

    # derive from prev_log if it has DOCUMENT_DATE_RECEIVED
    candidate_cols = [c for c in prev_log.columns if "DOCUMENT_DATE_RECEIVED" in c.upper() or "DATE_RECEIVED" in c.upper()]
    if not prev_log.empty and candidate_cols:
        col = candidate_cols[0]
        try:
            # prev_log should already have date-only values for DOCUMENT_DATE_RECEIVED
            prev_dates = pd.to_datetime(prev_log[col], errors="coerce").dt.date
            max_date = prev_dates.max()
            if pd.notna(max_date):
                return max_date
        except Exception:
            pass
    return None


# ---------------------------
# Conversion helpers (single-case)
# ---------------------------
def ensure_poppler_path() -> None:
    if PDF2IMAGE_POPPLER_PATH:
        os.environ["POPPLER_PATH"] = PDF2IMAGE_POPPLER_PATH
        logger.debug("Set POPPLER_PATH for pdf2image: %s", PDF2IMAGE_POPPLER_PATH)


def load_existing_combined(case_output_folder: Path, casenumber: str) -> Optional[Image.Image]:
    """
    Try to open existing combined PNG. Handle FileNotFound separately from corrupt/truncated.
    Use a quick existence check right before open and catch FileNotFoundError.
    Returns PIL.Image (copy) or None.
    """
    path = case_output_folder / f"{casenumber}_combined.png"
    if not path.exists():
        # nothing there
        return None

    # If many workers could race, attempt to acquire a short advisory lock on the folder's lockfile
    lock_path = case_output_folder / f".{casenumber}.lock"
    
    try:
        if fcntl is None:
            # no fcntl on this platform — skip locking but still re-check existence immediately
            logger.debug("fcntl not available on this platform; skipping advisory lock for %s", lock_path)
            if not path.exists():
                return None
            # proceed to open (we're vulnerable to races on non-POSIX systems)
            with Image.open(path) as im:
                return im.copy()
        else:
            # POSIX path: try non-blocking lock
            with open(lock_path, 'w') as lf:
                try:
                    fcntl.flock(lf, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    logger.debug("Could not acquire lock for %s; assume other process is handling it.", path)
                    return None
                try:
                    if not path.exists():
                        return None
                    with Image.open(path) as im:
                        return im.copy()
                finally:
                    try:
                        fcntl.flock(lf, fcntl.LOCK_UN)
                    except Exception:
                        pass
    except FileNotFoundError:
        return None
    except Exception as e:
        # If open raised FileNotFoundError while opening (race), treat as missing
        if isinstance(e, FileNotFoundError):
            logger.debug("Existing combined image %s not found at open time (race); treating as missing.", path)
            return None
        logger.exception("Unexpected error when trying to load existing combined image %s: %s", path, e)
        return None


def split_images_into_chunks(images: List[Image.Image], max_total_pixels: int) -> List[List[Image.Image]]:
    """
    Split a list of images into chunks so that each chunk's total pixels <= max_total_pixels
    Try to preserve page order. Each chunk will be stacked vertically into one output image.
    """
    chunks: List[List[Image.Image]] = []
    current: List[Image.Image] = []
    current_pixels = 0
    # Use max width of chunk conservatively by taking max width across images in chunk.
    # For pixel accounting we use sum(w*h).
    for img in images:
        pixels = img.width * img.height
        # if a single image itself is > max_total_pixels, we still put it alone in a chunk
        if pixels > max_total_pixels:
            if current:
                chunks.append(current)
                current = []
                current_pixels = 0
            chunks.append([img])
            continue
        if current_pixels + pixels <= max_total_pixels:
            current.append(img)
            current_pixels += pixels
        else:
            # start new chunk
            if current:
                chunks.append(current)
            current = [img]
            current_pixels = pixels
    if current:
        chunks.append(current)
    return chunks


def append_images_to_combined(existing: Optional[Image.Image], new_images: List[Image.Image], casenumber: str, case_output_folder: Path) -> List[str]:
    """
    Combine existing (optional) + new_images into one or more PNGs, respecting MAX_TOTAL_PIXELS.
    Returns list of output file paths created (strings). If existing is None and the new images
    produce a single combined file, that path is returned. If existing is present and small enough,
    we will combine them into a single file; if not, we will create part files for new images
    (and leave existing file untouched).
    """
    output_paths: List[str] = []

    # If there's no existing combined image, we only need to chunk and write new parts
    if existing is None:
        # split new_images into safe chunks
        chunks = split_images_into_chunks(new_images, MAX_TOTAL_PIXELS)
        for idx, chunk in enumerate(chunks, start=1):
            max_width = max(img.width for img in chunk)
            total_height = sum(img.height for img in chunk)
            combined = Image.new("RGB", (max_width, total_height), "white")
            y_offset = 0
            for img in chunk:
                if img.mode != "RGB":
                    img = img.convert("RGB")
                if img.width < max_width:
                    tmp = Image.new("RGB", (max_width, img.height), "white")
                    tmp.paste(img, (0, 0))
                    img = tmp
                combined.paste(img, (0, y_offset))
                y_offset += img.height
            # name parts sequentially
            if len(chunks) == 1:
                out_name = f"{casenumber}_combined.png"
            else:
                out_name = f"{casenumber}_combined_part{idx}.png"
            out_path = case_output_folder / out_name
            combined.save(out_path, "PNG")
            output_paths.append(str(out_path))
        return output_paths

    # existing is present. Check existing size vs MAX_TOTAL_PIXELS
    try:
        ex_w, ex_h = existing.size
        existing_pixels = ex_w * ex_h
    except Exception:
        # fallback: avoid loading, treat as absent
        existing = None
        return append_images_to_combined(None, new_images, casenumber, case_output_folder)

    # if merging existing + new images would exceed limit, do NOT attempt to load/merge
    sum_new_pixels = sum(img.width * img.height for img in new_images)
    if (existing_pixels + sum_new_pixels) > MAX_TOTAL_PIXELS:
        # prefer to leave existing file as-is and create new part(s) for the new images
        logger.warning("Merging existing combined (%d px) + new images (%d px) would exceed MAX_TOTAL_PIXELS (%d). Creating part files for new images and leaving existing file untouched.",
                       existing_pixels, sum_new_pixels, MAX_TOTAL_PIXELS)
        # create parts for new images
        new_outs = append_images_to_combined(None, new_images, casenumber, case_output_folder)
        # include original existing path in returned outputs (so log knows it's still present)
        existing_path = case_output_folder / f"{casenumber}_combined.png"
        output_paths.extend([str(existing_path)] + new_outs)
        return output_paths

    # otherwise we can merge into one final combined image
    # Build combined images list: start with existing (as single image) then new_images
    images_to_stack = [existing] + new_images
    max_width = max(img.width for img in images_to_stack)
    total_height = sum(img.height for img in images_to_stack)
    combined = Image.new("RGB", (max_width, total_height), "white")
    y_offset = 0
    for img in images_to_stack:
        if img.mode != "RGB":
            img = img.convert("RGB")
        if img.width < max_width:
            tmp = Image.new("RGB", (max_width, img.height), "white")
            tmp.paste(img, (0, 0))
            img = tmp
        combined.paste(img, (0, y_offset))
        y_offset += img.height
    out_path = case_output_folder / f"{casenumber}_combined.png"
    combined.save(out_path, "PNG")
    output_paths.append(str(out_path))
    return output_paths


def convert_unc_to_linux_path(unc_path: str) -> str:
    """Convert Windows UNC path to Linux mount path"""
    if not unc_path or not isinstance(unc_path, str):
        return unc_path
    
    linux_path = unc_path
    
    # Handle the most common UNC format: \\ccah-alliance.org\hspshare\...
    if linux_path.startswith('\\\\ccah-alliance.org\\hspshare\\'):
        linux_path = linux_path[len('\\\\ccah-alliance.org\\hspshare\\'):]
        linux_path = '/hspshare/' + linux_path
    # Handle other possible formats
    elif linux_path.startswith('\\\\ccah-alliance.org/hspshare/'):
        linux_path = linux_path[len('\\\\ccah-alliance.org/hspshare/'):]
        linux_path = '/hspshare/' + linux_path
    elif linux_path.startswith('ccah-alliance.org\\hspshare\\'):
        linux_path = linux_path[len('ccah-alliance.org\\hspshare\\'):]
        linux_path = '/hspshare/' + linux_path
    elif linux_path.startswith('ccah-alliance.org/hspshare/'):
        linux_path = linux_path[len('ccah-alliance.org/hspshare/'):]
        linux_path = '/hspshare/' + linux_path
    
    # Convert all remaining backslashes to forward slashes
    linux_path = linux_path.replace('\\', '/')
    
    return linux_path


def convert_files_to_images(file_records: List[Dict[str, Any]]) -> Tuple[List[Image.Image], List[Dict[str, Any]]]:
    images: List[Image.Image] = []
    processed_files: List[Dict[str, Any]] = []

    for rec in file_records:
        file_path = rec.get("FULLPATH")
        file_name = str(rec.get("DOCUMENTNUMBER", "unknown"))
        if not file_path or not isinstance(file_path, str):
            processed_files.append({"file": file_name, "type": "missing_path", "status": "skipped"})
            continue
        
        # Convert UNC path to Linux path
        original_path = file_path.strip()
        file_path = convert_unc_to_linux_path(original_path)
        
        file_ext = Path(file_path).suffix.lower()
        if not Path(file_path).exists():
            processed_files.append({"file": file_name, "type": "missing_file", "status": "skipped", 
                                   "original_path": original_path, "converted_path": file_path})
            continue
        try:
            if file_ext == ".pdf":
                pages = convert_from_path(file_path)
                images.extend(pages)
                processed_files.append({"file": file_name, "type": "pdf", "pages": len(pages)})
            elif file_ext in (".tif", ".tiff"):
                tif_pages = []
                with Image.open(file_path) as tif_image:
                    try:
                        while True:
                            tif_pages.append(tif_image.copy())
                            tif_image.seek(tif_image.tell() + 1)
                    except EOFError:
                        pass
                images.extend(tif_pages)
                processed_files.append({"file": file_name, "type": "tif", "pages": len(tif_pages)})
            else:
                with Image.open(file_path) as im:
                    images.append(im.copy())
                processed_files.append({"file": file_name, "type": "image", "pages": 1})
        except Exception as e:
            logger.exception("Error converting %s: %s", file_path, e)
            processed_files.append({"file": file_name, "type": "error", "error": str(e)})
    return images, processed_files


def convert_case_and_append(casenumber: str, rows: List[Dict[str, Any]], output_folder: str) -> Dict[str, Any]:
    """
    For a single case: find new documents (rows already filtered upstream),
    convert them to images, append to existing combined if present (safely),
    and save. Returns a log dict for this case. 'output_files' is a list.
    """
    out_folder = Path(output_folder)
    case_output_folder = out_folder / casenumber
    case_output_folder.mkdir(parents=True, exist_ok=True)

    logger.debug(f"Processing case {casenumber} with {len(rows)} document records")
    
    existing_combined = load_existing_combined(case_output_folder, casenumber)
    new_images, processed_files = convert_files_to_images(rows)

    if not new_images:
        # Log details about why no images were found
        skipped_files = [f for f in processed_files if f.get('status') == 'skipped']
        if skipped_files:
            logger.debug(f"Case {casenumber}: No new images - {len(skipped_files)} files skipped. Sample: {skipped_files[:3]}")
        return {
            "case": casenumber,
            "status": "no_new_images",
            "files_count": len(processed_files),
            "processed_files": processed_files,
            "output_files": []
        }

    try:
        # This returns a list of one or more output file paths.
        output_files = append_images_to_combined(existing_combined, new_images, casenumber, case_output_folder)

        return {
            "case": casenumber,
            "status": "success",
            "files_count": len(processed_files),
            "total_images_new": len(new_images),
            "output_files": output_files,
            "processed_files": processed_files,
            "run_utc": datetime.now(timezone.utc).isoformat()
        }
    except Exception as e:
        logger.exception("Error combining/appending images for case %s: %s", casenumber, e)
        return {
            "case": casenumber,
            "status": "error",
            "error": str(e),
            "files_count": len(processed_files),
            "processed_files": processed_files
        }

# ---------------------------
# Worker for multiprocessing
# ---------------------------
def worker_entry(args: Tuple[str, List[Dict[str, Any]], str]) -> Dict[str, Any]:
    casenumber, rows, output_folder = args
    try:
        ensure_poppler_path()
        res = convert_case_and_append(casenumber, rows, output_folder)
        logger.info("Worker finished case %s: %s", casenumber, res.get("status"))
        return res
    except Exception as e:
        logger.exception("Worker error for case %s: %s", casenumber, e)
        return {"case": casenumber, "status": "error", "error": str(e)}


# ---------------------------
# Parallel orchestration (per-case incremental)
# ---------------------------
def convert_cases_incremental_parallel(df: pd.DataFrame, base_output_folder: Path, workers: int,
                                       processed_by_case: Dict[str, Set[str]], since_cutoff: Optional[date]) -> pd.DataFrame:
    if df.empty:
        logger.info("No rows to process.")
        return pd.DataFrame()

    df = df.copy()
    df.columns = df.columns.str.upper()

    # DOCUMENT_DATE_RECEIVED is expected to be date-only already from fetch_metadata()
    if "DOCUMENT_DATE_RECEIVED" in df.columns:
        try:
            df["DOCUMENT_DATE_RECEIVED"] = pd.to_datetime(df["DOCUMENT_DATE_RECEIVED"], errors="coerce").dt.date
        except Exception:
            # leave as-is if conversion fails
            pass

    # Filter out rows already processed *for that case* by checking DOCUMENTNUMBER membership in processed_by_case
    def is_already_processed_row(r: pd.Series) -> bool:
        case = str(r.get("CASENUMBER", "")).strip()
        docnum = str(r.get("DOCUMENTNUMBER", "")).strip()
        if not case or not docnum:
            return False
        processed_set = processed_by_case.get(case)
        if processed_set and docnum in processed_set:
            return True
        return False

    already_mask = df.apply(is_already_processed_row, axis=1)
    initial_count = len(df)
    df = df[~already_mask]
    logger.info("Filtered out %d rows that were already processed per-case; %d remain.", initial_count - len(df), len(df))

    # apply since_cutoff (date) if provided
    if since_cutoff is not None and "DOCUMENT_DATE_RECEIVED" in df.columns:
        before = len(df)
        # compare as dates; ensure df col is date
        df = df[df["DOCUMENT_DATE_RECEIVED"].apply(lambda d: pd.NaT if pd.isna(d) else d) > since_cutoff]
        logger.info("Applied since_cutoff %s -> %d rows left (was %d).", since_cutoff.isoformat(), len(df), before)

    if df.empty:
        logger.info("No new documents to process after filters.")
        return pd.DataFrame()

    df["CASENUMBER"] = df["CASENUMBER"].astype(str).str.strip()
    grouped = df.groupby("CASENUMBER")
    work_items = []
    for casenumber, grp in grouped:
        rows = grp.to_dict(orient="records")
        work_items.append((str(casenumber), rows, str(base_output_folder)))
    total_cases = len(work_items)
    logger.info("Prepared %d case work items for processing with %d workers", total_cases, workers)

    results = []
    with mp.Pool(processes=workers) as pool:
        for i, res in enumerate(pool.imap_unordered(worker_entry, work_items), start=1):
            results.append(res)
            if i % 10 == 0 or i == total_cases:
                logger.info("Collected %d/%d results", i, total_cases)

    conversion_df = pd.DataFrame(results)
    return conversion_df


# ---------------------------
# Main
# ---------------------------
def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Incremental parallel PDR converter (case-keyed processed set, date-only).")
    parser.add_argument("--workers", type=int, default=max(1, (mp.cpu_count() or 2) - 1),
                        help="Number of worker processes")
    parser.add_argument("--output", type=str, default=str(DEFAULT_BASE_OUTPUT),
                        help="Output folder for combined images")
    parser.add_argument("--log-folder", type=str, default=str(DEFAULT_LOG_FOLDER),
                        help="Folder where conversion_log.csv will be stored (designated log folder)")
    parser.add_argument("--since-days", type=int, default=None,
                        help="Process files with DOCUMENT_DATE_RECEIVED within last N days (optional).")
    parser.add_argument("--since", type=str, default=None,
                        help="Process files with DOCUMENT_DATE_RECEIVED after this ISO date (YYYY-MM-DD).")
    parser.add_argument("--cases-csv", type=str, default=CASES_CSV_FILENAME,
                        help="Path to CSV file with case numbers to process (default: logs/cases_needing_conversion.csv)")
    args = parser.parse_args(argv)

    base_output = Path(args.output)
    base_output.mkdir(parents=True, exist_ok=True)
    log_folder = Path(args.log_folder)
    log_folder.mkdir(parents=True, exist_ok=True)
    workers = args.workers
    cases_csv_path = Path(args.cases_csv)

    logger.info("Starting incremental conversion. output=%s workers=%d log_folder=%s cases_csv=%s", 
                base_output, workers, log_folder, cases_csv_path)

    # Load case numbers that need conversion
    cases_to_process = load_cases_needing_conversion(cases_csv_path)

    prev_log = load_previous_log(log_folder)
    processed_by_case = build_processed_by_case(prev_log)
    since_cutoff = compute_since_cutoff(prev_log, args.since, args.since_days)

    try:
        # Use Snowflake connection instead of SQL Server
        df = fetch_metadata_from_snowflake(cases_to_process)
    except Exception as e:
        logger.exception("Failed to fetch metadata from Snowflake: %s", e)
        return

    if df.empty:
        logger.info("No metadata returned from DB; exiting.")
        return

    required_cols = {"CASENUMBER", "FULLPATH", "DOCUMENTNUMBER"}
    missing = required_cols - set(df.columns)
    if missing:
        logger.warning("Expected columns missing from query result: %s. Script will continue but may skip rows.", missing)

    ensure_poppler_path()

    conversion_df = convert_cases_incremental_parallel(df, base_output, workers, processed_by_case, since_cutoff)

    # Serialize processed_files column as JSON strings for safe storage
    if not conversion_df.empty and "processed_files" in conversion_df.columns:
        conversion_df["processed_files"] = conversion_df["processed_files"].apply(lambda x: json.dumps(x, default=str) if pd.notna(x) else "[]")

    final_log_path = log_folder / LOG_FILENAME
    try:
        if prev_log.empty:
            if conversion_df.empty:
                # Write an empty log so future runs see a file
                pd.DataFrame().to_csv(final_log_path, index=False)
                logger.info("Wrote empty conversion log to %s", final_log_path)
            else:
                conversion_df.to_csv(final_log_path, index=False)
                logger.info("Wrote conversion log with %d rows to %s", len(conversion_df), final_log_path)
        else:
            if conversion_df.empty:
                logger.info("No new conversion rows to append to existing log.")
            else:
                combined = pd.concat([prev_log, conversion_df], ignore_index=True, sort=False)
                # Ensure processed_files column JSON-serialized
                if "processed_files" in combined.columns:
                    def _ensure_json(val):
                        if pd.isna(val):
                            return "[]"
                        if isinstance(val, str) and val.strip().startswith("["):
                            return val
                        try:
                            return json.dumps(val, default=str)
                        except Exception:
                            return str(val)
                    combined["processed_files"] = combined["processed_files"].apply(_ensure_json)
                combined.to_csv(final_log_path, index=False)
                logger.info("Appended %d rows to existing log -> %s", len(conversion_df), final_log_path)
    except Exception:
        logger.exception("Failed to update conversion log at %s", final_log_path)

    # Summary
    if not conversion_df.empty:
        logger.info("Conversion status counts (new run):\n%s", conversion_df["status"].value_counts(dropna=False))
        success_cases = conversion_df[conversion_df["status"] == "success"]
        if not success_cases.empty:
            total_new_images = success_cases["total_images_new"].sum()
            avg_new_images = success_cases["total_images_new"].mean()
            logger.info("Successfully processed %d cases; total new images: %d; avg new images/case: %.1f",
                        len(success_cases), int(total_new_images), float(avg_new_images))

    logger.info("Incremental conversion run complete.")


if __name__ == "__main__":
    mp.freeze_support()
    main()
