#!/usr/bin/env python3
"""
Test script to run PDF query and attempt to open/display documents
"""

import sys
import os
from pathlib import Path
import pandas as pd
import subprocess
from PIL import Image

# Add the snowflake directory to Python path
snowflake_dir = Path(__file__).parent.parent / 'snowflake'
sys.path.append(str(snowflake_dir))

# Change to snowflake directory so config.cfg can be found
original_cwd = os.getcwd()
os.chdir(str(snowflake_dir))

# SQL Query from pdf2png.py
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
ORDER BY d.DocumentNumber
LIMIT 20;
"""

def is_pdf_or_tiff(filepath):
    """Check if file is PDF or TIFF based on extension"""
    if not filepath:
        return False
    path_str = str(filepath).lower()
    return path_str.endswith(('.pdf', '.tiff', '.tif'))

def convert_unc_to_linux_path(unc_path):
    """Convert UNC path to Linux mount path"""
    if not unc_path or not isinstance(unc_path, str):
        return None
    
    # Replace UNC path with Linux mount path
    linux_path = unc_path.replace('\\\\ccah-alliance.org\\hspshare\\', '/hspshare/')
    linux_path = linux_path.replace('\\', '/')
    return linux_path

def attempt_open_document(filepath):
    """Attempt to open and display basic info about document"""
    try:
        linux_path = convert_unc_to_linux_path(filepath)
        if not linux_path:
            return f"❌ Could not convert path: {filepath}"
            
        path_obj = Path(linux_path)
        
        # Check if file exists
        if not path_obj.exists():
            return f"❌ File not found: {linux_path}"
            
        # Get file info
        file_size = path_obj.stat().st_size
        file_ext = path_obj.suffix.lower()
        
        result = f"✅ Found {file_ext} file: {path_obj.name} ({file_size:,} bytes)"
        
        # If it's a TIFF, try to get image info
        if file_ext in ['.tiff', '.tif']:
            try:
                with Image.open(linux_path) as img:
                    result += f" - {img.size[0]}x{img.size[1]} pixels, {img.mode} mode"
            except Exception as e:
                result += f" - Could not read image info: {e}"
        
        # If it's a PDF, try to get page count
        elif file_ext == '.pdf':
            try:
                # Use pdfinfo if available
                cmd_result = subprocess.run(['pdfinfo', linux_path], 
                                          capture_output=True, text=True, timeout=5)
                if cmd_result.returncode == 0:
                    lines = cmd_result.stdout.split('\n')
                    for line in lines:
                        if 'Pages:' in line:
                            pages = line.split(':')[1].strip()
                            result += f" - {pages} pages"
                            break
                else:
                    result += " - Could not get PDF info"
            except Exception as e:
                result += f" - Could not get PDF info: {e}"
                
        return result
        
    except Exception as e:
        return f"❌ Error accessing {filepath}: {e}"

try:
    import sf_connect
    print("✅ Successfully imported sf_connect module")
    
    # Test config loading
    config = sf_connect.load_config()
    if config:
        print("✅ Config file loaded successfully")
        
        # Test Snowflake connection and run query
        print("🔄 Testing Snowflake connection...")
        conn = sf_connect.connect_to_snowflake()
        if conn:
            print("✅ Snowflake connection established successfully")
            
            try:
                cursor = conn.cursor()
                print("🔄 Running PDF/TIFF document query...")
                
                # Execute the query
                cursor.execute(SQL_QUERY)
                
                # Get results
                columns = [desc[0] for desc in cursor.description]
                rows = cursor.fetchall()
                
                if rows:
                    df = pd.DataFrame(rows, columns=columns)
                    print(f"✅ Retrieved {len(df)} documents")
                    
                    # Filter for PDF/TIFF files
                    pdf_tiff_docs = df[df['FULLPATH'].apply(is_pdf_or_tiff)]
                    print(f"📄 Found {len(pdf_tiff_docs)} PDF/TIFF documents")
                    
                    if not pdf_tiff_docs.empty:
                        print("\n📂 Document Details:")
                        for idx, row in pdf_tiff_docs.head(10).iterrows():  # Limit to first 10
                            print(f"\n🔍 Case: {row['CASENUMBER']}, Doc: {row['DOCUMENTNUMBER']}")
                            print(f"   Path: {row['FULLPATH']}")
                            
                            # Attempt to open the document
                            open_result = attempt_open_document(row['FULLPATH'])
                            print(f"   {open_result}")
                    else:
                        print("❌ No PDF/TIFF documents found in results")
                else:
                    print("❌ No results returned from query")
                    
            except Exception as e:
                print(f"❌ Error running query: {e}")
            finally:
                conn.close()
                print("✅ Connection closed successfully")
        else:
            print("❌ Failed to connect to Snowflake")
    else:
        print("❌ Failed to load config")
        
except ImportError as e:
    print(f"❌ Import error: {e}")
except Exception as e:
    print(f"❌ Error during testing: {e}")
finally:
    # Restore original working directory
    os.chdir(original_cwd)
    # Restore original working directory
    os.chdir(original_cwd)