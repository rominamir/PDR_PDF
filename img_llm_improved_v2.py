"""
ultra_fast_viewer.py - IMPROVED

ULTRA FAST with better OCR for scanned documents!

New features:
1. Type case number directly (no scrolling!)
2. Enhanced OCR with image preprocessing for scanned docs
3. Handles low-quality scans

Usage:
  pip install opencv-python  # For better OCR
  streamlit run ultra_fast_viewer.py
"""

import os
import sqlite3
from typing import List, Dict, Tuple
from PIL import Image, ImageDraw
import streamlit as st
import json

# Try to import OpenCV for preprocessing
try:
    import cv2
    import numpy as np
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

# Increase PIL image size limit
Image.MAX_IMAGE_PIXELS = None

# Safe image size limits
MAX_DIMENSION = 10000
MAX_PIXELS = 100_000_000

# ---------------- Config ----------------
DATA_ROOT_DEFAULT = "/hspshare/converted_images"
DB_PATH = "ocr_cache.db"
SUPPORTED_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}

# Highlight colors
HIGHLIGHT_COLOR = (255, 255, 0, 100)
BORDER_COLOR = (255, 200, 0, 255)

# ---------------- Image Preprocessing for Better OCR ----------------
def preprocess_for_ocr(img: Image.Image) -> Image.Image:
    """
    Enhance image quality for better OCR on scanned documents
    Handles: low contrast, noise, skewed scans
    """
    if not HAS_CV2:
        return img.convert('L')
    
    try:
        # Convert PIL to OpenCV format
        img_array = np.array(img.convert('RGB'))
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # Apply denoising
        denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
        
        # Increase contrast with adaptive histogram equalization
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(denoised)
        
        # Apply adaptive thresholding for better text separation
        binary = cv2.adaptiveThreshold(
            enhanced, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11, 2
        )
        
        # Convert back to PIL
        result = Image.fromarray(binary)
        return result
        
    except Exception:
        return img.convert('L')

# ---------------- Safe Image Loading ----------------
def load_image_safe(image_path: str, max_dimension: int = MAX_DIMENSION) -> Image.Image:
    """Safely load image and resize if too large"""
    try:
        img = Image.open(image_path)
        
        width, height = img.size
        total_pixels = width * height
        
        if width > max_dimension or height > max_dimension or total_pixels > MAX_PIXELS:
            scale = min(max_dimension / width, max_dimension / height)
            
            if total_pixels > MAX_PIXELS:
                pixel_scale = (MAX_PIXELS / total_pixels) ** 0.5
                scale = min(scale, pixel_scale)
            
            new_width = int(width * scale)
            new_height = int(height * scale)
            
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        return img
        
    except Exception as e:
        st.error(f"Error loading: {e}")
        return Image.new('RGB', (800, 600), color='lightgray')

# ---------------- Database ----------------
@st.cache_resource
def get_db_connection():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS ocr_cache (
        filepath TEXT PRIMARY KEY,
        ocr_text TEXT,
        ocr_boxes TEXT
    )
    """)
    conn.commit()
    return conn

def get_ocr_from_cache(filepath: str) -> Tuple[str, List[Dict]]:
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT ocr_text, ocr_boxes FROM ocr_cache WHERE filepath = ?", (filepath,))
    result = cur.fetchone()
    
    if result and result[0]:
        return result[0], json.loads(result[1]) if result[1] else []
    return "", []

def save_ocr_to_cache(filepath: str, text: str, boxes: List[Dict]):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        "INSERT OR REPLACE INTO ocr_cache (filepath, ocr_text, ocr_boxes) VALUES (?, ?, ?)",
        (filepath, text, json.dumps(boxes))
    )
    conn.commit()

# ---------------- Enhanced OCR ----------------
def run_ocr_with_boxes(image_path: str) -> Tuple[str, List[Dict]]:
    """Run OCR with preprocessing for scanned documents"""
    try:
        import pytesseract
        
        img = load_image_safe(image_path)
        
        # Try with preprocessing (better for scans)
        preprocessed = preprocess_for_ocr(img)
        
        # OCR config optimized for documents
        custom_config = r'--oem 3 --psm 3'
        data = pytesseract.image_to_data(
            preprocessed, 
            output_type=pytesseract.Output.DICT,
            config=custom_config
        )
        
        words = []
        text_parts = []
        
        for i in range(len(data['text'])):
            word = data['text'][i].strip()
            conf = int(data['conf'][i])
            
            # Accept all detected text (even low confidence for scans)
            if word and conf > -1:
                words.append({
                    'text': word,
                    'bbox': {
                        'x': data['left'][i],
                        'y': data['top'][i],
                        'w': data['width'][i],
                        'h': data['height'][i]
                    }
                })
                text_parts.append(word)
        
        full_text = ' '.join(text_parts)
        
        # If too little text, try original image
        if len(text_parts) < 10:
            data = pytesseract.image_to_data(
                img, 
                output_type=pytesseract.Output.DICT,
                config=custom_config
            )
            
            words2 = []
            text_parts2 = []
            
            for i in range(len(data['text'])):
                word = data['text'][i].strip()
                conf = int(data['conf'][i])
                
                if word and conf > -1:
                    words2.append({
                        'text': word,
                        'bbox': {
                            'x': data['left'][i],
                            'y': data['top'][i],
                            'w': data['width'][i],
                            'h': data['height'][i]
                        }
                    })
                    text_parts2.append(word)
            
            # Use whichever gave more text
            if len(text_parts2) > len(text_parts):
                words = words2
                full_text = ' '.join(text_parts2)
        
        return full_text, words
        
    except Exception as e:
        return "", []

def get_ocr(filepath: str) -> Tuple[str, List[Dict]]:
    cached_text, cached_boxes = get_ocr_from_cache(filepath)
    if cached_text:
        return cached_text, cached_boxes
    
    text, boxes = run_ocr_with_boxes(filepath)
    
    if text:
        save_ocr_to_cache(filepath, text, boxes)
    
    return text, boxes

# ---------------- Discovery ----------------
@st.cache_data
def find_cases(data_root: str) -> Dict[str, List[str]]:
    cases = {}
    
    if not os.path.exists(data_root):
        return cases
    
    for case_dir in os.listdir(data_root):
        case_path = os.path.join(data_root, case_dir)
        if not os.path.isdir(case_path):
            continue
        
        images = [
            os.path.join(case_path, f)
            for f in sorted(os.listdir(case_path))
            if os.path.splitext(f)[1].lower() in SUPPORTED_EXT
        ]
        
        if images:
            cases[case_dir] = images
    
    return cases

# ---------------- Highlighting ----------------
def draw_highlights(image_path: str, boxes: List[Dict], query: str) -> Tuple[Image.Image, int]:
    img = load_image_safe(image_path)
    img = img.convert("RGBA")
    overlay = Image.new('RGBA', img.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)
    
    query_lower = query.lower()
    matches = 0
    
    for word_data in boxes:
        if query_lower in word_data['text'].lower():
            b = word_data['bbox']
            x1, y1 = b['x'], b['y']
            x2, y2 = x1 + b['w'], y1 + b['h']
            
            draw.rectangle([x1, y1, x2, y2], fill=HIGHLIGHT_COLOR)
            draw.rectangle([x1, y1, x2, y2], outline=BORDER_COLOR, width=2)
            matches += 1
    
    result = Image.alpha_composite(img, overlay).convert('RGB')
    return result, matches

# ---------------- UI ----------------
st.set_page_config(page_title="⚡ Fast Viewer", layout="wide")

st.title("⚡ Ultra Fast Case Viewer")
st.caption("Type case number or select • Enhanced OCR for scanned documents")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    DATA_ROOT = st.text_input("Folder", DATA_ROOT_DEFAULT)
    
    st.markdown("---")
    
    # Check Tesseract
    try:
        import pytesseract
        pytesseract.get_tesseract_version()
        st.success("✅ Tesseract")
    except:
        st.error("❌ Tesseract needed")
        st.code("sudo apt-get install tesseract-ocr")
    
    # Check OpenCV
    if HAS_CV2:
        st.success("✅ OpenCV (enhanced OCR)")
    else:
        st.warning("⚠️ Install for better OCR")
        st.code("pip install opencv-python")
    
    st.markdown("---")
    
    # Stats
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM ocr_cache WHERE ocr_text != ''")
    cached = cur.fetchone()[0]
    st.metric("Cached", cached)
    
    if st.button("Clear Cache"):
        conn.execute("DELETE FROM ocr_cache")
        conn.commit()
        st.rerun()

# Main
st.markdown("---")

cases = find_cases(DATA_ROOT)

if not cases:
    st.warning(f"No cases in: {DATA_ROOT}")
else:
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.subheader("📁 Cases")
        
        # Type case number
        typed_case = st.text_input(
            "Type case:",
            placeholder="DOC34179497",
            help="Type exact case number"
        )
        
        # OR select from list
        st.caption("OR select:")
        selected_list = st.selectbox(
            "List",
            [""] + sorted(cases.keys()),
            format_func=lambda x: "-- Choose --" if x == "" else f"{x} ({len(cases[x])}p)",
            label_visibility="collapsed"
        )
        
        # Which to use?
        selected = None
        if typed_case.strip():
            if typed_case in cases:
                selected = typed_case
                st.success(f"✅ {len(cases[selected])} pages")
            else:
                st.error(f"'{typed_case}' not found")
                similar = [c for c in cases if typed_case.lower() in c.lower()]
                if similar:
                    st.caption(f"Similar: {', '.join(similar[:2])}")
        elif selected_list:
            selected = selected_list
            st.success(f"✅ {len(cases[selected])} pages")
    
    with col2:
        if selected:
            st.subheader(f"📋 {selected}")
            
            images = cases[selected]
            
            # Search
            query = st.text_input("🔎 Search:", placeholder="keyword...")
            
            matches = set()
            total_matches = 0
            
            if query.strip():
                with st.spinner(f"Searching..."):
                    for i, img_path in enumerate(images):
                        text, boxes = get_ocr(img_path)
                        if text and query.lower() in text.lower():
                            matches.add(i)
                            if boxes:
                                for w in boxes:
                                    if query.lower() in w['text'].lower():
                                        total_matches += 1
                
                if matches:
                    st.success(f"✅ **{total_matches} matches** in {len(matches)} pages")
                else:
                    st.info("No matches")
            
            st.markdown("---")
            
            # Display
            for i, img_path in enumerate(images):
                try:
                    if i in matches and query.strip():
                        _, boxes = get_ocr(img_path)
                        
                        if boxes:
                            highlighted, _ = draw_highlights(img_path, boxes, query)
                            st.image(highlighted, use_container_width=True)
                        else:
                            st.image(load_image_safe(img_path), use_container_width=True)
                    else:
                        st.image(load_image_safe(img_path), use_container_width=True)
                
                except Exception as e:
                    st.error(f"Error: {e}")
        else:
            st.info("👈 Type or select a case")

st.caption("💡 Enhanced OCR for scans • Type case number for instant access")