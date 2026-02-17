"""
ultra_fast_viewer.py

ULTRA FAST - Images display immediately, OCR on first search only!

Speed optimizations:
1. Images display INSTANTLY (no OCR wait)
2. OCR only runs when you first search
3. Results cached permanently
4. No unnecessary processing
5. Smart image resizing to prevent errors

Usage:
  streamlit run ultra_fast_viewer.py
"""

import os
import sqlite3
from typing import List, Dict, Tuple
from PIL import Image, ImageDraw
import streamlit as st
import json

# Increase PIL image size limit (prevents decompression bomb error)
Image.MAX_IMAGE_PIXELS = None  # Remove limit, we'll handle it ourselves

# Safe image size limits
MAX_DIMENSION = 10000  # Max width or height in pixels
MAX_PIXELS = 100_000_000  # Max total pixels (~10000x10000)

# ---------------- Config ----------------
DATA_ROOT_DEFAULT = "/hspshare/converted_images"
DB_PATH = "ocr_cache.db"
SUPPORTED_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}

# Highlight colors
HIGHLIGHT_COLOR = (255, 255, 0, 100)
BORDER_COLOR = (255, 200, 0, 255)

# ---------------- Safe Image Loading ----------------
def load_image_safe(image_path: str, max_dimension: int = MAX_DIMENSION) -> Image.Image:
    """
    Safely load image and resize if too large
    Prevents decompression bomb errors
    """
    try:
        # Open image
        img = Image.open(image_path)
        
        # Check size
        width, height = img.size
        total_pixels = width * height
        
        # If image is too large, resize it
        if width > max_dimension or height > max_dimension or total_pixels > MAX_PIXELS:
            # Calculate scale factor
            scale = min(max_dimension / width, max_dimension / height)
            
            # Also check total pixels
            if total_pixels > MAX_PIXELS:
                pixel_scale = (MAX_PIXELS / total_pixels) ** 0.5
                scale = min(scale, pixel_scale)
            
            new_width = int(width * scale)
            new_height = int(height * scale)
            
            # Resize
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        return img
        
    except Exception as e:
        st.error(f"Error loading image: {e}")
        # Return a small placeholder image
        return Image.new('RGB', (800, 600), color='lightgray')

# ---------------- Database ----------------
@st.cache_resource
def get_db_connection():
    """Reuse single database connection"""
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
    """Get OCR from cache - returns empty if not cached"""
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT ocr_text, ocr_boxes FROM ocr_cache WHERE filepath = ?", (filepath,))
    result = cur.fetchone()
    
    if result and result[0]:
        return result[0], json.loads(result[1]) if result[1] else []
    return "", []

def save_ocr_to_cache(filepath: str, text: str, boxes: List[Dict]):
    """Save OCR to cache"""
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        "INSERT OR REPLACE INTO ocr_cache (filepath, ocr_text, ocr_boxes) VALUES (?, ?, ?)",
        (filepath, text, json.dumps(boxes))
    )
    conn.commit()

# ---------------- OCR ----------------
def run_ocr_with_boxes(image_path: str) -> Tuple[str, List[Dict]]:
    """Run OCR and get word positions"""
    try:
        import pytesseract
        
        # Load image safely (handles large images)
        img = load_image_safe(image_path)
        
        # Run OCR
        data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
        
        words = []
        text_parts = []
        
        for i in range(len(data['text'])):
            word = data['text'][i].strip()
            conf = int(data['conf'][i])
            
            if word and conf > 0:
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
        
        return ' '.join(text_parts), words
    except Exception as e:
        st.warning(f"OCR failed for {os.path.basename(image_path)}: {e}")
        return "", []

def get_ocr(filepath: str) -> Tuple[str, List[Dict]]:
    """Get OCR - from cache or run fresh"""
    # Check cache first
    cached_text, cached_boxes = get_ocr_from_cache(filepath)
    if cached_text:
        return cached_text, cached_boxes
    
    # Run OCR
    text, boxes = run_ocr_with_boxes(filepath)
    
    # Save to cache
    if text:
        save_ocr_to_cache(filepath, text, boxes)
    
    return text, boxes

# ---------------- Discovery ----------------
@st.cache_data
def find_cases(data_root: str) -> Dict[str, List[str]]:
    """Find all cases - cached for speed"""
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
    """Draw yellow highlights on matching words"""
    # Load image safely
    img = load_image_safe(image_path)
    
    # Convert to RGBA for overlay
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
st.set_page_config(page_title="⚡ Ultra Fast Viewer", layout="wide")

st.title("⚡ Ultra Fast Case Viewer")
st.caption("Images load instantly • Search triggers OCR only when needed")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    DATA_ROOT = st.text_input("Folder", DATA_ROOT_DEFAULT)
    
    st.markdown("---")
    
    # Check Tesseract
    try:
        import pytesseract
        pytesseract.get_tesseract_version()
        st.success("✅ Tesseract Ready")
    except:
        st.error("❌ Install Tesseract")
        st.code("sudo apt-get install tesseract-ocr")
    
    st.markdown("---")
    
    # Cache stats
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM ocr_cache WHERE ocr_text != ''")
    cached = cur.fetchone()[0]
    st.metric("Cached", cached)

# Main
st.markdown("---")

cases = find_cases(DATA_ROOT)

if not cases:
    st.warning(f"No cases in: {DATA_ROOT}")
else:
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.subheader("📁 Cases")
        selected = st.selectbox(
            "Select:",
            [""] + sorted(cases.keys()),
            format_func=lambda x: "-- Choose --" if x == "" else f"{x} ({len(cases[x])})"
        )
        
        if selected:
            st.success(f"{len(cases[selected])} pages")
    
    with col2:
        if selected:
            st.subheader(f"📋 {selected}")
            
            images = cases[selected]
            
            # Search
            query = st.text_input("🔎 Search:", placeholder="Type keyword...")
            
            # Find matches (only if searching)
            matches = set()
            total_keyword_matches = 0
            
            if query.strip():
                with st.spinner(f"Searching {len(images)} pages..."):
                    for i, img_path in enumerate(images):
                        text, boxes = get_ocr(img_path)
                        if text and query.lower() in text.lower():
                            matches.add(i)
                            # Count how many times keyword appears in this page
                            if boxes:
                                for word_data in boxes:
                                    if query.lower() in word_data['text'].lower():
                                        total_keyword_matches += 1
                
                if matches:
                    st.success(f"✅ Found **{total_keyword_matches} keyword matches** across {len(matches)} pages")
                else:
                    st.info("No matches")
            
            st.markdown("---")
            
            # Display ALL images - CONTINUOUS FLOW
            for i, img_path in enumerate(images):
                page = i + 1
                
                # Display image
                try:
                    if i in matches and query.strip():
                        # Get OCR boxes for highlighting
                        _, boxes = get_ocr(img_path)
                        
                        if boxes:
                            # Highlight
                            highlighted, count = draw_highlights(img_path, boxes, query)
                            st.image(highlighted, use_container_width=True)
                        else:
                            # No boxes, show original
                            img = load_image_safe(img_path)
                            st.image(img, use_container_width=True)
                    else:
                        # Show original - INSTANT with safe loading
                        img = load_image_safe(img_path)
                        st.image(img, use_container_width=True)
                
                except Exception as e:
                    st.error(f"Error loading image: {e}")
                    st.caption("⚠️ Image may be corrupted or too large")
        else:
            st.info("👈 Select a case")

st.caption("💡 First search takes a few seconds (OCR), then instant forever (cached) • Large images auto-resized for safety")