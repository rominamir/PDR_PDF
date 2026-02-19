"""
ultra_fast_viewer.py - OPTIMIZED

Clean, fast document viewer with optional AI analysis

Features:
1. Type case number directly
2. Enhanced OCR for scanned documents
3. Manual AI analysis (on-demand only)
4. Clean, professional interface

Usage:
  pip install opencv-python ollama  # Optional for AI
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

# Try to import Ollama for AI
try:
    import ollama
    HAS_OLLAMA = True
except ImportError:
    HAS_OLLAMA = False

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

# ---------------- Image Preprocessing ----------------
def preprocess_for_ocr(img: Image.Image) -> Image.Image:
    """Enhance image quality for better OCR"""
    if not HAS_CV2:
        return img.convert('L')
    
    try:
        img_array = np.array(img.convert('RGB'))
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(denoised)
        binary = cv2.adaptiveThreshold(
            enhanced, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
        return Image.fromarray(binary)
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

# ---------------- OCR ----------------
def run_ocr_with_boxes(image_path: str) -> Tuple[str, List[Dict]]:
    """Run OCR with preprocessing"""
    try:
        import pytesseract
        
        img = load_image_safe(image_path)
        preprocessed = preprocess_for_ocr(img)
        
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
        
        # Try original if preprocessing didn't work well
        if len(text_parts) < 10:
            data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT, config=custom_config)
            words2, text_parts2 = [], []
            
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
            
            if len(text_parts2) > len(text_parts):
                words = words2
                full_text = ' '.join(text_parts2)
        
        return full_text, words
    except Exception:
        return "", []

def get_ocr(filepath: str) -> Tuple[str, List[Dict]]:
    cached_text, cached_boxes = get_ocr_from_cache(filepath)
    if cached_text:
        return cached_text, cached_boxes
    text, boxes = run_ocr_with_boxes(filepath)
    if text:
        save_ocr_to_cache(filepath, text, boxes)
    return text, boxes

# ---------------- AI Analysis ----------------
def analyze_with_llm(combined_text: str) -> Dict:
    """Use local LLM to analyze document"""
    if not HAS_OLLAMA:
        return {
            "error": "Install ollama package: pip install ollama",
            "overview": "AI analysis unavailable",
            "key_points": [],
            "notable_items": []
        }
    
    try:
        max_chars = 12000
        if len(combined_text) > max_chars:
            combined_text = combined_text[:max_chars] + "...[truncated]"
        
        prompt = f"""Analyze this document and provide:
1. Brief overview (1-2 sentences)
2. 3-5 key points
3. Notable dates, numbers, or entities

Document:
{combined_text}

Format:
OVERVIEW: [overview]
KEY POINTS:
- [point 1]
- [point 2]
NOTABLE ITEMS:
- [item 1]"""
        
        models = ['llama3.2:latest', 'phi3.5:latest', 'gemma:latest', 'mistral:latest']
        
        for model in models:
            try:
                response = ollama.chat(
                    model=model,
                    messages=[{'role': 'user', 'content': prompt}],
                    options={'temperature': 0.3}
                )
                
                return parse_llm_response(response['message']['content'])
            except Exception:
                continue
        
        return {
            "overview": "No Ollama models available",
            "key_points": ["Install a model: ollama pull llama3.2"],
            "notable_items": []
        }
    except Exception as e:
        return {
            "overview": f"AI analysis failed: {str(e)}",
            "key_points": [],
            "notable_items": []
        }

def parse_llm_response(text: str) -> Dict:
    """Parse LLM response"""
    lines = text.strip().split('\n')
    overview = ""
    key_points = []
    notable_items = []
    current = None
    
    for line in lines:
        line = line.strip()
        if line.startswith('OVERVIEW:'):
            overview = line.replace('OVERVIEW:', '').strip()
            current = 'overview'
        elif line.startswith('KEY POINTS:'):
            current = 'key_points'
        elif line.startswith('NOTABLE ITEMS:'):
            current = 'notable_items'
        elif line.startswith('- ') and current == 'key_points':
            key_points.append(line[2:].strip())
        elif line.startswith('- ') and current == 'notable_items':
            notable_items.append(line[2:].strip())
    
    if not overview:
        sentences = text.split('.')
        overview = '. '.join(sentences[:2]).strip() + '.'
    
    return {
        "overview": overview or "Analysis complete",
        "key_points": key_points or ["See document for details"],
        "notable_items": notable_items
    }

def analyze_case(images: List[str]) -> Dict:
    """Analyze all pages in case - FAST version using cached OCR"""
    all_text = []
    
    # Use cached OCR data (super fast!)
    for img_path in images:
        # Check cache first - don't re-run OCR
        text, _ = get_ocr_from_cache(img_path)
        
        # If not cached, get it (will cache for future)
        if not text:
            text, _ = get_ocr(img_path)
        
        if text.strip():
            all_text.append(text)
    
    combined = " ".join(all_text)
    
    if not combined.strip():
        return {
            "overview": "No text found in documents",
            "key_points": ["OCR may have failed", "Try better quality scans"],
            "notable_items": [],
            "stats": {
                "total_pages": len(images),
                "pages_with_text": 0,
                "total_words": 0
            }
        }
    
    analysis = analyze_with_llm(combined)
    analysis["stats"] = {
        "total_pages": len(images),
        "pages_with_text": len(all_text),
        "total_words": len(combined.split())
    }
    
    return analysis

def analyze_case_fast(images: List[str]) -> Dict:
    """Ultra-fast analysis - only uses cached OCR, doesn't run new OCR"""
    all_text = []
    
    # Only use already cached data (instant!)
    for img_path in images:
        text, _ = get_ocr_from_cache(img_path)
        if text and text.strip():
            all_text.append(text)
    
    combined = " ".join(all_text)
    
    if not combined.strip():
        return {
            "overview": "No OCR data cached yet. Search documents first to build cache.",
            "key_points": ["Try searching for a keyword", "This will cache OCR for all pages"],
            "notable_items": [],
            "stats": {
                "total_pages": len(images),
                "pages_with_text": 0,
                "total_words": 0
            }
        }
    
    # LLM analysis
    analysis = analyze_with_llm(combined)
    analysis["stats"] = {
        "total_pages": len(images),
        "pages_with_text": len(all_text),
        "total_words": len(combined.split())
    }
    
    return analysis

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
st.set_page_config(page_title="Document Viewer", layout="wide", initial_sidebar_state="collapsed")

# Custom CSS for sleek, professional look
st.markdown("""
<style>
    /* Main layout improvements */
    .main > div {
        padding-top: 1.5rem;
    }
    
    /* Typography */
    h1 {
        font-size: 1.75rem !important;
        font-weight: 600 !important;
        margin-bottom: 0.25rem !important;
        color: #1a1a1a;
    }
    
    h3 {
        font-size: 1.25rem !important;
        font-weight: 500 !important;
        color: #2d3748;
    }
    
    /* Search input styling */
    .stTextInput > div > div > input {
        font-size: 0.95rem;
        border-radius: 8px;
        border: 1.5px solid #e2e8f0;
        padding: 0.625rem 0.875rem;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #4f46e5;
        box-shadow: 0 0 0 3px rgba(79, 70, 229, 0.1);
    }
    
    /* Button styling */
    .stButton > button {
        border-radius: 6px;
        font-weight: 500;
        transition: all 0.2s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
    }
    
    /* Selectbox styling */
    .stSelectbox > div > div {
        border-radius: 6px;
    }
    
    /* Metric cards */
    [data-testid="stMetricValue"] {
        font-size: 1.25rem;
        font-weight: 600;
    }
    
    /* Divider */
    hr {
        margin: 1.5rem 0;
        border-color: #e2e8f0;
    }
    
    /* Remove extra padding */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Success/Info messages */
    .stSuccess, .stInfo {
        border-radius: 6px;
        font-size: 0.875rem;
    }
</style>
""", unsafe_allow_html=True)

st.title("📄 PDR Document Viewer")
st.caption("PDR document viewing • OCR search • AI-powered analysis")

# Sidebar - minimal
with st.sidebar:
    st.header("Settings")
    DATA_ROOT = st.text_input("Documents Folder", DATA_ROOT_DEFAULT)
    
    st.divider()
    
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM ocr_cache WHERE ocr_text != ''")
    cached = cur.fetchone()[0]
    
    st.metric("📦 Cached Pages", cached)
    if st.button("Clear Cache", width="stretch"):
        conn.execute("DELETE FROM ocr_cache")
        conn.commit()
        st.rerun()

# Main
st.divider()

cases = find_cases(DATA_ROOT)

if not cases:
    st.warning(f"📂 No cases found in: {DATA_ROOT}")
    st.info("Check the folder path in the sidebar")
else:
    # Two column layout
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.markdown("### 📁 Cases")
        
        # Type case number
        typed_case = st.text_input(
            "Case Number",
            placeholder="Type case number...",
            label_visibility="collapsed"
        )
        
        # OR select from dropdown
        st.caption("or select from list:")
        selected_list = st.selectbox(
            "Select case",
            [""] + sorted(cases.keys()),
            format_func=lambda x: "-- Select --" if x == "" else f"{x} ({len(cases[x])} pages)",
            label_visibility="collapsed"
        )
        
        # Determine selection
        selected = None
        if typed_case.strip():
            if typed_case in cases:
                selected = typed_case
            else:
                st.error(f"Case not found")
                similar = [c for c in cases if typed_case.lower() in c.lower()]
                if similar:
                    st.caption(f"💡 Similar: {similar[0]}")
        elif selected_list:
            selected = selected_list
        
        if selected:
            st.success(f"✓ {len(cases[selected])} pages")
            
            # AI Analysis button - modern design
            st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)
            
            # Custom styled button
            analyze_button = st.button(
                "🤖 Analyze with AI",
                width="stretch",
                type="secondary",
                help="Get AI-powered summary and insights"
            )
            
            if analyze_button:
                if f"analyze_{selected}" not in st.session_state:
                    st.session_state[f"analyze_{selected}"] = False
                st.session_state[f"analyze_{selected}"] = not st.session_state[f"analyze_{selected}"]
                st.rerun()
            
            # Show hint if analysis is active
            if st.session_state.get(f"analyze_{selected}", False):
                st.markdown("""
                    <div style='background: #f0f9ff; 
                                border-left: 3px solid #3b82f6; 
                                padding: 0.5rem 0.75rem; 
                                border-radius: 4px;
                                margin-top: 0.5rem;
                                font-size: 0.8rem;
                                color: #1e40af;'>
                        ✓ AI analysis active
                    </div>
                """, unsafe_allow_html=True)
    
    with col2:
        if selected:
            st.markdown(f"### 📋 {selected}")
            
            images = cases[selected]
            
            # Search box - ALWAYS AT TOP (never hidden)
            query = st.text_input(
                "🔍 Search documents",
                placeholder="Enter keyword to highlight...",
                label_visibility="collapsed",
                key=f"search_{selected}"
            )
            
            # Search logic
            matches = set()
            total_matches = 0
            
            if query.strip():
                with st.spinner("Searching..."):
                    for i, img_path in enumerate(images):
                        text, boxes = get_ocr(img_path)
                        if text and query.lower() in text.lower():
                            matches.add(i)
                            if boxes:
                                for w in boxes:
                                    if query.lower() in w['text'].lower():
                                        total_matches += 1
                
                if matches:
                    st.success(f"✓ {total_matches} matches in {len(matches)} pages")
                else:
                    st.info("No matches found")
            
            st.divider()
            
            # AI Analysis card - Sleek minimal design
            show_analysis = st.session_state.get(f"analyze_{selected}", False)
            analysis_key = f"analysis_data_{selected}"
            
            if show_analysis:
                analysis = st.session_state.get(analysis_key)
                
                if not analysis:
                    # Processing state - minimal design
                    st.markdown("""
                        <div style='background: #f8fafc; 
                                    border: 1px solid #e2e8f0; 
                                    border-left: 3px solid #6366f1;
                                    padding: 1rem 1.25rem; 
                                    border-radius: 8px;
                                    margin-bottom: 1.5rem;'>
                            <div style='display: flex; align-items: center; gap: 0.75rem;'>
                                <span style='font-size: 1.25rem;'>🤖</span>
                                <div>
                                    <div style='font-weight: 600; color: #1e293b; font-size: 0.95rem;'>AI Analysis</div>
                                    <div style='color: #64748b; font-size: 0.85rem; margin-top: 0.25rem;'>Processing in background...</div>
                                </div>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
                    
                else:
                    # Results - clean card design
                    st.markdown("""
                        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                                    padding: 0.15rem; 
                                    border-radius: 10px;
                                    margin-bottom: 1.5rem;'>
                            <div style='background: white; 
                                        padding: 1.25rem; 
                                        border-radius: 9px;'>
                                <div style='display: flex; align-items: center; justify-content: space-between; margin-bottom: 1rem;'>
                                    <div style='display: flex; align-items: center; gap: 0.75rem;'>
                                        <span style='font-size: 1.25rem;'>🤖</span>
                                        <span style='font-weight: 600; color: #1e293b; font-size: 1rem;'>AI Document Analysis</span>
                                    </div>
                                </div>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    # Stats in clean grid
                    col_a, col_b, col_c = st.columns([2, 2, 1])
                    stats = analysis.get('stats', {})
                    with col_a:
                        st.metric(
                            "Pages Analyzed", 
                            f"{stats.get('pages_with_text', 0)}/{stats.get('total_pages', 0)}",
                            help="Pages with readable text"
                        )
                    with col_b:
                        st.metric(
                            "Total Words", 
                            f"{stats.get('total_words', 0):,}",
                            help="Words detected across all pages"
                        )
                    with col_c:
                        if st.button("✕ Close", key="close_analysis", help="Close analysis", type="secondary"):
                            del st.session_state[f"analyze_{selected}"]
                            if analysis_key in st.session_state:
                                del st.session_state[analysis_key]
                            st.rerun()
                    
                    # Collapsible details - clean styling
                    with st.expander("📊 View Full Analysis", expanded=False):
                        if analysis.get('overview'):
                            st.markdown("##### Overview")
                            st.markdown(f"<p style='color: #475569; line-height: 1.6;'>{analysis['overview']}</p>", unsafe_allow_html=True)
                            st.divider()
                        
                        col_left, col_right = st.columns(2)
                        
                        with col_left:
                            if analysis.get('key_points'):
                                st.markdown("##### Key Points")
                                for point in analysis['key_points']:
                                    st.markdown(f"<div style='margin-bottom: 0.5rem; color: #475569;'>• {point}</div>", unsafe_allow_html=True)
                        
                        with col_right:
                            if analysis.get('notable_items'):
                                st.markdown("##### Notable Items")
                                for item in analysis['notable_items']:
                                    st.markdown(f"<div style='margin-bottom: 0.5rem; color: #475569;'>• {item}</div>", unsafe_allow_html=True)
                    
                    st.divider()
            
            # Display images - ALWAYS render these first, before any processing
            for i, img_path in enumerate(images):
                try:
                    if i in matches and query.strip():
                        _, boxes = get_ocr(img_path)
                        if boxes:
                            highlighted, _ = draw_highlights(img_path, boxes, query)
                            st.image(highlighted, width="stretch")
                        else:
                            st.image(load_image_safe(img_path), width="stretch")
                    else:
                        st.image(load_image_safe(img_path), width="stretch")
                except Exception as e:
                    st.error(f"Error loading image: {e}")
            
            # Process AI analysis AFTER images are rendered
            if show_analysis and not analysis:
                # Run analysis now (after images displayed)
                analysis = analyze_case_fast(images)
                st.session_state[analysis_key] = analysis
                st.rerun()
        else:
            st.info("👈 Select or type a case number to begin")

# Footer
st.divider()
st.markdown("""
    <div style='text-align: center; color: #64748b; font-size: 0.85rem; padding: 1rem 0;'>
        💡 <strong>Tips:</strong> Search keywords are highlighted in yellow • AI analysis uses cached OCR for speed
    </div>
""", unsafe_allow_html=True)