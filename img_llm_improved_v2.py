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
    
    # Create OCR cache table
    cur.execute("""
    CREATE TABLE IF NOT EXISTS ocr_cache (
        filepath TEXT PRIMARY KEY,
        ocr_text TEXT,
        ocr_boxes TEXT
    )
    """)
    
    # Create or migrate analysis cache table
    try:
        # Try to create new table
        cur.execute("""
        CREATE TABLE IF NOT EXISTS analysis_cache (
            case_id TEXT PRIMARY KEY,
            analysis_result TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
    except sqlite3.OperationalError:
        pass
    
    # Check if table exists but has old schema
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='analysis_cache'")
    if cur.fetchone():
        # Check columns
        cur.execute("PRAGMA table_info(analysis_cache)")
        columns = [row[1] for row in cur.fetchall()]
        
        # If case_id doesn't exist, recreate table
        if 'case_id' not in columns:
            cur.execute("DROP TABLE IF EXISTS analysis_cache")
            cur.execute("""
            CREATE TABLE analysis_cache (
                case_id TEXT PRIMARY KEY,
                analysis_result TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
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

# ---------------- Smart Text Processing for Speed ----------------
def clean_ocr_text(text: str) -> str:
    """Remove OCR noise and artifacts for cleaner, shorter text"""
    import re
    
    # Remove excessive whitespace
    text = ' '.join(text.split())
    
    # Remove common OCR artifacts
    text = re.sub(r'[^\w\s\.,;:!?\-\(\)\'\"]+', ' ', text)
    
    # Remove single characters (OCR noise)
    text = re.sub(r'\b[A-Z]\b', '', text)
    
    # Remove isolated numbers (page numbers, artifacts)
    text = re.sub(r'\b\d{1,2}\b(?!\d)', '', text)
    
    # Normalize whitespace again
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def smart_summarize_text(text: str, max_tokens: int = 2000) -> str:
    """Intelligently reduce text to ~max_tokens while keeping key info"""
    # Rough estimate: 1 token ≈ 4 characters
    max_chars = max_tokens * 4
    
    if len(text) <= max_chars:
        return text
    
    # Strategy: Keep beginning (30%), middle sample (20%), end (30%)
    # Skip the boring middle repetitive parts
    part1_size = int(max_chars * 0.35)
    part2_size = int(max_chars * 0.20)
    part3_size = max_chars - part1_size - part2_size
    
    # Beginning (usually has important context)
    part1 = text[:part1_size]
    
    # Middle sample (for variety)
    middle_start = len(text) // 2 - part2_size // 2
    part2 = text[middle_start:middle_start + part2_size]
    
    # End (often has conclusions/signatures)
    part3 = text[-part3_size:]
    
    return f"{part1}... {part2}... {part3}"

# ---------------- AI Analysis ----------------
def analyze_with_llm(combined_text: str) -> Dict:
    """Use local LLM to analyze document - OPTIMIZED"""
    if not HAS_OLLAMA:
        return {
            "overview": "AI analysis unavailable - Ollama not installed",
            "key_points": ["Install: pip install ollama", "Then: ollama pull llama3.2"],
            "notable_items": []
        }
    
    try:
        # Clean and compress text for speed
        cleaned = clean_ocr_text(combined_text)
        compressed = smart_summarize_text(cleaned, max_tokens=2000)
        
        # Much shorter, focused prompt
        prompt = f"""Summarize in 3 parts:
OVERVIEW: Main topic (1 sentence)
KEY POINTS: (3-5 bullets)
NOTABLE: Important items

Text: {compressed}"""
        
        # Try fastest models first
        models = ['llama3.2:latest', 'phi3.5:latest', 'gemma2:2b', 'gemma:2b']
        
        for model in models:
            try:
                response = ollama.chat(
                    model=model,
                    messages=[{'role': 'user', 'content': prompt}],
                    options={
                        'temperature': 0.2,
                        'num_predict': 256,  # Limit output length
                    }
                )
                
                return parse_llm_response(response['message']['content'])
            except Exception:
                continue
        
        return {
            "overview": "No Ollama models responded",
            "key_points": ["Try: ollama pull llama3.2"],
            "notable_items": []
        }
    except Exception as e:
        return {
            "overview": f"Analysis error: {str(e)}",
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
    """Analyze all pages with smart caching - FAST!"""
    import hashlib
    
    # Create case ID from image paths
    case_id = hashlib.md5('|'.join(sorted(images)).encode()).hexdigest()
    
    # Check analysis cache first (instant if exists!)
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT analysis_result FROM analysis_cache 
        WHERE case_id = ? AND datetime(created_at) > datetime('now', '-7 days')
    """, (case_id,))
    cached = cur.fetchone()
    
    if cached:
        try:
            return json.loads(cached[0])
        except:
            pass
    
    # Gather OCR text (uses cache - fast!)
    all_text = []
    for img_path in images:
        text, _ = get_ocr_from_cache(img_path)
        if not text:
            text, _ = get_ocr(img_path)
        if text.strip():
            all_text.append(text)
    
    combined = " ".join(all_text)
    
    if not combined.strip():
        result = {
            "overview": "No text found in documents",
            "key_points": ["OCR may have failed", "Try better scan quality"],
            "notable_items": [],
            "stats": {
                "total_pages": len(images),
                "pages_with_text": 0,
                "total_words": 0
            }
        }
    else:
        # Run LLM analysis (now 5-10x faster!)
        analysis = analyze_with_llm(combined)
        result = analysis.copy()
        result["stats"] = {
            "total_pages": len(images),
            "pages_with_text": len(all_text),
            "total_words": len(combined.split())
        }
    
    # Cache the result (lasts 7 days)
    try:
        cur.execute("""
            INSERT OR REPLACE INTO analysis_cache (case_id, analysis_result)
            VALUES (?, ?)
        """, (case_id, json.dumps(result)))
        conn.commit()
    except:
        pass
    
    return result

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

# ---------------- Q&A Function ----------------
def ask_document_question(images: List[str], question: str, case_id: str) -> str:
    """Answer questions about the document using LLM"""
    if not HAS_OLLAMA:
        return "❌ Ollama not installed. Install with: pip install ollama"
    
    # Get all document text (from cache)
    all_text = []
    for img_path in images:
        text, _ = get_ocr_from_cache(img_path)
        if not text:
            text, _ = get_ocr(img_path)
        if text.strip():
            all_text.append(text)
    
    combined = " ".join(all_text)
    
    if not combined.strip():
        return "❌ No text found in documents. Try searching keywords first to build OCR cache."
    
    # Clean and compress text for speed
    cleaned = clean_ocr_text(combined)
    compressed = smart_summarize_text(cleaned, max_tokens=2500)
    
    # Create focused prompt
    prompt = f"""Based on this document, answer the question concisely.

Document: {compressed}

Question: {question}

Answer (be brief and specific):"""
    
    try:
        # Try fastest models
        models = ['llama3.2:latest', 'phi3.5:latest', 'gemma2:2b']
        
        for model in models:
            try:
                response = ollama.chat(
                    model=model,
                    messages=[{'role': 'user', 'content': prompt}],
                    options={
                        'temperature': 0.2,
                        'num_predict': 150,
                    }
                )
                
                answer = response['message']['content'].strip()
                return answer if answer else "I couldn't find a clear answer in the document."
                
            except Exception:
                continue
        
        return "⚠️ No Ollama models responded. Try: ollama pull llama3.2"
        
    except Exception as e:
        return f"❌ Error: {str(e)}"

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

st.title("📄 Document Viewer")
st.caption("Professional document viewing • OCR search • AI-powered analysis")

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
        conn.execute("DELETE FROM analysis_cache")
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
            
            st.markdown("<div style='margin-top: 1rem;'></div>", unsafe_allow_html=True)
            
            # Manual AI Analysis toggle button
            analysis_visible = st.session_state.get(f"show_analysis_{selected}", False)
            
            button_label = "📊 View AI Summary" if not analysis_visible else "✕ Hide AI Summary"
            button_type = "secondary" if not analysis_visible else "primary"
            
            if st.button(button_label, width="stretch", type=button_type, help="Toggle AI analysis panel"):
                st.session_state[f"show_analysis_{selected}"] = not analysis_visible
                # Start processing if not done yet
                if not analysis_visible and f"processing_{selected}" not in st.session_state:
                    st.session_state[f"processing_{selected}"] = True
                st.rerun()
            
            # # Show processing status indicator
            # if st.session_state.get(f"processing_{selected}", False) and f"analysis_data_{selected}" not in st.session_state:
            #     st.markdown("""
            #         <div style='background: #fef3c7; 
            #                     border-left: 3px solid #f59e0b; 
            #                     padding: 0.5rem 0.75rem; 
            #                     border-radius: 4px;
            #                     margin-top: 0.5rem;
            #                     font-size: 0.8rem;
            #                     color: #92400e;'>
            #             🤖 AI analyzing in background...
            #         </div>
            #     """, unsafe_allow_html=True)
    
    with col2:
        if selected:
            st.markdown(f"### 📋 {selected}")
            
            images = cases[selected]
            
            # Search box - ALWAYS AT TOP
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
            
            # AI Analysis card - Only shows when user clicks button
            show_analysis_card = st.session_state.get(f"show_analysis_{selected}", False)
            analysis_key = f"analysis_data_{selected}"
            processing_key = f"processing_{selected}"
            
            if show_analysis_card:
                analysis = st.session_state.get(analysis_key)
                is_processing = st.session_state.get(processing_key, False) and not analysis
                
                if is_processing:
                    # Show processing state
                    st.markdown("""
                        <div style='background: #f8fafc; 
                                    border: 1px solid #e2e8f0; 
                                    border-left: 3px solid #6366f1;
                                    padding: 1rem 1.25rem; 
                                    border-radius: 8px;
                                    margin-bottom: 1.5rem;'>
                            <div style='display: flex; align-items: center; gap: 0.75rem;'>
                                <div class="spinner" style='width: 20px; height: 20px; border: 3px solid #e2e8f0; border-top-color: #6366f1; border-radius: 50%; animation: spin 1s linear infinite;'></div>
                                <div>
                                    <div style='font-weight: 600; color: #1e293b; font-size: 0.95rem;'>AI Analysis Running...</div>
                                    <div style='color: #64748b; font-size: 0.85rem; margin-top: 0.25rem;'>Processing document content. This may take 5-10 seconds.</div>
                                </div>
                            </div>
                        </div>
                        <style>
                        @keyframes spin {
                            to { transform: rotate(360deg); }
                        }
                        </style>
                    """, unsafe_allow_html=True)
                    
                elif analysis:
                    # Show results
                    st.markdown("""
                        <div style='background: #f8fafc;
                                    border: 1px solid #e2e8f0;
                                    border-radius: 8px;
                                    padding: 1.25rem;
                                    margin-bottom: 1.5rem;'>
                            <div style='display: flex; align-items: center; gap: 0.75rem; margin-bottom: 1rem;'>
                                <span style='font-size: 1.25rem;'>🤖</span>
                                <span style='font-weight: 600; color: #1e293b; font-size: 1rem;'>AI Document Analysis</span>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    # Stats
                    col_a, col_b = st.columns(2)
                    stats = analysis.get('stats', {})
              
                    # Detailed breakdown
                    col_left, col_right = st.columns(2)
                    
                    with col_left:
                        if analysis.get('key_points'):
                            st.markdown("**Key Points**")
                            for point in analysis['key_points']:
                                st.markdown(f"<div style='margin-bottom: 0.5rem; color: #475569;'>• {point}</div>", unsafe_allow_html=True)
                    
                    with col_right:
                        if analysis.get('notable_items'):
                            st.markdown("**Notable Items**")
                            for item in analysis['notable_items']:
                                st.markdown(f"<div style='margin-bottom: 0.5rem; color: #475569;'>• {item}</div>", unsafe_allow_html=True)
                    
                    st.divider()
                    
                    # Q&A Section
                    st.markdown("**💬 Ask Questions About This Document**")
                    
                    # Initialize chat history
                    chat_key = f"chat_{selected}"
                    if chat_key not in st.session_state:
                        st.session_state[chat_key] = []
                    
                    # Show chat history
                    if st.session_state[chat_key]:
                        st.markdown("""
                            <div style='background: #f9fafb; 
                                        border: 1px solid #e5e7eb; 
                                        border-radius: 6px; 
                                        padding: 0.75rem;
                                        margin-bottom: 1rem;
                                        max-height: 300px;
                                        overflow-y: auto;'>
                        """, unsafe_allow_html=True)
                        
                        for i, msg in enumerate(st.session_state[chat_key]):
                            if msg['role'] == 'user':
                                st.markdown(f"""
                                    <div style='background: #eff6ff; 
                                                border-left: 3px solid #3b82f6;
                                                padding: 0.5rem 0.75rem; 
                                                margin-bottom: 0.5rem;
                                                border-radius: 4px;'>
                                        <strong>You:</strong> {msg['content']}
                                    </div>
                                """, unsafe_allow_html=True)
                            else:
                                st.markdown(f"""
                                    <div style='background: white; 
                                                border-left: 3px solid #10b981;
                                                padding: 0.5rem 0.75rem; 
                                                margin-bottom: 0.5rem;
                                                border-radius: 4px;'>
                                        <strong>AI:</strong> {msg['content']}
                                    </div>
                                """, unsafe_allow_html=True)
                        
                        st.markdown("</div>", unsafe_allow_html=True)
                        
                        # Clear chat button
                        if st.button("🗑️ Clear conversation", key=f"clear_chat_{selected}", type="secondary"):
                            st.session_state[chat_key] = []
                            st.rerun()
                    
                    # Question input
                    question = st.text_input(
                        "Your question:",
                        placeholder="e.g., What is the policy number? When does coverage end?",
                        key=f"question_{selected}",
                        label_visibility="collapsed"
                    )
                    
                    if question and question.strip():
                        if st.button("Ask", key=f"ask_{selected}", type="primary"):
                            # Add user question to chat
                            st.session_state[chat_key].append({"role": "user", "content": question})
                            
                            # Get answer from LLM
                            with st.spinner("Thinking..."):
                                answer = ask_document_question(images, question, selected)
                                st.session_state[chat_key].append({"role": "assistant", "content": answer})
                            
                            st.rerun()
                    
                    st.divider()
            
            # Display images - ALWAYS render these first
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
            
            # Process AI analysis AFTER images (only if button was clicked)
            processing_key = f"processing_{selected}"
            analysis_key = f"analysis_data_{selected}"
            
            if (st.session_state.get(processing_key, False) and 
                analysis_key not in st.session_state):
                # Run analysis now
                with st.spinner(""):  # Silent
                    analysis = analyze_case(images)
                    st.session_state[analysis_key] = analysis
                    st.session_state[processing_key] = False
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