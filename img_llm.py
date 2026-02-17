"""
img_llm.py

Streamlit web app for:
 - browsing "cases" (folders with images)
 - semantic text search over PNG/JPG/TIFF images (CLIP embeddings + FAISS)
 - building / loading FAISS index
 - a training skeleton function if you want to fine-tune a VLM locally.

Usage:
  1) Install dependencies:
     pip install streamlit pillow sqlalchemy transformers torch torchvision tqdm faiss-cpu paddlepaddle paddleocr
     # or faiss-gpu if you have GPU + CUDA and want faster indexing/search

  2) Edit DATA_ROOT (default "./data") or supply via UI.

  3) Build index from UI (Discover images -> Build index) OR run:
     python img_llm.py --index

  4) Run:
     streamlit run img_llm.py
"""
import os
import io
import json
import sqlite3
import argparse
from typing import List, Tuple

# Set environment variable to bypass PaddleOCR connectivity check
os.environ['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK'] = 'True'

from PIL import Image
import numpy as np
import streamlit as st
from tqdm import tqdm

import torch
from transformers import CLIPProcessor, CLIPModel

# FAISS optional
try:
    import faiss
except Exception:
    faiss = None

# PaddleOCR optional
try:
    from paddleocr import PaddleOCR
except Exception:
    PaddleOCR = None

# EasyOCR as fallback
try:
    import easyocr
except Exception:
    easyocr = None

# ---------------- Config ----------------
DATA_ROOT_DEFAULT = "/hspshare/converted_images"
DB_PATH = "cases_streamlit.db"
FAISS_INDEX_PATH = "faiss_streamlit.index"
IDMAP_PATH = "faiss_streamlit_idmap.json"
EMBED_DIM = 512  # default for CLIP ViT-B/32; change if using another model
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
SUPPORTED_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}

# ---------------- DB helpers ----------------
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS images (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        case_number TEXT,
        filename TEXT,
        filepath TEXT,
        ocr_text TEXT,
        processed INTEGER DEFAULT 0
    )
    """)
    conn.commit()
    conn.close()

def upsert_image(case_number: str, filename: str, filepath: str, ocr_text: str = None):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT id, ocr_text FROM images WHERE filepath = ?", (filepath,))
    existing = cur.fetchone()
    
    if existing is None:
        # Insert new record
        cur.execute("INSERT INTO images (case_number, filename, filepath, ocr_text) VALUES (?, ?, ?, ?)",
                    (case_number, filename, filepath, ocr_text))
        st.write(f"   ✅ Added: {filename} with OCR: {'Yes' if ocr_text else 'No'}")
    else:
        # Update existing record, especially if OCR text is provided or missing
        if ocr_text is not None:
            cur.execute("UPDATE images SET ocr_text = ?, case_number = ?, filename = ? WHERE filepath = ?", 
                       (ocr_text, case_number, filename, filepath))
            st.write(f"   🔄 Updated OCR for: {filename}")
        elif not existing[1]:  # No existing OCR text
            cur.execute("UPDATE images SET case_number = ?, filename = ? WHERE filepath = ?", 
                       (case_number, filename, filepath))
            st.write(f"   🔄 Updated: {filename} (no OCR)")
    
    conn.commit()
    conn.close()

def list_open_cases() -> List[str]:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT DISTINCT case_number FROM images ORDER BY case_number")
    rows = [r[0] for r in cur.fetchall()]
    conn.close()
    return rows

def list_images_for_case(case_number: str) -> List[dict]:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute("SELECT * FROM images WHERE case_number = ? ORDER BY filename", (case_number,))
    rows = cur.fetchall()
    conn.close()
    return [{"filename": r["filename"], "filepath": r["filepath"]} for r in rows]

def gather_all_filepaths_from_db() -> List[str]:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT filepath FROM images ORDER BY id")
    rows = [r[0] for r in cur.fetchall()]
    conn.close()
    return rows

# ---------------- OCR Processing ----------------
def extract_text_from_image(image_path: str) -> str:
    """Extract text from image using PaddleOCR or EasyOCR as fallback"""
    
    # Check if image exists and is readable first
    if not os.path.exists(image_path):
        st.error(f"❌ Image not found: {image_path}")
        return ""
        
    try:
        test_img = Image.open(image_path)
        test_img.close()
    except Exception as img_error:
        st.error(f"❌ Cannot read image {image_path}: {img_error}")
        return ""
    
    # Try PaddleOCR first
    if PaddleOCR is not None:
        try:
            st.write(f"   🔍 Running PaddleOCR on: {os.path.basename(image_path)}")
            
            # Initialize OCR with updated parameters for newer PaddleOCR versions
            try:
                ocr = PaddleOCR(use_textline_orientation=True, lang='en')
            except Exception:
                # Fallback for older PaddleOCR versions
                ocr = PaddleOCR(lang='en')
            
            result = ocr.ocr(image_path, cls=True)
            
            if not result or not result[0]:
                st.warning(f"   ⚠️ PaddleOCR found no text in {os.path.basename(image_path)}")
            else:
                extracted_text = []
                total_lines = len(result[0])
                
                for line_idx, line in enumerate(result[0]):
                    if len(line) > 1 and len(line[1]) > 1:
                        text = line[1][0]
                        confidence = line[1][1]
                        if confidence > 0.3:
                            extracted_text.append(text)
                            st.write(f"     • Line {line_idx+1}: '{text}' (conf: {confidence:.2f})")
                
                full_text = " ".join(extracted_text)
                
                if full_text.strip():
                    st.success(f"   ✅ PaddleOCR extracted {len(extracted_text)}/{total_lines} text lines")
                    st.write(f"   📝 Full text: {full_text[:200]}..." if len(full_text) > 200 else f"   📝 Full text: {full_text}")
                    return full_text
                else:
                    st.warning(f"   ⚠️ No high-confidence text found with PaddleOCR")
                    
        except Exception as paddle_error:
            st.warning(f"   ⚠️ PaddleOCR failed: {str(paddle_error)}")
            st.write("   Trying EasyOCR as fallback...")
    
    # Fallback to EasyOCR
    if easyocr is not None:
        try:
            st.write(f"   🔍 Running EasyOCR on: {os.path.basename(image_path)}")
            reader = easyocr.Reader(['en'])
            result = reader.readtext(image_path)
            
            extracted_text = []
            for detection in result:
                text = detection[1]
                confidence = detection[2]
                if confidence > 0.3:
                    extracted_text.append(text)
                    st.write(f"     • '{text}' (conf: {confidence:.2f})")
            
            full_text = " ".join(extracted_text)
            
            if full_text.strip():
                st.success(f"   ✅ EasyOCR extracted {len(extracted_text)} text elements")
                st.write(f"   📝 Full text: {full_text[:200]}..." if len(full_text) > 200 else f"   📝 Full text: {full_text}")
                return full_text
            else:
                st.warning(f"   ⚠️ No high-confidence text found with EasyOCR")
                
        except Exception as easy_error:
            st.error(f"   ❌ EasyOCR also failed: {str(easy_error)}")
    
    # If both failed
    if PaddleOCR is None and easyocr is None:
        st.error("❌ No OCR library available! Install with: pip install paddlepaddle paddleocr OR pip install easyocr")
    else:
        st.error(f"❌ All OCR methods failed for {os.path.basename(image_path)}")
    
    return ""

def process_ocr_for_all_images():
    """Process OCR for all images in database that don't have OCR text yet"""
    if PaddleOCR is None and easyocr is None:
        st.error("❌ No OCR library installed! Install with:")
        st.code("pip install paddlepaddle paddleocr")
        st.write("OR")
        st.code("pip install easyocr")
        return 0
        
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT filepath, filename, case_number FROM images WHERE ocr_text IS NULL OR ocr_text = ''")
    rows = cur.fetchall()
    conn.close()
    
    if not rows:
        st.info("✅ All images already have OCR text!")
        return 0
    
    st.write(f"🔍 Processing OCR for {len(rows)} images...")
    progress_bar = st.progress(0)
    processed = 0
    
    for i, (filepath, filename, case_number) in enumerate(rows):
        st.write(f"\n📄 Processing {i+1}/{len(rows)}: {filename}")
        
        if os.path.exists(filepath):
            ocr_text = extract_text_from_image(filepath)
            # Update with proper case info
            upsert_image(case_number, filename, filepath, ocr_text)
            processed += 1
        else:
            st.error(f"❌ File not found: {filepath}")
            
        progress_bar.progress((i + 1) / len(rows))
    
    st.success(f"✅ Processed OCR for {processed} images.")
    return processed

def search_images_by_text(query: str, case_filter: str = None) -> List[dict]:
    """Search images by OCR text content"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    
    sql = "SELECT * FROM images WHERE ocr_text LIKE ?"
    params = [f"%{query}%"]
    
    if case_filter:
        sql += " AND case_number = ?"
        params.append(case_filter)
    
    sql += " ORDER BY case_number, filename"
    
    cur.execute(sql, params)
    rows = cur.fetchall()
    conn.close()
    
    return [{
        "case_number": r["case_number"],
        "filename": r["filename"], 
        "filepath": r["filepath"],
        "ocr_text": r["ocr_text"]
    } for r in rows]

# ---------------- Discovery ----------------
def discover_and_register_images(data_root: str, with_ocr: bool = False):
    init_db()
    cnt = 0
    
    # Look for case directories directly in data_root
    if not os.path.exists(data_root):
        return cnt
        
    for case_dir in os.listdir(data_root):
        case_path = os.path.join(data_root, case_dir)
        if not os.path.isdir(case_path):
            continue
            
        case_number = case_dir
        st.write(f"Checking case: {case_number}")
        
        # Look for ALL PNG files in the directory
        png_files_found = []
        for f in os.listdir(case_path):
            if os.path.splitext(f)[1].lower() in SUPPORTED_EXT:
                png_files_found.append(f)
                
        st.write(f"Found {len(png_files_found)} image files: {png_files_found}")
        
        # Process all found image files
        for f in png_files_found:
            fp = os.path.join(case_path, f)
            ocr_text = None
            if with_ocr:
                st.write(f"Processing OCR for {f}...")
                ocr_text = extract_text_from_image(fp)
            upsert_image(case_number, f, fp, ocr_text)
            cnt += 1
            
    return cnt

# ---------------- Embedding & Index ----------------
class EmbedIndex:
    def __init__(self, model_name=CLIP_MODEL_NAME, device=DEVICE, dim=EMBED_DIM):
        self.device = device
        st.info(f"Loading CLIP model {model_name} on {device} (may take a moment)...")
        self.model = CLIPModel.from_pretrained(model_name).to(device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.dim = dim
        self.index = None
        self.id_map = []

    def encode_image(self, pil_image: Image.Image) -> np.ndarray:
        inputs = self.processor(images=pil_image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            image_emb = self.model.get_image_features(**inputs)
        emb = image_emb.cpu().numpy().astype(np.float32)
        emb /= (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-10)
        return emb[0]

    def encode_text(self, text: str) -> np.ndarray:
        inputs = self.processor(text=[text], return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            text_emb = self.model.get_text_features(**inputs)
        emb = text_emb.cpu().numpy().astype(np.float32)
        emb /= (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-10)
        return emb[0]

    def build_index(self, filepaths: List[str], progress_callback=None):
        if faiss is None:
            raise RuntimeError("faiss not installed. Install faiss-cpu or faiss-gpu.")
        n = len(filepaths)
        xb = np.zeros((n, self.dim), dtype=np.float32)
        self.id_map = []
        iterator = tqdm(enumerate(filepaths), total=n, desc="Embedding images")
        for i, fp in iterator:
            try:
                img = Image.open(fp).convert("RGB")
                emb = self.encode_image(img)
                xb[i] = emb
                self.id_map.append(fp)
                if progress_callback:
                    progress_callback(i+1, n)
            except Exception as e:
                st.warning(f"Failed to process {fp}: {e}")
        idx = faiss.IndexFlatIP(self.dim)
        idx.add(xb)
        self.index = idx

    def save_index(self, index_path=FAISS_INDEX_PATH, idmap_path=IDMAP_PATH):
        if self.index is None:
            raise RuntimeError("Index not built.")
        faiss.write_index(self.index, index_path)
        with open(idmap_path, "w") as f:
            json.dump(self.id_map, f)

    def load_index(self, index_path=FAISS_INDEX_PATH, idmap_path=IDMAP_PATH):
        if faiss is None:
            raise RuntimeError("faiss not installed.")
        if not os.path.exists(index_path) or not os.path.exists(idmap_path):
            raise FileNotFoundError("Index or idmap not found.")
        self.index = faiss.read_index(index_path)
        with open(idmap_path, "r") as f:
            self.id_map = json.load(f)

    def search_text(self, query: str, k=10):
        if self.index is None:
            raise RuntimeError("Index not loaded.")
        q_emb = self.encode_text(query).reshape(1, -1).astype(np.float32)
        D, I = self.index.search(q_emb, k)
        results = []
        for score, idx in zip(D[0], I[0]):
            if idx < 0 or idx >= len(self.id_map):
                continue
            results.append((self.id_map[idx], float(score)))
        return results

# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="VLM Case Image Search", layout="wide")
st.title("Case Browser + Semantic Image Search (Streamlit)")

# Sidebar: configuration & index controls
with st.sidebar:
    st.header("Configuration")
    DATA_ROOT = st.text_input("DATA_ROOT (folder with case subfolders)", DATA_ROOT_DEFAULT)
    st.write("Expect structure: DATA_ROOT/{casenumber}/{casenumber}_combined.png")
    st.write(f"Device: {DEVICE}")
    st.markdown("---")

    st.header("Index Controls")
    
    # Show OCR status
    if PaddleOCR is None and easyocr is None:
        st.error("❌ No OCR library installed")
        st.code("pip install paddlepaddle paddleocr")
        st.write("OR")
        st.code("pip install easyocr")
    elif PaddleOCR is not None:
        st.success("✅ PaddleOCR available")
    elif easyocr is not None:
        st.success("✅ EasyOCR available")
    
    with_ocr = st.checkbox("Process OCR during discovery", help="Extract text from images for keyword search")
    if st.button("Discover & register images"):
        with st.spinner("Discovering images..."):
            n = discover_and_register_images(DATA_ROOT, with_ocr=with_ocr)
        st.success(f"Discovered & registered {n} images (if any).")
    
    if st.button("Process OCR for existing images"):
        with st.spinner("Processing OCR..."):
            n = process_ocr_for_all_images()
        st.success(f"Processed OCR for {n} images.")
    
    st.write("After discovery, build or load FAISS index for semantic search.")
    build_index_btn = st.button("Build FAISS index (embed all images)")
    load_index_btn = st.button("Load FAISS index (from disk)")

    st.markdown("---")
    st.write("Model:")
    st.text(CLIP_MODEL_NAME)

# initialize DB if not present
init_db()

# persistent embedder via session_state
if "embedder" not in st.session_state:
    st.session_state.embedder = None
if "index_built" not in st.session_state:
    st.session_state.index_built = False

# Build index flow
if build_index_btn:
    filepaths = gather_all_filepaths_from_db()
    if not filepaths:
        st.error("No images found in DB. Run Discover & register images first.")
    else:
        st.session_state.embedder = EmbedIndex()
        progress_bar = st.progress(0)
        def progress_cb(done, total):
            progress_bar.progress(int(done/total*100))
        try:
            st.info(f"Embedding {len(filepaths)} images; this may take a while.")
            st.session_state.embedder.build_index(filepaths, progress_callback=progress_cb)
            st.session_state.embedder.save_index()
            st.session_state.index_built = True
            st.success("Index built and saved to disk.")
        except Exception as e:
            st.error(f"Index build failed: {e}")

# Load index flow
if load_index_btn:
    try:
        st.session_state.embedder = EmbedIndex()
        st.session_state.embedder.load_index()
        st.session_state.index_built = True
        st.success("Index loaded into memory.")
    except Exception as e:
        st.error(f"Failed to load index: {e}")

# Main: case selection and document viewer
cases = list_open_cases()
col1, col2 = st.columns([1, 3])

with col1:
    st.subheader("📁 Cases")
    if not cases:
        st.info("No cases found. Run 'Discover & register images' in the sidebar.")
    else:
        selected_case = st.selectbox("Select a case", options=[""] + cases)
        
        if selected_case:
            imgs = list_images_for_case(selected_case)
            st.write(f"📄 {len(imgs)} pages")
            
            # Search box for current case
            st.markdown("---")
            st.subheader("🔍 Search in Case")
            search_query = st.text_input("Search text in this case:", key="case_search")
            
            if search_query.strip():
                # Search within the current case only
                results = search_images_by_text(search_query, selected_case)
                st.write(f"**Found in {len(results)} pages**")
                
                # Debug: Show what OCR text exists for this case
                with st.expander("🔍 Debug: OCR Text in Database"):
                    conn = sqlite3.connect(DB_PATH)
                    conn.row_factory = sqlite3.Row
                    cur = conn.cursor()
                    cur.execute("SELECT filename, ocr_text FROM images WHERE case_number = ? ORDER BY filename", (selected_case,))
                    debug_rows = cur.fetchall()
                    conn.close()
                    
                    for row in debug_rows:
                        if row["ocr_text"]:
                            st.write(f"📄 {row['filename']}: {row['ocr_text'][:100]}...")
                        else:
                            st.write(f"📄 {row['filename']}: ❌ No OCR text")
                
                # Show quick navigation to results
                for i, result in enumerate(results):
                    page_num = next((idx+1 for idx, img in enumerate(imgs) if img['filepath'] == result['filepath']), 0)
                    if st.button(f"📄 Page {page_num}", key=f"nav_{i}"):
                        st.session_state.scroll_to_page = page_num

with col2:
    if selected_case:
        st.subheader(f"📋 Case: {selected_case}")
        
        imgs = list_images_for_case(selected_case)
        search_query = st.session_state.get('case_search', '') if 'case_search' in st.session_state else ""
        
        # Create a scrollable document view
        for i, img in enumerate(imgs):
            page_num = i + 1
            
            # Page header
            st.markdown(f"### 📄 Page {page_num}")
            st.markdown(f"*{img['filename']}*")
            
            # Check if this page contains search results
            page_has_match = False
            matching_text = ""
            if search_query.strip():
                # Get OCR text for this specific image
                conn = sqlite3.connect(DB_PATH)
                conn.row_factory = sqlite3.Row
                cur = conn.cursor()
                cur.execute("SELECT ocr_text FROM images WHERE filepath = ?", (img["filepath"],))
                row = cur.fetchone()
                conn.close()
                
                if row and row["ocr_text"] and search_query.lower() in row["ocr_text"].lower():
                    page_has_match = True
                    # Extract context around the match
                    text = row["ocr_text"]
                    query_lower = search_query.lower()
                    text_lower = text.lower()
                    match_pos = text_lower.find(query_lower)
                    if match_pos != -1:
                        start = max(0, match_pos - 50)
                        end = min(len(text), match_pos + len(search_query) + 50)
                        context = text[start:end]
                        # Highlight the search term
                        highlighted = context.replace(search_query, f"🔍**{search_query}**", 1)
                        matching_text = f"...{highlighted}..." if start > 0 or end < len(text) else highlighted
            
            # Highlight pages with matches
            if page_has_match:
                st.success(f"🎯 **MATCH FOUND ON THIS PAGE**")
                st.markdown(f"**Matching text:** {matching_text}")
            
            # Display the image (full width for better readability)
            try:
                pil_img = Image.open(img["filepath"]).convert("RGB")
                # Scale image to fit column width while maintaining aspect ratio
                max_width = 800
                if pil_img.width > max_width:
                    ratio = max_width / pil_img.width
                    new_height = int(pil_img.height * ratio)
                    pil_img = pil_img.resize((max_width, new_height), Image.Resampling.LANCZOS)
                
                buf = io.BytesIO()
                pil_img.save(buf, format="JPEG", quality=85)
                st.image(buf.getvalue(), width=max_width)
                
            except Exception as e:
                st.error(f"Error loading image: {e}")
            
            # Add some spacing between pages
            st.markdown("---")
    else:
        st.info("👈 Select a case from the left panel to view its documents")

st.markdown("---")
st.caption("Replace CLIP with your on-prem model (OpenCLIP or another) by swapping EmbedIndex's model loader. For large datasets, use faiss-gpu and batching.")

# ---------------- Optional: CLI training skeleton ----------------
def train_vlm_skeleton(train_metadata: List[Tuple[str,str]], output_dir="vlm_finetune", epochs=3, batch_size=32, lr=1e-5):
    """
    Very small training skeleton to fine-tune a CLIP-like model contrastively.
    train_metadata: list of (image_path, text_caption)
    """
    import os
    import numpy as np
    import torch
    from PIL import Image
    os.makedirs(output_dir, exist_ok=True)
    device = DEVICE
    st.info("Starting training skeleton (runs in-process).")
    model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(device)
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    def batchify(dataset, bs):
        for i in range(0, len(dataset), bs):
            yield dataset[i:i+bs]

    for epoch in range(epochs):
        np.random.shuffle(train_metadata)
        losses = []
        for batch in batchify(train_metadata, batch_size):
            images = [Image.open(p).convert("RGB") for p,_ in batch]
            texts = [t for _,t in batch]
            inputs = processor(text=texts, images=images, return_tensors="pt", padding=True).to(device)
            outputs = model(**inputs)
            image_emb = outputs.image_embeds
            text_emb = outputs.text_embeds
            image_emb = image_emb / image_emb.norm(dim=-1, keepdim=True)
            text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)
            logits = image_emb @ text_emb.t()
            labels = torch.arange(len(batch), device=device)
            loss_i = torch.nn.functional.cross_entropy(logits, labels)
            loss_t = torch.nn.functional.cross_entropy(logits.t(), labels)
            loss = (loss_i + loss_t) / 2.0
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        st.write(f"Epoch {epoch+1}/{epochs} avg loss: {np.mean(losses):.4f}")
        model.save_pretrained(os.path.join(output_dir, f"checkpoint_epoch_{epoch+1}"))
    st.success("Training skeleton finished.")

# Allow running training skeleton via CLI if user runs with --train-skeleton
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", action="store_true", help="Discover images and build index (headless).")
    parser.add_argument("--train-skeleton", action="store_true", help="Run training skeleton on a small sample.")
    args = parser.parse_args()
    # headless index (non-streamlit run)
    if args.index:
        init_db()
        n = discover_and_register_images(DATA_ROOT_DEFAULT)
        print(f"discovered {n} images")
        fps = gather_all_filepaths_from_db()
        if not fps:
            print("no files to index")
            exit(0)
        emb = EmbedIndex()
        emb.build_index(fps)
        emb.save_index()
        print("Index saved.")
        exit(0)
    if args.train_skeleton:
        init_db()
        fps = gather_all_filepaths_from_db()
        if not fps:
            print("no images for training skeleton")
            exit(1)
        # toy captions = filenames; replace with real captions/OCR
        train_meta = [(p, f"Document image named {os.path.basename(p)}") for p in fps[:256]]
        train_vlm_skeleton(train_meta, epochs=1, batch_size=8)
        exit(0)
