"""
img_llm.py

Streamlit web app for:
 - browsing "cases" (folders with images)
 - semantic text search over PNG/JPG/TIFF images (CLIP embeddings + FAISS)
 - building / loading FAISS index
 - a training skeleton function if you want to fine-tune a VLM locally.

Usage:
  1) Install dependencies:
     pip install streamlit pillow sqlalchemy transformers torch torchvision tqdm faiss-cpu
     # or faiss-gpu if you have GPU + CUDA and want faster indexing/search

  2) Edit DATA_ROOT (default "./data") or supply via UI.

  3) Build index from UI (Discover images -> Build index) OR run:
     python streamlit_vlm_search.py --index

  4) Run:
     streamlit run img_llm.py
"""
import os
import io
import json
import sqlite3
import argparse
from typing import List, Tuple

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

# ---------------- Config ----------------
DATA_ROOT_DEFAULT = "./data"
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
        processed INTEGER DEFAULT 0
    )
    """)
    conn.commit()
    conn.close()

def upsert_image(case_number: str, filename: str, filepath: str):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT id FROM images WHERE filepath = ?", (filepath,))
    if cur.fetchone() is None:
        cur.execute("INSERT INTO images (case_number, filename, filepath) VALUES (?, ?, ?)",
                    (case_number, filename, filepath))
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

# ---------------- Discovery ----------------
def discover_and_register_images(data_root: str):
    init_db()
    cnt = 0
    for root, dirs, files in os.walk(data_root):
        rel = os.path.relpath(root, data_root)
        if rel == ".":
            continue
        case_number = rel.replace(os.sep, "/")
        for f in files:
            if os.path.splitext(f)[1].lower() in SUPPORTED_EXT:
                fp = os.path.join(root, f)
                upsert_image(case_number, f, fp)
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
    st.write("Expect structure: DATA_ROOT/case_xyz/*.png")
    st.write(f"Device: {DEVICE}")
    st.markdown("---")

    st.header("Index Controls")
    if st.button("Discover & register images"):
        with st.spinner("Discovering images..."):
            n = discover_and_register_images(DATA_ROOT)
        st.success(f"Discovered & registered {n} images (if any).")
    st.write("After discovery, build or load FAISS index.")
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

# Main: case selection and gallery
cases = list_open_cases()
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Cases")
    if not cases:
        st.info("No cases found. Run 'Discover & register images' in the sidebar.")
    selected_case = st.selectbox("Select a case", options=[""] + cases)
    if selected_case:
        imgs = list_images_for_case(selected_case)
        st.write(f"{len(imgs)} images in {selected_case}")
        # show thumbnails
        for img in imgs:
            cols = st.columns([1, 4])
            with cols[0]:
                try:
                    pil = Image.open(img["filepath"]).convert("RGB")
                    pil.thumbnail((180, 180))
                    buf = io.BytesIO()
                    pil.save(buf, format="JPEG")
                    st.image(buf.getvalue(), use_column_width=True)
                except Exception as e:
                    st.write("thumbnail error")
            with cols[1]:
                st.write(img["filename"])
                if st.button(f"Open ({img['filename']})", key=img["filepath"]):
                    st.image(img["filepath"])

with col2:
    st.subheader("Semantic image search")
    query = st.text_input("Enter text query to search across images (semantic search)")
    case_filter = st.selectbox("Optional case filter (restrict results)", options=[""] + cases)
    k = st.slider("Max results (k)", min_value=1, max_value=50, value=12)
    if st.button("Search"):
        if st.session_state.embedder is None or not st.session_state.index_built:
            st.warning("Index not loaded. Build or Load index from the sidebar first.")
        elif not query.strip():
            st.info("Type a query first.")
        else:
            with st.spinner("Searching..."):
                try:
                    results = st.session_state.embedder.search_text(query, k=k)
                    if case_filter:
                        results = [r for r in results if case_filter in r[0]]
                    if not results:
                        st.info("No results.")
                    else:
                        for fp,score in results:
                            cols = st.columns([1,4])
                            with cols[0]:
                                try:
                                    pil = Image.open(fp).convert("RGB")
                                    pil.thumbnail((240,240))
                                    buf = io.BytesIO()
                                    pil.save(buf, format="JPEG")
                                    st.image(buf.getvalue())
                                except Exception as e:
                                    st.write("img err")
                            with cols[1]:
                                st.markdown(f"**Score:** {score:.4f}")
                                st.write(fp)
                except Exception as e:
                    st.error(f"Search failed: {e}")

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
