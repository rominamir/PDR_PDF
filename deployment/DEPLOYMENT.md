# RStudio Connect Deployment

## Local Environment Setup
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

## Local Testing
streamlit run app.py

## Deploy to RStudio Connect

### Prerequisites
- rsconnect-python installed: `pip install rsconnect-python`
- RStudio Connect server URL and API key configured

### Deployment Command
```bash
rsconnect deploy streamlit . \
  --server https://your-rstudio-connect-server \
  --api-key your-api-key \
  --title "PDR Document Viewer" \
  --python-version 3.9
```

### Alternative deployment (with more options):
```bash
rsconnect deploy streamlit . \
  --server https://your-rstudio-connect-server \
  --api-key your-api-key \
  --title "PDR Document Viewer" \
  --description "Document viewer with OCR and AI analysis capabilities" \
  --python-version 3.9 \
  --environment ENVIRONMENT_NAME
```

### Environment Variables
Set these in RStudio Connect after deployment:
- `DATA_ROOT`: Path to document storage (default: /hspshare/converted_images)

## Notes
- The application uses SQLite databases for caching (ocr_cache.db, cases_streamlit.db)
- Requires access to file system where documents are stored
- Optional Ollama integration for AI analysis (requires Ollama server)
- OpenCV is used for image preprocessing if available