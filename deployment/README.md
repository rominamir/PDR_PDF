# RStudio Connect Deployment Files

This folder contains all the files needed for deploying the PDR Document Viewer to RStudio Connect.

## Files:

### Configuration Files:
- `.python-version` - Specifies Python version (3.12)
- `runtime.txt` - Runtime specification for deployment
- `pyproject.toml` - Python project configuration with dependencies
- `requirements_deploy.txt` - Minimal deployment requirements

### Documentation:
- `DEPLOYMENT.md` - Complete deployment instructions
- `deployment_notes.md` - Deployment history and troubleshooting

### Deployment Command:
```bash
# From the main PDR_PDF directory:
rsconnect deploy streamlit . \
  --entrypoint img_llm_improved_v2.py \
  --server http://aadsprodsv:3939/ \
  --api-key YOUR_API_KEY \
  --title "PDR Document Viewer"
```

## Environment:
- Successfully deployed using Python 3.12.12 in the `pdr_clean` conda environment
- Main application file: `img_llm_improved_v2.py` (located in parent directory)

## Notes:
- The deployment was successful after upgrading to Python 3.12
- RStudio Connect server supports Python 3.12.x but not the specific minor versions we initially tried
- All deployment configuration files should remain in this folder for future updates