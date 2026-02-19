# Deployment Notes

## Successful Deployment - February 19, 2026

### Final Working Configuration:
- **Python Version**: 3.12.12
- **Environment**: pdr_clean conda environment
- **Server**: http://aadsprodsv:3939/
- **Entry Point**: img_llm_improved_v2.py

### Troubleshooting History:

#### Issue: Python Version Compatibility
- **Problem**: RStudio Connect server didn't support the specific Python versions available in local environments
- **Attempted Versions**:
  - Python 3.9.25 ❌ (not supported)
  - Python 3.10.19 ❌ (not supported)
- **Solution**: Upgraded pdr_clean environment to Python 3.12.12 ✅

#### Key Learnings:
1. `rsconnect` reads Python version from the active environment, not from `.python-version` or `runtime.txt`
2. RStudio Connect server has specific Python versions installed - need to match exactly
3. Python 3.12.x was supported on this server
4. The `--entrypoint` parameter was crucial for specifying the main application file

### Files Created During Deployment Process:
- `.python-version` - Version specification file
- `runtime.txt` - Runtime environment specification  
- `pyproject.toml` - Project configuration with dependencies
- `DEPLOYMENT.md` - Deployment instructions
- `manifest.json` (temporary) - Was removed during troubleshooting

### Future Deployments:
1. Ensure using pdr_clean environment with Python 3.12
2. Run from `/home/rmir/PDR/PDR_PDF` directory
3. Use the exact command that worked:
   ```bash
   rsconnect deploy streamlit . --entrypoint img_llm_improved_v2.py --server http://aadsprodsv:3939/ --api-key YOUR_API_KEY --title "PDR Document Viewer"
   ```