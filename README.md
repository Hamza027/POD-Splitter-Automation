# POD Splitter (D-number) — PyMuPDF + Tesseract

Splits inbound scanned POD PDFs into per-**D-number** PDFs with:
- **Document-level orientation** (0°/180°) using header keyword votes, OSD, and D-hit tie-breaks
- **Robust OCR** via Tesseract with confidence gating and look-ahead grouping
- Deterministic outputs: `{DNUMBER}.pdf` saved **upright** in `Outbound/`

## Requirements
- Python 3.10+
- Tesseract OCR on Windows  
  Default path expected by the script:
  `C:\Program Files\Tesseract-OCR\tesseract.exe`
- Python packages: see `requirements.txt`

## Quick Start
1. Install deps:
   ```bash
   pip install -r requirements.txt
   ```
2. Ensure Tesseract is installed and tesseract.exe exists at the default path
  (or update the path in the script).

3. Set your working folder in the script:
  BASE_DIR = r"C:\Path\To\Working\Folder"

4. Run:
```bash
python split_pods_by_dnumber.py 
```


Folder Layout (auto-created under BASE_DIR)
````text
inbound/          # source PDFs
Outbound/         # outputs: {DNUMBER}.pdf (overwritten)
Processed/        # logs: {ORIGINAL_STEM}__detected_map.txt
Errored/          # originals on error + {stem}.txt reason
inbound_archive/  # originals after successful processing
````


Notes
Tune OCR/grouping via constants in the script (RENDER_SCALE, CONF_*, ANCHOR_KEYWORDS).

If Pillow >= 11 triggers a warning, prefer < 11 (see requirements.txt).
