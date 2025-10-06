# -*- coding: utf-8 -*-
"""
Split inbound PDFs of scanned PODs into per-Dnumber PDFs (doc-level orientation, robust OCR, look-ahead grouping).

Folders (under BASE_DIR):
  ./inbound/          -> source PDFs
  ./Outbound/         -> outputs: {DNUMBER}.pdf (always overwritten; pages saved upright)
  ./Processed/        -> logs only: {ORIGINAL_STEM}__detected_map.txt
  ./Errored/          -> originals moved here on error + .txt reason
  ./inbound_archive/  -> originals moved here after successful processing

Pipeline:
- Choose WHOLE-PDF orientation (0° or 180°): keyword totals -> OSD votes -> sample D# hits.
- Detect D-number once per page on that orientation using Tesseract image_to_data() with whitelist; pick best-confidence candidate.
- Group with confidence-gated look-ahead to avoid false 1-digit splits.
- Write split PDFs (rotated to chosen orientation), archive original, clean inbound.
"""

import re
import sys
import os
import time
import shutil
from pathlib import Path

# --- Dependencies ---
try:
    import fitz  # PyMuPDF
except Exception:
    print("ERROR: PyMuPDF (fitz) is not installed. Try: pip install --upgrade pymupdf")
    sys.exit(1)

try:
    from PIL import Image, ImageOps, ImageFilter, __version__ as PIL_VERSION
except Exception:
    print("ERROR: Pillow is not installed. Try: pip install 'pillow<11,>=10.0.0'")
    sys.exit(1)

try:
    import pytesseract
    from pytesseract import Output
except Exception:
    print("ERROR: pytesseract is not installed. Try: pip install pytesseract")
    sys.exit(1)

try:
    import numpy as np
except Exception:
    print("ERROR: numpy is not installed. Try: pip install numpy")
    sys.exit(1)

# --- Enforce Pillow compatibility (<11) ---
def _parse_ver(v):
    m = re.match(r"(\d+)\.(\d+)\.(\d+)", v)
    return tuple(int(x) for x in m.groups()) if m else (0, 0, 0)

if _parse_ver(PIL_VERSION) >= (11, 0, 0):
    print(f"WARNING: Pillow {PIL_VERSION} detected. Code was originally tested with <11, but continuing anyway.")

# --- Hard-point pytesseract to system Tesseract (Chocolatey default path) ---
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# --- Settings / Knobs ---
DEBUG_LOG = True
RENDER_SCALE = 4               # 72 * 4 = ~288 DPI
OSD_CONF_THRESHOLD = 8.0       # only trust OSD when confidence >= this
SAMPLE_MAX_PAGES = 12

# Grouping thresholds:
CONF_STRONG_NEW = 80           # strong confidence -> allow new group immediately
CONF_MIN_CONFIRM = 55          # else require next page to confirm with at least this
HAMMING_MAX_SNAP = 1           # treat ≤1-digit diff as potentially noisy without confirmation

# Keywords expected in headers (case-insensitive)
ANCHOR_KEYWORDS = [
    "delivery note",
    "delivery no",
    "order date",
    "despatch date",
    "account no",
    "delivery method",
    "consignment no",
    "method of transport",
    "no of parcels",
    "customer name",
    "address",
    "onsite support",
]

# D-number patterns (strict first, then fallback)
DNUM_PATTERNS = [
    re.compile(r"\bD\d{7}\b"),
    re.compile(r"\bD\d{8}\b"),
]

# ----------------- Common helpers -----------------

def ensure_dirs(base: Path):
    inbound   = base / "inbound"
    processed = base / "Processed"       # keep for logs (__detected_map.txt)
    outbound  = base / "Outbound"        # NEW: final PDFs go here
    errored   = base / "Errored"
    archive   = base / "inbound_archive"
    for p in (inbound, processed, outbound, errored, archive):
        p.mkdir(parents=True, exist_ok=True)
    return inbound, processed, outbound, errored, archive

def normalize_dnum(raw: str) -> str:
    """Normalize to canonical D####### / D######## form; reject if not exact."""
    s = raw.strip().upper()
    s = s.replace("O", "0").replace("I", "1").replace("L", "1")
    s = re.sub(r"[\s\-]", "", s)
    for pat in DNUM_PATTERNS:
        m = pat.search(s)
        if m:
            return m.group(0)
    return ""

def extract_dnumber_from_text(text: str) -> str | None:
    if not text:
        return None
    for pat in DNUM_PATTERNS:
        m = pat.search(text)
        if m:
            d = normalize_dnum(m.group(0))
            if d:
                return d
    return None

def dnum_hamming(a: str | None, b: str | None) -> int:
    """Hamming distance on numeric tail; return large number if shapes differ."""
    if not a or not b:
        return 99
    a = a.upper(); b = b.upper()
    if not (a.startswith("D") and b.startswith("D")):
        return 99
    ta, tb = a[1:], b[1:]
    if len(ta) != len(tb):
        return 99
    return sum(1 for x, y in zip(ta, tb) if x != y)

# ----------------- OCR / Image helpers -----------------

def _render_page_image(page, scale=RENDER_SCALE) -> Image.Image:
    mat = fitz.Matrix(scale, scale)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    mode = "RGB" if pix.n < 4 else "RGBA"
    return Image.frombytes(mode, (pix.width, pix.height), pix.samples)

def _binarize(img: Image.Image) -> Image.Image:
    g = ImageOps.grayscale(img)
    g = g.filter(ImageFilter.SHARPEN)
    arr = np.array(g)
    thr = max(120, int(arr.mean() * 0.9))
    bw = (arr > thr).astype(np.uint8) * 255
    return Image.fromarray(bw, mode="L")

def _ocr_text(img: Image.Image, psm=6) -> str:
    return pytesseract.image_to_string(
        img, config=f"--psm {psm} -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789:/.-"
    )

def _osd_info(img: Image.Image):
    """Return (rotate_deg, orientation_confidence) or (None, 0)."""
    try:
        osd = pytesseract.image_to_osd(img)
        m1 = re.search(r"Rotate:\s+(\d+)", osd)
        m2 = re.search(r"Orientation confidence:\s+([\d.]+)", osd)
        rot = int(m1.group(1)) if m1 else None
        conf = float(m2.group(1)) if m2 else 0.0
        return rot, conf
    except Exception:
        return None, 0.0

# ----------------- Orientation logic (document-level) -----------------

def _header_rois(img: Image.Image):
    w, h = img.size
    band_h = int(0.28 * h)
    return [
        img.crop((0, 0, w, band_h)),                                   # full top band
        img.crop((int(0.55 * w), 0, w, band_h)),                       # top-right
        img.crop((int(0.30 * w), 0, int(0.75 * w), band_h)),           # top-center/right
    ]

def _keyword_score(img: Image.Image) -> int:
    score = 0
    for roi in _header_rois(img):
        texts = []
        try: texts.append(_ocr_text(roi, psm=6))
        except: pass
        try: texts.append(_ocr_text(_binarize(roi), psm=6))
        except: pass
        t = " ".join(texts).lower()
        for kw in ANCHOR_KEYWORDS:
            if kw in t:
                score += 1
    return score

def _dnum_roi(img: Image.Image) -> Image.Image:
    w, h = img.size
    return img.crop((int(0.58 * w), 0, w, int(0.35 * h)))

# ----------------- High-confidence D-number detection -----------------

def _ocr_find_best_dnum(img: Image.Image, psm=6) -> tuple[str | None, int]:
    """
    Use image_to_data() to get word-level confidences. Return (best_dnum, best_conf).
    """
    best_d, best_conf = None, -1
    try:
        data = pytesseract.image_to_data(
            img,
            config=f"--psm {psm} -c tessedit_char_whitelist=Dd0123456789",
            output_type=Output.DICT
        )
        n = len(data.get("text", []))
        for i in range(n):
            s = (data["text"][i] or "").strip()
            if not s:
                continue
            cand = extract_dnumber_from_text(s)
            if not cand:
                continue
            conf = int(data["conf"][i]) if data["conf"][i] not in (None, "", "-1") else -1
            if conf > best_conf:
                best_d, best_conf = cand, conf
    except Exception:
        pass

    if not best_d:
        # Fallback plain OCR:
        try:
            txt = pytesseract.image_to_string(
                img, config=f"--psm {psm} -c tessedit_char_whitelist=Dd0123456789"
            )
            cand = extract_dnumber_from_text(txt)
            if cand:
                best_d, best_conf = cand, 0
        except Exception:
            pass

    return best_d, best_conf

def detect_dnumber_on_oriented(img: Image.Image) -> tuple[str | None, int]:
    """
    Return (best_dnum, best_conf). Check top-right ROI (orig + bin), then headers, then page.
    """
    best_d, best_conf = None, -1

    # 1) Top-right ROI
    roi = _dnum_roi(img)
    d, c = _ocr_find_best_dnum(roi, psm=6)
    if d and c > best_conf: best_d, best_conf = d, c
    d, c = _ocr_find_best_dnum(_binarize(roi), psm=6)
    if d and c > best_conf: best_d, best_conf = d, c

    # 2) Header bands
    for hdr in _header_rois(img):
        d, c = _ocr_find_best_dnum(hdr, psm=6)
        if d and c > best_conf: best_d, best_conf = d, c
        d, c = _ocr_find_best_dnum(_binarize(hdr), psm=6)
        if d and c > best_conf: best_d, best_conf = d, c

    # 3) Full page fallback
    d, c = _ocr_find_best_dnum(img, psm=6)
    if d and c > best_conf: best_d, best_conf = d, c
    d, c = _ocr_find_best_dnum(_binarize(img), psm=6)
    if d and c > best_conf: best_d, best_conf = d, c

    return best_d, best_conf

# ----------------- Document orientation vote -----------------

def _doc_orientation_vote(images: list[Image.Image]) -> tuple[int, dict]:
    sum0 = 0
    sum180 = 0
    osd_votes_0 = 0
    osd_votes_180 = 0
    osd_conf_count = 0

    for img in images:
        sum0 += _keyword_score(img)
        sum180 += _keyword_score(img.rotate(180, expand=True))
        rot, conf = _osd_info(img)
        if rot is not None and conf >= OSD_CONF_THRESHOLD:
            osd_conf_count += 1
            if rot % 360 == 0: osd_votes_0 += 1
            if rot % 360 == 180: osd_votes_180 += 1

    debug = {
        "sum0": sum0, "sum180": sum180,
        "osd_votes_0": osd_votes_0, "osd_votes_180": osd_votes_180,
        "osd_conf_count": osd_conf_count,
    }

    if sum0 > sum180: return 0, debug
    if sum180 > sum0: return 180, debug

    if osd_conf_count > 0:
        if osd_votes_180 > osd_votes_0: return 180, debug
        if osd_votes_0 > osd_votes_180: return 0, debug

    # Tie-breaker: sample pages and count valid D-hits per orientation
    sample = images[:SAMPLE_MAX_PAGES]
    hits0 = 0
    hits180 = 0
    for img in sample:
        d0, _ = detect_dnumber_on_oriented(img)
        if d0: hits0 += 1
        d1, _ = detect_dnumber_on_oriented(img.rotate(180, expand=True))
        if d1: hits180 += 1

    debug.update({"sample_hits0": hits0, "sample_hits180": hits180})
    return (180 if hits180 > hits0 else 0), debug

# ----------------- IO helpers -----------------

def write_groups_to_pdfs(src_pdf: Path, groups: list[tuple[str, list[int]]], out_dir: Path, rotate_deg: int = 0):
    """
    Always overwrite existing {dnum}.pdf.
    After inserting, explicitly rotate every page by rotate_deg (0 or 180),
    adding to any existing /Rotate the page may already have.
    """
    rotate_deg = (rotate_deg or 0) % 360
    with fitz.open(src_pdf) as in_doc:
        for dnum, page_idxs in groups:
            out_path = out_dir / f"{dnum}.pdf"
            out_doc = fitz.open()

            # insert pages as-is (no rotate kwarg)
            for i in page_idxs:
                out_doc.insert_pdf(in_doc, from_page=i, to_page=i)

            # force rotation on ALL pages in the output file
            if rotate_deg:
                for pg in out_doc:
                    try:
                        current = pg.rotation or 0
                    except Exception:
                        current = 0
                    pg.set_rotation((current + rotate_deg) % 360)

            out_doc.save(out_path, deflate=True)
            out_doc.close()

            
def move_to_errored(src: Path, errored_dir: Path, reasons: list[str], detected_map=None):
    dst = errored_dir / src.name
    try:
        shutil.move(str(src), str(dst))
    except Exception:
        try:
            shutil.copy2(src, dst)
        except Exception:
            pass
    reason_path = errored_dir / f"{src.stem}.txt"
    try:
        with open(reason_path, "w", encoding="utf-8") as f:
            f.write(f"Failed to split: {src.name}\n\n")
            for r in reasons:
                f.write(f"- {r}\n")
    except Exception:
        pass
    if DEBUG_LOG and detected_map:
        log_path = errored_dir / f"{src.stem}__detected_map.txt"
        try:
            with open(log_path, "w", encoding="utf-8") as f:
                for line in detected_map:
                    f.write(line + "\n")
        except Exception:
            pass
    print(f" -> {src.name}: moved to Errored/ (see {reason_path.name})")

# ----------------- Core splitter -----------------

def split_pdf_by_dnumber(
    pdf_path: Path,
    out_dir: Path,                 # PDFs go here (Outbound)
    errored_dir: Path,
    archive_dir: Path,
    processed_log_dir: Path | None = None  # __detected_map.txt goes here (Processed)
) -> None:
    reasons: list[str] = []
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        reasons.append(f"Cannot open PDF: {e}")
        move_to_errored(pdf_path, errored_dir, reasons)
        return

    # 1) Render all pages
    try:
        page_images = [_render_page_image(p) for p in doc]
    except Exception as e:
        reasons.append(f"Rendering error: {e}")
        move_to_errored(pdf_path, errored_dir, reasons)
        try: doc.close()
        except: pass
        return

    # 2) Decide doc-level orientation (0 or 180)
    doc_deg, debug_info = _doc_orientation_vote(page_images)

    # 3) Detect once per page on chosen orientation (store raw + conf + context)
    detections = []  # list of dicts per page
    for i, base_img in enumerate(page_images):
        oriented = base_img if doc_deg == 0 else base_img.rotate(doc_deg, expand=True)
        raw, conf = detect_dnumber_on_oriented(oriented)
        score0 = _keyword_score(base_img)
        score180 = _keyword_score(base_img.rotate(180, expand=True))
        osd_rot, osd_conf = _osd_info(base_img)
        detections.append({
            "raw": raw, "conf": conf,
            "score0": score0, "score180": score180,
            "osd_rot": osd_rot, "osd_conf": osd_conf
        })

    # 4) Group with confidence-gated look-ahead
    groups: list[tuple[str, list[int]]] = []
    current_dnum: str | None = None
    leading_missing: list[int] = []
    log_lines = []
    N = len(detections)

    def page_log(i, decision, assigned, snapped, info):
        log_lines.append(
            f"page {i+1:>3}: DOC_ROT={doc_deg:>3}° "
            f"score0={info['score0']:>2} score180={info['score180']:>2} "
            f"osd_rot={(info['osd_rot'] if info['osd_rot'] is not None else 'None'):>4} "
            f"osd_conf={info['osd_conf']:>4.1f}  "
            f"raw={info['raw'] or 'None'}(conf={info['conf']}) "
            f"snapped={'Y' if snapped else 'N'} decision={decision} -> assigned {assigned or 'None'}"
        )

    i = 0
    while i < N:
        info = detections[i]
        raw, conf = info["raw"], info["conf"]
        snapped = False
        decision = ""

        if raw:
            if current_dnum is None:
                # Start first group
                groups.append((raw, [i]))
                current_dnum = raw
                decision = "START"
            else:
                if raw == current_dnum:
                    groups[-1][1].append(i)
                    decision = "CONTINUE"
                else:
                    # diff from current
                    hd = dnum_hamming(raw, current_dnum)
                    if hd <= HAMMING_MAX_SNAP:
                        # Maybe noise; require strong conf or next-page confirmation to switch
                        if conf >= CONF_STRONG_NEW:
                            # Strong enough: start new group
                            groups.append((raw, [i]))
                            current_dnum = raw
                            decision = "NEW_STRONG"
                        else:
                            # Check next page confirmation (if exists)
                            if i+1 < N and detections[i+1]["raw"] == raw and detections[i+1]["conf"] >= CONF_MIN_CONFIRM:
                                groups.append((raw, [i]))
                                current_dnum = raw
                                decision = "NEW_CONFIRMED_NEXT"
                            else:
                                # Snap back to current group
                                groups[-1][1].append(i)
                                snapped = True
                                decision = "SNAP_KEEP"
                    else:
                        # Clearly a different D#: require some confidence to start new group
                        if conf >= CONF_MIN_CONFIRM or (i+1 == N):  # allow last page with moderate conf
                            groups.append((raw, [i]))
                            current_dnum = raw
                            decision = "NEW_CLEAR"
                        else:
                            # Try next-page confirmation
                            if i+1 < N and detections[i+1]["raw"] == raw and detections[i+1]["conf"] >= CONF_MIN_CONFIRM:
                                groups.append((raw, [i]))
                                current_dnum = raw
                                decision = "NEW_CONFIRMED_NEXT"
                            else:
                                # fallback: keep current
                                groups[-1][1].append(i)
                                snapped = True
                                decision = "SNAP_KEEP"
        else:
            # No D read; inherit if we have a current group
            if current_dnum:
                groups[-1][1].append(i)
                decision = "INHERIT"
            else:
                leading_missing.append(i+1)
                decision = "LEADING_NONE"

        assigned = current_dnum if current_dnum else None
        page_log(i, decision, assigned, snapped, info)
        i += 1

    # 5) Error checks
    reasons = []
    if not groups:
        reasons.append("No D-number detected in any page.")
    if leading_missing:
        reasons.append(f"Starting pages missing D-number: pages {leading_missing}.")
    if reasons:
        header = (
            f"[DOC ORIENTATION] chosen={doc_deg}°  "
            f"sum0={debug_info.get('sum0',0)}  sum180={debug_info.get('sum180',0)}  "
            f"osd_votes_0={debug_info.get('osd_votes_0',0)}  osd_votes_180={debug_info.get('osd_votes_180',0)}  "
            f"osd_conf_cnt={debug_info.get('osd_conf_count',0)}  "
            f"sample_hits0={debug_info.get('sample_hits0','-')}  sample_hits180={debug_info.get('sample_hits180','-')}"
        )
        move_to_errored(pdf_path, errored_dir, reasons, [header] + log_lines)
        try: doc.close()
        except: pass
        return

    # 6) Write outputs (rotate pages so outputs are upright) -> to Outbound
    try:
        write_groups_to_pdfs(pdf_path, groups, out_dir, rotate_deg=doc_deg)
    except Exception as e:
        reasons = [f"Writing outputs failed: {e}"]
        header = (
            f"[DOC ORIENTATION] chosen={doc_deg}°  "
            f"sum0={debug_info.get('sum0',0)}  sum180={debug_info.get('sum180',0)}  "
            f"osd_votes_0={debug_info.get('osd_votes_0',0)}  osd_votes_180={debug_info.get('osd_votes_180',0)}  "
            f"osd_conf_cnt={debug_info.get('osd_conf_count',0)}  "
            f"sample_hits0={debug_info.get('sample_hits0','-')}  sample_hits180={debug_info.get('sample_hits180','-')}"
        )
        move_to_errored(pdf_path, errored_dir, reasons, [header] + log_lines)
        try: doc.close()
        except: pass
        return

    # 7) Close doc before file ops
    try:
        doc.close()
    except Exception:
        pass

    # 8) Archive original and ensure inbound is cleaned
    dst = archive_dir / pdf_path.name
    archived = False
    try:
        shutil.move(str(pdf_path), str(dst))
        archived = True
    except Exception:
        try:
            shutil.copy2(pdf_path, dst)
            archived = True
        except Exception:
            archived = False
        if archived:
            for _ in range(10):
                try:
                    pdf_path.unlink()
                    break
                except Exception:
                    time.sleep(0.2)
            else:
                try:
                    os.remove(pdf_path)
                except Exception:
                    pass

    # 9) Write detection map next to outputs (with doc decision header) -> to Processed
    if DEBUG_LOG:
        target_logs_dir = processed_log_dir if processed_log_dir else out_dir
        log_path = target_logs_dir / f"{pdf_path.stem}__detected_map.txt"
        try:
            with open(log_path, "w", encoding="utf-8") as f:
                f.write(
                    f"[DOC ORIENTATION] chosen={doc_deg}°  "
                    f"sum0={debug_info.get('sum0',0)}  sum180={debug_info.get('sum180',0)}  "
                    f"osd_votes_0={debug_info.get('osd_votes_0',0)}  osd_votes_180={debug_info.get('osd_votes_180',0)}  "
                    f"osd_conf_cnt={debug_info.get('osd_conf_count',0)}  "
                    f"sample_hits0={debug_info.get('sample_hits0','-')}  sample_hits180={debug_info.get('sample_hits180','-')}\n"
                )
                for line in log_lines:
                    f.write(line + "\n")
        except Exception:
            pass

    print(f" -> {pdf_path.name}: wrote {len(groups)} file(s) to Outbound/, archived original.")

# ----------------- main -----------------
BASE_DIR = r"C:\Users\HamzaA\Desktop\Automation Scripts\__HA_Projects\Docuware_Split_Inbound"

def main():
    # Use manual base path from CONFIG
    base = Path(BASE_DIR).expanduser().resolve()
    if not base.exists():
        try:
            base.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            print(f"ERROR: Could not create base directory '{base}': {e}")
            sys.exit(1)

    # Note: ensure_dirs now returns 5 items
    inbound_dir, processed_dir, outbound_dir, errored_dir, archive_dir = ensure_dirs(base)

    pdfs = sorted(inbound_dir.glob("*.pdf"))
    print(f"Base: {base}")
    print(f"Found {len(pdfs)} file(s) in {inbound_dir}")

    try:
        print("Tesseract:", pytesseract.get_tesseract_version())
    except Exception as e:
        print("WARNING: Could not read Tesseract version:", e)

    if not pdfs:
        print("No PDFs to process.")
        return

    for p in pdfs:
        print(f"Processing: {p.name}")
        # PDFs -> Outbound ; Logs -> Processed
        split_pdf_by_dnumber(p, outbound_dir, errored_dir, archive_dir, processed_log_dir=processed_dir)

    print("Done.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(1)
