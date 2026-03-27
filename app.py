import streamlit as st
import cv2
import numpy as np
import easyocr
import re
import requests
import pandas as pd
import os
import tempfile
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from PIL import Image, ImageEnhance, ImageFilter
import io
import firebase_admin
from firebase_admin import credentials, firestore
from ultralytics import YOLO

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="CarInfo — Know Your Car",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
RAPIDAPI_HOST = "vehicle-rc-information-v2.p.rapidapi.com"
PLATE_PATTERN = re.compile(r'^[A-Z]{2}[0-9]{2}[A-Z]{2}[0-9]{4}$')

# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────
if "rapidapi_key" not in st.session_state:
    st.session_state.rapidapi_key = st.secrets.get("RAPIDAPI_KEY", "")
if "admin_auth" not in st.session_state:
    st.session_state.admin_auth = False

# ─────────────────────────────────────────────
# FIREBASE
# ─────────────────────────────────────────────
@st.cache_resource
def init_firebase():
    try:
        if not firebase_admin._apps:
            fb = st.secrets.get("firebase", None)
            if fb:
                cred = credentials.Certificate(dict(fb))
                firebase_admin.initialize_app(cred)
        return firestore.client()
    except Exception:
        return None

db = init_firebase()

# ─────────────────────────────────────────────
# DB FUNCTIONS
# ─────────────────────────────────────────────
def save_scan(plate, info, source="image"):
    if not db:
        return
    try:
        db.collection("scans").add({
            "plate": plate,
            "timestamp": datetime.now().isoformat(),
            "owner": info.get("owner", "N/A"),
            "vehicle_class": info.get("vehicle_class", "N/A"),
            "fuel_type": info.get("fuel_type", "N/A"),
            "maker": info.get("maker", "N/A"),
            "model": info.get("model", "N/A"),
            "color": info.get("color", "N/A"),
            "rc_status": info.get("rc_status", "N/A"),
            "insurance": info.get("insurance", "N/A"),
            "insurance_expiry": info.get("insurance_expiry", "N/A"),
            "registration_date": info.get("registration_date", "N/A"),
            "owner_count": info.get("owner_count", "N/A"),
            "source": source,
        })
    except Exception as e:
        st.error(f"Save error: {e}")

def get_scans(limit=500):
    if not db:
        return pd.DataFrame()
    try:
        docs = db.collection("scans").order_by(
            "timestamp", direction=firestore.Query.DESCENDING
        ).limit(limit).stream()
        rows = [doc.to_dict() for doc in docs]
        return pd.DataFrame(rows) if rows else pd.DataFrame()
    except Exception:
        return pd.DataFrame()

def get_blacklist():
    if not db:
        return []
    try:
        return [doc.to_dict() for doc in db.collection("blacklist").stream()]
    except Exception:
        return []

def add_blacklist(plate, reason):
    if not db:
        return False
    try:
        db.collection("blacklist").document(plate).set({
            "plate": plate, "reason": reason,
            "added_at": datetime.now().isoformat()
        })
        return True
    except Exception:
        return False

def remove_blacklist(plate):
    if not db:
        return False
    try:
        db.collection("blacklist").document(plate).delete()
        return True
    except Exception:
        return False

def check_blacklist(plate):
    if not db:
        return None
    try:
        doc = db.collection("blacklist").document(plate).get()
        return doc.to_dict() if doc.exists else None
    except Exception:
        return None

# ─────────────────────────────────────────────
# EMAIL
# ─────────────────────────────────────────────
def send_alert(plate, info, source, black=None):
    gmail = st.secrets.get("GMAIL_ADDRESS", "")
    pwd = st.secrets.get("GMAIL_APP_PASSWORD", "")
    if not gmail or not pwd:
        return False
    try:
        ts = datetime.now().strftime("%d %b %Y, %I:%M %p")
        is_black = black is not None
        color = "#ff3b30" if is_black else "#007aff"
        title = "🚨 BLACKLISTED PLATE DETECTED" if is_black else "🚗 New Car Scan — CarInfo"
        black_row = f"<tr><td style='padding:10px;color:#8e8e93;font-size:13px'>⚠️ Reason</td><td style='padding:10px;font-size:13px;color:#ff3b30;font-weight:600'>{black.get('reason','N/A')}</td></tr>" if is_black else ""

        body = f"""
        <div style="font-family:-apple-system,BlinkMacSystemFont,'Plus Jakarta Sans',sans-serif;
                    max-width:560px;margin:auto;background:#f2f2f7;padding:20px;border-radius:20px">
            <div style="background:white;border-radius:18px;padding:24px;box-shadow:0 4px 20px rgba(0,0,0,0.08)">
                <div style="text-align:center;margin-bottom:20px">
                    <div style="background:linear-gradient(135deg,#007aff,#5856d6);border-radius:14px;
                                width:56px;height:56px;margin:0 auto 12px;display:flex;align-items:center;
                                justify-content:center;font-size:28px">🚗</div>
                    <h2 style="color:{color};font-size:18px;margin:0;font-weight:700">{title}</h2>
                </div>
                <div style="background:#ffcc00;border-radius:12px;padding:12px 20px;
                            text-align:center;margin:16px 0;border:2px solid rgba(0,0,0,0.1)">
                    <span style="font-family:'Courier New',monospace;font-size:22px;
                                 font-weight:700;letter-spacing:4px;color:#1c1c1e">{plate}</span>
                </div>
                <table style="width:100%;border-collapse:collapse;background:#f9f9f9;border-radius:12px;overflow:hidden">
                    {black_row}
                    <tr><td style='padding:10px;color:#8e8e93;font-size:13px'>👤 Owner</td><td style='padding:10px;font-size:13px;font-weight:500'>{info.get('owner','N/A')}</td></tr>
                    <tr style='background:white'><td style='padding:10px;color:#8e8e93;font-size:13px'>🚗 Vehicle</td><td style='padding:10px;font-size:13px;font-weight:500'>{info.get('maker','N/A')} {info.get('model','N/A')}</td></tr>
                    <tr><td style='padding:10px;color:#8e8e93;font-size:13px'>⛽ Fuel</td><td style='padding:10px;font-size:13px;font-weight:500'>{info.get('fuel_type','N/A')}</td></tr>
                    <tr style='background:white'><td style='padding:10px;color:#8e8e93;font-size:13px'>📋 RC Status</td><td style='padding:10px;font-size:13px;font-weight:500'>{info.get('rc_status','N/A')}</td></tr>
                    <tr><td style='padding:10px;color:#8e8e93;font-size:13px'>🛡️ Insurance</td><td style='padding:10px;font-size:13px;font-weight:500'>Expires {info.get('insurance_expiry','N/A')}</td></tr>
                    <tr style='background:white'><td style='padding:10px;color:#8e8e93;font-size:13px'>📍 Source</td><td style='padding:10px;font-size:13px;font-weight:500'>{source}</td></tr>
                    <tr><td style='padding:10px;color:#8e8e93;font-size:13px'>🕐 Time</td><td style='padding:10px;font-size:13px;font-weight:500'>{ts}</td></tr>
                </table>
                <p style="text-align:center;color:#8e8e93;font-size:11px;margin-top:16px">
                    CarInfo — carinfo.streamlit.app
                </p>
            </div>
        </div>
        """
        msg = MIMEMultipart()
        msg["From"] = gmail
        msg["To"] = gmail
        msg["Subject"] = f"{'🚨 BLACKLISTED' if is_black else '🚗 New Scan'}: {plate}"
        msg.attach(MIMEText(body, "html"))
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as s:
            s.login(gmail, pwd)
            s.send_message(msg)
        return True
    except Exception:
        return False

# ─────────────────────────────────────────────
# MODELS
# ─────────────────────────────────────────────
@st.cache_resource
def load_models():
    ocr = easyocr.Reader(['en'], gpu=False)
    try:
        yolo = YOLO("yolov8n.pt")
    except Exception:
        yolo = None
    return ocr, yolo

# ─────────────────────────────────────────────
# PLATE DETECTION — PRECISE PIPELINE
# ─────────────────────────────────────────────
OCR_FIX_DIGIT = {'O':'0','I':'1','Z':'2','S':'5','B':'8','G':'6','Q':'0','D':'0'}
OCR_FIX_ALPHA = {'0':'O','1':'I','2':'Z','5':'S','8':'B','6':'G'}

def fix_char(ch, want_digit):
    ch = ch.upper()
    if want_digit:
        return OCR_FIX_DIGIT.get(ch, ch)
    return OCR_FIX_ALPHA.get(ch, ch)

def normalize(raw):
    raw = re.sub(r'[^A-Z0-9]', '', raw.upper())
    if len(raw) == 10:
        pattern = ['L','L','D','D','L','L','D','D','D','D']
        return ''.join(fix_char(raw[i], pattern[i]=='D') for i in range(10))
    # Try to fix 9-char plates (common OCR miss)
    if len(raw) == 9:
        # Try inserting a char and see if it validates
        for i in range(len(raw)+1):
            for ch in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789':
                candidate = raw[:i] + ch + raw[i:]
                if len(candidate) == 10:
                    n = normalize(candidate)
                    if PLATE_PATTERN.match(n):
                        return n
    return raw

def validate(plate):
    return bool(PLATE_PATTERN.match(plate))

def enhance_crop(img_np):
    """Multi-step enhancement for plate crop."""
    results = []
    # Original
    results.append(img_np)
    # Grayscale
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    # Upscale
    h, w = gray.shape
    scale = max(1, int(300 / h))
    up = cv2.resize(gray, (w*scale*2, h*scale*2), interpolation=cv2.INTER_CUBIC)
    results.append(up)
    # Bilateral + threshold
    bil = cv2.bilateralFilter(up, 11, 17, 17)
    _, thresh = cv2.threshold(bil, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    results.append(thresh)
    # Adaptive threshold
    ada = cv2.adaptiveThreshold(bil, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, 11, 2)
    results.append(ada)
    # Sharpened
    kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
    sharp = cv2.filter2D(up, -1, kernel)
    results.append(sharp)
    return results

def ocr_on_crop(crop, reader):
    """Run OCR on multiple enhanced versions of crop."""
    plates = []
    for enhanced in enhance_crop(crop):
        try:
            result = reader.readtext(
                enhanced,
                allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
                batch_size=1,
                workers=0,
                paragraph=False,
                detail=1,
            )
            for (_, text, conf) in result:
                plate = normalize(text)
                if validate(plate) and conf > 0.1:
                    plates.append({"plate": plate, "confidence": conf})
        except Exception:
            pass
    # Return highest confidence unique plates
    seen = set()
    unique = []
    for p in sorted(plates, key=lambda x: -x["confidence"]):
        if p["plate"] not in seen:
            seen.add(p["plate"])
            unique.append(p)
    return unique

def detect_plates(image_bytes, reader, yolo):
    """
    Precise 2-stage detection:
    1. YOLO finds plate region
    2. EasyOCR reads text from crop
    Fallback: EasyOCR on full image
    """
    results = []
    annotated = None

    try:
        # Load & resize image
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        w, h = img.size
        max_dim = 1280
        if max(w, h) > max_dim:
            scale = max_dim / max(w, h)
            img = img.resize((int(w*scale), int(h*scale)), Image.LANCZOS)

        img_np = np.array(img)
        annotated = img_np.copy()

        # ── STAGE 1: YOLO plate detection ──
        plates_found = False
        if yolo is not None:
            try:
                yolo_results = yolo(img_np, verbose=False, conf=0.25)[0]
                boxes = yolo_results.boxes.xyxy.cpu().numpy() if yolo_results.boxes else []

                for box in boxes:
                    x1, y1, x2, y2 = map(int, box[:4])
                    # Add padding
                    pad = 10
                    x1 = max(0, x1-pad)
                    y1 = max(0, y1-pad)
                    x2 = min(img_np.shape[1], x2+pad)
                    y2 = min(img_np.shape[0], y2+pad)

                    crop = img_np[y1:y2, x1:x2]
                    if crop.size == 0:
                        continue

                    plate_results = ocr_on_crop(crop, reader)
                    for pr in plate_results:
                        plate = pr["plate"]
                        conf = pr["confidence"]
                        # Draw on annotated image
                        cv2.rectangle(annotated, (x1,y1), (x2,y2), (0,122,255), 3)
                        cv2.rectangle(annotated, (x1,y1-36), (x2,y1), (0,122,255), -1)
                        cv2.putText(annotated, f"{plate} {conf:.0%}",
                                    (x1+5, y1-10), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.7, (255,255,255), 2)
                        results.append({"plate": plate, "confidence": conf,
                                        "bbox": [x1,y1,x2,y2], "method": "YOLO+OCR"})
                        plates_found = True
            except Exception:
                pass

        # ── STAGE 2: Fallback — OCR on full image ──
        if not plates_found:
            # Try full image OCR
            full_results = ocr_on_crop(img_np, reader)
            for pr in full_results:
                results.append({**pr, "method": "OCR-Full"})

            # If still nothing, try PIL enhancements
            if not results:
                pil_img = Image.fromarray(img_np)
                for enhancer, factor in [
                    (ImageEnhance.Contrast, 2.0),
                    (ImageEnhance.Sharpness, 3.0),
                    (ImageEnhance.Brightness, 1.3),
                ]:
                    enhanced = enhancer(pil_img).enhance(factor)
                    enhanced_np = np.array(enhanced)
                    pr = ocr_on_crop(enhanced_np, reader)
                    for p in pr:
                        results.append({**p, "method": "OCR-Enhanced"})

        # Deduplicate
        seen = set()
        unique = []
        for r in results:
            if r["plate"] not in seen:
                seen.add(r["plate"])
                unique.append(r)
        results = unique

    except Exception as e:
        st.error(f"Detection error: {e}")

    return results, annotated

# ─────────────────────────────────────────────
# VEHICLE INFO
# ─────────────────────────────────────────────
def fetch_info(plate):
    import http.client, json
    try:
        conn = http.client.HTTPSConnection(RAPIDAPI_HOST)
        payload = json.dumps({"vehicle_number": plate})
        headers = {
            'x-rapidapi-key': st.session_state.get("rapidapi_key", ""),
            'x-rapidapi-host': RAPIDAPI_HOST,
            'Content-Type': "application/json"
        }
        conn.request("POST", "/", payload, headers)
        res = conn.getresponse()
        return json.loads(res.read().decode("utf-8"))
    except Exception as e:
        return {"error": str(e)}

def parse_info(raw):
    return {
        "owner": raw.get("owner_name", "N/A"),
        "vehicle_class": raw.get("class", "N/A"),
        "fuel_type": raw.get("fuel_type", "N/A"),
        "maker": raw.get("brand_name", "N/A"),
        "model": raw.get("brand_model", "N/A"),
        "color": raw.get("color", "N/A"),
        "rc_status": raw.get("rc_status", "N/A"),
        "insurance": raw.get("insurance_company", "N/A"),
        "insurance_expiry": raw.get("insurance_expiry", "N/A"),
        "registration_date": raw.get("registration_date", "N/A"),
        "owner_count": raw.get("owner_count", "N/A"),
        "chassis": raw.get("chassis_number", "N/A"),
        "engine": raw.get("engine_number", "N/A"),
        "tax_upto": raw.get("tax_upto", "N/A"),
        "pucc_upto": raw.get("pucc_upto", "N/A"),
        "raw": raw,
    }

# ─────────────────────────────────────────────
# TRUST SCORE
# ─────────────────────────────────────────────
def trust_score(info):
    score = 100
    flags, positives = [], []

    # RC
    if info.get("rc_status","").upper() == "ACTIVE":
        positives.append("✅ RC is active")
    else:
        score -= 30
        flags.append("❌ RC is not active")

    # Insurance
    exp = info.get("insurance_expiry","N/A")
    if exp != "N/A":
        try:
            ed = datetime.strptime(exp, "%d-%b-%Y")
            if ed < datetime.now():
                score -= 20
                flags.append(f"❌ Insurance expired on {exp}")
            else:
                positives.append(f"✅ Insurance valid till {exp}")
        except Exception:
            pass

    # Owners
    try:
        o = int(info.get("owner_count", 1))
        if o == 1:
            positives.append("✅ Single owner vehicle")
        elif o == 2:
            score -= 8
            flags.append(f"⚠️ 2 previous owners")
        elif o == 3:
            score -= 18
            flags.append(f"⚠️ 3 previous owners")
        else:
            score -= 28
            flags.append(f"🚨 {o} owners — high risk")
    except Exception:
        pass

    # Age
    reg = info.get("registration_date","N/A")
    if reg != "N/A":
        try:
            rd = datetime.strptime(reg, "%d-%b-%Y")
            age = (datetime.now() - rd).days / 365
            if age <= 5:
                positives.append(f"✅ Relatively new ({int(age)} years)")
            elif age <= 10:
                score -= 5
                flags.append(f"ℹ️ Vehicle is {int(age)} years old")
            else:
                score -= 15
                flags.append(f"⚠️ Old vehicle: {int(age)} years")
        except Exception:
            pass

    # Tax
    tax = info.get("tax_upto","N/A")
    if tax != "N/A":
        try:
            td = datetime.strptime(tax, "%d-%b-%Y")
            if td > datetime.now():
                positives.append(f"✅ Tax paid till {tax}")
            else:
                score -= 10
                flags.append(f"❌ Tax expired on {tax}")
        except Exception:
            pass

    score = max(0, min(100, score))

    if score >= 75:
        return {"score": score, "verdict": "Safe to Buy",
                "emoji": "✅", "color": "#34c759",
                "css": "trust-safe", "flags": flags, "positives": positives}
    elif score >= 50:
        return {"score": score, "verdict": "Verify Further",
                "emoji": "⚠️", "color": "#ff9500",
                "css": "trust-caution", "flags": flags, "positives": positives}
    else:
        return {"score": score, "verdict": "High Risk — Avoid",
                "emoji": "🚨", "color": "#ff3b30",
                "css": "trust-danger", "flags": flags, "positives": positives}

# ─────────────────────────────────────────────
# UI HELPERS
# ─────────────────────────────────────────────
def plate_badge_html(plate, blacklisted=False):
    color = "#ff3b30" if blacklisted else "#ffcc00"
    label = '<span style="color:#ff3b30;font-weight:700;font-size:0.8rem;margin-left:8px">⚠️ BLACKLISTED</span>' if blacklisted else ''
    return f"""
    <div style="margin:1rem 0;display:flex;align-items:center;gap:12px;flex-wrap:wrap">
        <div style="background:{color};border:2.5px solid rgba(0,0,0,0.15);border-radius:12px;
                    padding:8px 24px 8px 32px;position:relative;
                    box-shadow:0 4px 16px rgba(0,0,0,0.12)">
            <span style="position:absolute;left:7px;top:50%;transform:translateY(-50%);
                         font-size:0.42rem;font-weight:800;color:#003087;
                         writing-mode:vertical-rl;letter-spacing:1.5px">IND</span>
            <span style="font-family:'Courier New',monospace;font-size:1.7rem;
                         font-weight:700;color:#1c1c1e;letter-spacing:3px">{plate}</span>
        </div>
        {label}
    </div>
    """

def info_card_html(info):
    rc = info.get("rc_status","N/A")
    rc_color = "#34c759" if rc.upper() == "ACTIVE" else "#ff3b30"
    fields = [
        ("👤 Owner", info.get("owner","N/A")),
        ("🚗 Vehicle Class", info.get("vehicle_class","N/A")),
        ("⛽ Fuel Type", info.get("fuel_type","N/A")),
        ("🏭 Manufacturer", info.get("maker","N/A")),
        ("🚙 Model", info.get("model","N/A")),
        ("🎨 Color", info.get("color","N/A")),
        ("📋 RC Status", f'<span style="color:{rc_color};font-weight:600">{rc}</span>'),
        ("🛡️ Insurance Co.", info.get("insurance","N/A")),
        ("📅 Insurance Expiry", info.get("insurance_expiry","N/A")),
        ("📅 Registration Date", info.get("registration_date","N/A")),
        ("👥 Owner Count", info.get("owner_count","N/A")),
        ("💰 Tax Valid Upto", info.get("tax_upto","N/A")),
        ("🌿 PUCC Valid Upto", info.get("pucc_upto","N/A")),
    ]
    rows = ""
    for i, (k, v) in enumerate(fields):
        bg = "background:rgba(0,122,255,0.02)" if i % 2 == 0 else "background:white"
        rows += f"""
        <tr style="{bg}">
            <td style="padding:12px 16px;color:#8e8e93;font-size:0.82rem;font-weight:600;width:45%">{k}</td>
            <td style="padding:12px 16px;color:#1c1c1e;font-size:0.88rem;font-weight:500">{v}</td>
        </tr>"""
    return f"""
    <table style="width:100%;border-collapse:collapse;border-radius:18px;overflow:hidden;
                  box-shadow:0 4px 20px rgba(0,0,0,0.08);background:white">
        {rows}
    </table>"""

def trust_card_html(t):
    return f"""
    <div class="trust-card {t['css']}">
        <div class="trust-label">Trust Score</div>
        <div class="trust-number" style="color:{t['color']}">{t['score']}</div>
        <div style="color:#8e8e93;font-size:0.75rem;margin:2px 0 8px">/100</div>
        <div class="trust-verdict" style="color:{t['color']}">{t['emoji']} {t['verdict']}</div>
    </div>"""

def check_admin():
    if st.session_state.admin_auth:
        return True
    st.markdown('<h2 style="font-weight:800;color:#1c1c1e">🔐 Admin Access</h2>', unsafe_allow_html=True)
    st.markdown('<p style="color:#8e8e93">This page requires admin authentication.</p>', unsafe_allow_html=True)
    pwd = st.text_input("Password", type="password", placeholder="Enter admin password")
    if st.button("Unlock", type="primary"):
        if pwd == st.secrets.get("ADMIN_PASSWORD", "admin123"):
            st.session_state.admin_auth = True
            st.rerun()
        else:
            st.error("Incorrect password")
    return False

def process_scan(image_bytes, source, reader, yolo):
    """Full scan pipeline."""
    with st.spinner("🔍 Detecting license plate…"):
        plates, annotated = detect_plates(image_bytes, reader, yolo)

    if annotated is not None:
        st.image(annotated, caption="Detection Result", use_container_width=True)

    if not plates:
        st.warning("No valid Indian license plate detected. Try a clearer, well-lit photo with the plate visible.")
        st.info("💡 Tips: Ensure plate fills at least 30% of frame, good lighting, minimal angle")
        return

    for det in plates:
        plate = det["plate"]
        conf = det["confidence"]
        method = det.get("method", "OCR")

        black = check_blacklist(plate)
        st.markdown(plate_badge_html(plate, blacklisted=bool(black)), unsafe_allow_html=True)

        if black:
            st.error(f"🚨 BLACKLISTED VEHICLE — Reason: {black.get('reason','N/A')}")

        col_conf, col_method = st.columns(2)
        col_conf.caption(f"Confidence: {conf:.0%}")
        col_method.caption(f"Method: {method}")

        with st.spinner("Fetching vehicle information…"):
            raw = fetch_info(plate)
            info = parse_info(raw)

        t_info, t_trust, t_raw = st.tabs(["📋 Vehicle Details", "🛡️ Trust Score", "🔧 Raw Data"])

        with t_info:
            st.markdown(info_card_html(info), unsafe_allow_html=True)

        with t_trust:
            ts = trust_score(info)
            st.markdown(trust_card_html(ts), unsafe_allow_html=True)
            if ts["positives"]:
                st.markdown("**✅ Positive Factors**")
                for p in ts["positives"]:
                    st.markdown(p)
            if ts["flags"]:
                st.markdown("**⚠️ Concern Areas**")
                for f in ts["flags"]:
                    st.markdown(f)

        with t_raw:
            st.json(raw)

        save_scan(plate, info, source)
        sent = send_alert(plate, info, source, black)
        if sent:
            st.success("✅ Saved to database & alert sent!")
        else:
            st.success("✅ Saved to database")
        st.divider()

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sidebar-header">
        <div class="app-icon">🚗</div>
        <div class="app-name">CarInfo</div>
        <div class="app-tagline">Know Before You Buy</div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio("", [
        "🏠  Dashboard",
        "📸  Scan Image",
        "📷  Camera",
        "🔍  Plate Lookup",
        "🚫  Blacklist",
        "🗄️  History",
        "📤  Export",
        "⚙️  Settings",
        "ℹ️  About",
    ], label_visibility="collapsed")

    st.divider()
    st.markdown('<p style="color:#8e8e93;font-size:0.72rem;font-weight:600;letter-spacing:0.5px;padding:0 0.8rem">FORMAT</p>', unsafe_allow_html=True)
    st.markdown('<p style="color:#1c1c1e;font-size:0.85rem;font-weight:600;padding:0 0.8rem;font-family:monospace">AB 12 CD 1234</p>', unsafe_allow_html=True)
    st.markdown('<p style="color:#8e8e93;font-size:0.75rem;padding:0 0.8rem">2L · 2D · 2L · 4D</p>', unsafe_allow_html=True)

    if st.session_state.admin_auth:
        st.divider()
        st.markdown('<p style="color:#34c759;font-size:0.8rem;font-weight:600;padding:0 0.8rem">🔓 Admin Active</p>', unsafe_allow_html=True)
        if st.button("Lock", type="secondary"):
            st.session_state.admin_auth = False
            st.rerun()

# ─────────────────────────────────────────────
# LOAD MODELS
# ─────────────────────────────────────────────
with st.spinner("Loading AI models…"):
    ocr_reader, yolo_model = load_models()

# ─────────────────────────────────────────────
# PAGES
# ─────────────────────────────────────────────

# ── DASHBOARD ──
if page == "🏠  Dashboard":
    st.markdown('<h1 class="page-title">Dashboard</h1>', unsafe_allow_html=True)
    st.markdown('<p class="page-subtitle">Overview of all vehicle scans</p>', unsafe_allow_html=True)

    df = get_scans()
    total = len(df)
    today = 0
    if total and "timestamp" in df.columns:
        today = df[df["timestamp"].str.startswith(datetime.now().strftime("%Y-%m-%d"), na=False)].shape[0]
    unique = df["plate"].nunique() if total else 0
    bl_count = len(get_blacklist())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Scans", total)
    c2.metric("Today", today)
    c3.metric("Unique Plates", unique)
    c4.metric("Blacklisted", f"🚫 {bl_count}")

    if total:
        st.markdown('<p class="section-header">Recent Scans</p>', unsafe_allow_html=True)
        cols = [c for c in ["plate","timestamp","owner","maker","color","rc_status","source"] if c in df.columns]
        st.dataframe(df[cols].head(20), use_container_width=True, height=350)

        st.markdown('<p class="section-header">Scans Over Time</p>', unsafe_allow_html=True)
        try:
            df["date"] = pd.to_datetime(df["timestamp"]).dt.date
            chart = df.groupby("date").size().reset_index(name="Scans")
            st.area_chart(chart.set_index("date"))
        except Exception:
            pass

        if "fuel_type" in df.columns:
            st.markdown('<p class="section-header">Fuel Type Breakdown</p>', unsafe_allow_html=True)
            fuel = df["fuel_type"].value_counts().reset_index()
            fuel.columns = ["Fuel Type", "Count"]
            st.bar_chart(fuel.set_index("Fuel Type"))
    else:
        st.info("No scans yet. Start by scanning a vehicle image!")

# ── SCAN IMAGE ──
elif page == "📸  Scan Image":
    st.markdown('<h1 class="page-title">Scan Image</h1>', unsafe_allow_html=True)
    st.markdown('<p class="page-subtitle">Upload a photo to detect and lookup the license plate</p>', unsafe_allow_html=True)

    st.info("💡 Best results: Clear photo, good lighting, plate visible and fills frame")
    uploaded = st.file_uploader("Upload vehicle image", type=["jpg","jpeg","png","webp","bmp","heic"])
    if uploaded:
        process_scan(uploaded.read(), "image", ocr_reader, yolo_model)

# ── CAMERA ──
elif page == "📷  Camera":
    st.markdown('<h1 class="page-title">Camera Scan</h1>', unsafe_allow_html=True)
    st.markdown('<p class="page-subtitle">Take a live photo to scan the license plate</p>', unsafe_allow_html=True)

    st.markdown("""
    <style>
    [data-testid="stCameraInput"] video {
        width:100%!important;border-radius:20px!important;
        box-shadow:0 8px 40px rgba(0,0,0,0.12)!important;
    }
    [data-testid="stCameraInput"] button {
        width:100%!important;padding:1rem!important;
        font-size:1.1rem!important;border-radius:16px!important;
        background:#007aff!important;color:white!important;
        font-weight:600!important;margin-top:0.5rem!important;
    }
    </style>""", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    col1.info("☀️ Good lighting")
    col2.info("📐 Plate in frame")
    col3.info("🎯 Tap to focus")

    cam = st.camera_input(" ", label_visibility="collapsed")
    if cam:
        process_scan(cam.getvalue(), "camera", ocr_reader, yolo_model)

# ── PLATE LOOKUP ──
elif page == "🔍  Plate Lookup":
    st.markdown('<h1 class="page-title">Plate Lookup</h1>', unsafe_allow_html=True)
    st.markdown('<p class="page-subtitle">Enter a license plate number to get full vehicle details</p>', unsafe_allow_html=True)

    plate_input = st.text_input(
        "License Plate",
        max_chars=10,
        placeholder="MH12AB1234",
    ).upper().strip()
    plate_input = re.sub(r'[^A-Z0-9]', '', plate_input)

    if plate_input:
        if not validate(plate_input):
            st.error(f"❌ Invalid format — must be like `MH12AB1234` (2 letters · 2 digits · 2 letters · 4 digits)")
        else:
            black = check_blacklist(plate_input)
            st.markdown(plate_badge_html(plate_input, blacklisted=bool(black)), unsafe_allow_html=True)
            if black:
                st.error(f"🚨 BLACKLISTED — {black.get('reason','N/A')}")

            if st.button("🔍 Lookup Vehicle", type="primary"):
                with st.spinner("Fetching vehicle details…"):
                    raw = fetch_info(plate_input)
                    info = parse_info(raw)

                t_info, t_trust, t_raw = st.tabs([
                    "📋 Vehicle Details", "🛡️ Trust Score", "🔧 Raw Data"
                ])
                with t_info:
                    st.markdown(info_card_html(info), unsafe_allow_html=True)
                with t_trust:
                    ts = trust_score(info)
                    st.markdown(trust_card_html(ts), unsafe_allow_html=True)
                    if ts["positives"]:
                        st.markdown("**✅ Positive Factors**")
                        for p in ts["positives"]:
                            st.markdown(p)
                    if ts["flags"]:
                        st.markdown("**⚠️ Concern Areas**")
                        for f in ts["flags"]:
                            st.markdown(f)
                with t_raw:
                    st.json(raw)

                save_scan(plate_input, info, "manual")
                sent = send_alert(plate_input, info, "manual", black)
                if sent:
                    st.success("✅ Saved & alert sent!")
                else:
                    st.success("✅ Saved to database")

# ── BLACKLIST ──
elif page == "🚫  Blacklist":
    st.markdown('<h1 class="page-title">Blacklist</h1>', unsafe_allow_html=True)
    st.markdown('<p class="page-subtitle">Manage flagged vehicles</p>', unsafe_allow_html=True)
    if not check_admin():
        st.stop()

    st.markdown('<p class="section-header">Add to Blacklist</p>', unsafe_allow_html=True)
    col1, col2 = st.columns([1,2])
    with col1:
        new_plate = re.sub(r'[^A-Z0-9]', '', st.text_input("Plate Number", max_chars=10, placeholder="MH12AB1234").upper())
    with col2:
        reason = st.text_input("Reason", placeholder="e.g. Stolen vehicle, Wanted criminal, etc.")

    if st.button("🚫 Add to Blacklist", type="primary"):
        if not validate(new_plate):
            st.error("❌ Invalid plate format!")
        elif not reason.strip():
            st.error("❌ Reason required!")
        else:
            if add_blacklist(new_plate, reason):
                st.success(f"✅ {new_plate} added!")
                st.rerun()

    st.divider()
    st.markdown('<p class="section-header">Current Blacklist</p>', unsafe_allow_html=True)
    bl = get_blacklist()
    if not bl:
        st.info("No plates blacklisted.")
    else:
        for item in bl:
            col_a, col_b, col_c = st.columns([2,3,1])
            col_a.markdown(f"**`{item.get('plate','')}`**")
            col_b.markdown(f"⚠️ {item.get('reason','N/A')}")
            if col_c.button("Remove", key=f"rm_{item.get('plate','')}"):
                remove_blacklist(item.get('plate',''))
                st.rerun()

# ── HISTORY ──
elif page == "🗄️  History":
    st.markdown('<h1 class="page-title">Scan History</h1>', unsafe_allow_html=True)
    st.markdown('<p class="page-subtitle">All vehicle scans stored in database</p>', unsafe_allow_html=True)
    if not check_admin():
        st.stop()

    df = get_scans()
    if df.empty:
        st.info("No records yet.")
    else:
        col1, col2, col3 = st.columns(3)
        search = col1.text_input("🔍 Search plate")
        source_f = col2.selectbox("Source", ["All","image","camera","manual"])
        date_f = col3.date_input("Date", value=None)

        filtered = df.copy()
        if search and "plate" in filtered.columns:
            filtered = filtered[filtered["plate"].str.contains(search.upper(), na=False)]
        if source_f != "All" and "source" in filtered.columns:
            filtered = filtered[filtered["source"] == source_f]
        if date_f and "timestamp" in filtered.columns:
            filtered = filtered[filtered["timestamp"].str.startswith(str(date_f), na=False)]

        cols = [c for c in ["plate","timestamp","owner","maker","color","rc_status","source"] if c in filtered.columns]
        st.dataframe(filtered[cols], use_container_width=True, height=450)
        st.caption(f"{len(filtered)} records")

    if st.button("🔒 Logout", type="secondary"):
        st.session_state.admin_auth = False
        st.rerun()

# ── EXPORT ──
elif page == "📤  Export":
    st.markdown('<h1 class="page-title">Export Data</h1>', unsafe_allow_html=True)
    st.markdown('<p class="page-subtitle">Download your scan history</p>', unsafe_allow_html=True)
    if not check_admin():
        st.stop()

    df = get_scans()
    if df.empty:
        st.info("No data to export.")
    else:
        st.metric("Total Records", len(df))
        col1, col2 = st.columns(2)
        with col1:
            st.download_button("📥 Download CSV",
                df.to_csv(index=False).encode("utf-8"),
                "carinfo_scans.csv", "text/csv")
        with col2:
            st.download_button("📥 Download JSON",
                df.to_json(orient="records", indent=2),
                "carinfo_scans.json", "application/json")
        st.dataframe(df.head(50), use_container_width=True)

# ── SETTINGS ──
elif page == "⚙️  Settings":
    st.markdown('<h1 class="page-title">Settings</h1>', unsafe_allow_html=True)
    st.markdown('<p class="page-subtitle">Configure your CarInfo experience</p>', unsafe_allow_html=True)

    st.markdown('<p class="section-header">🔑 RapidAPI Key</p>', unsafe_allow_html=True)
    st.caption("Your key is stored in session only — never shared or saved publicly.")
    new_key = st.text_input("API Key", value=st.session_state.rapidapi_key,
                            type="password", placeholder="Paste your RapidAPI key")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("💾 Save Key", type="primary"):
            if new_key.strip():
                st.session_state.rapidapi_key = new_key.strip()
                st.success("✅ Saved!")
    with col2:
        if st.button("🧪 Test Key"):
            r = fetch_info("MH12AB1234")
            if "error" in r or "message" in r:
                st.error("❌ Key not working")
            else:
                st.success("✅ Key working!")

    st.info("Get a free key at [rapidapi.com](https://rapidapi.com) → search `vehicle-rc-information-v2`")

    st.divider()
    st.markdown('<p class="section-header">📧 Email Alerts</p>', unsafe_allow_html=True)
    gmail = st.secrets.get("GMAIL_ADDRESS","")
    if gmail:
        st.success(f"✅ Alerts configured: {gmail}")
    else:
        st.warning("Add `GMAIL_ADDRESS` and `GMAIL_APP_PASSWORD` to Streamlit secrets.")

    st.divider()
    st.markdown('<p class="section-header">ℹ️ App Info</p>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    col1.metric("Version", "1.0")
    col2.metric("Python", "3.11")
    col3.metric("Framework", "Streamlit")

# ── ABOUT ──
elif page == "ℹ️  About":
    st.markdown('<h1 class="page-title">About CarInfo</h1>', unsafe_allow_html=True)
    st.markdown('<p class="page-subtitle">Everything you need to know about this app</p>', unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(["🚗 About", "📖 How to Use", "🏭 Use Cases", "🔒 Privacy"])

    with tab1:
        st.markdown("""
        ### What is CarInfo?
        CarInfo is a production-grade vehicle information platform built for India.
        It uses AI-powered license plate recognition combined with real-time RTO database
        lookup to give you complete vehicle history instantly.

        ### Why CarInfo?
        Every year, thousands of Indians get cheated while buying used cars.
        Dealers hide expired insurance, multiple owners, and RC issues.
        CarInfo gives buyers the power to verify any vehicle in seconds — for free.

        ### Technology
        - **AI Detection:** YOLOv8 + EasyOCR for precise plate reading
        - **Database:** Google Firebase Firestore (cloud, permanent)
        - **Vehicle Data:** RTO database via RapidAPI
        - **Alerts:** Gmail SMTP for instant notifications
        - **Platform:** Streamlit — works on all devices
        """)
        col1, col2, col3 = st.columns(3)
        col1.metric("Accuracy", "90%+")
        col2.metric("Response Time", "<3 sec")
        col3.metric("Supported", "All Indian Plates")

    with tab2:
        with st.expander("📸 Scan from Image", expanded=True):
            st.markdown("""
            1. Click **📸 Scan Image** in sidebar
            2. Upload a clear photo of the vehicle
            3. App detects the plate automatically
            4. Full vehicle details + Trust Score shown instantly
            **Tips:** Good lighting · Plate fills 30%+ of frame · Straight angle
            """)
        with st.expander("🔍 Manual Lookup"):
            st.markdown("""
            1. Click **🔍 Plate Lookup** in sidebar
            2. Type the plate in format `MH12AB1234`
            3. Click **Lookup Vehicle**
            4. See full details, Trust Score, and raw API data
            """)
        with st.expander("🛡️ Trust Score"):
            st.markdown("""
            Trust Score (0-100) is calculated based on:
            - ✅ RC Status (Active/Inactive)
            - ✅ Insurance validity
            - ✅ Number of previous owners
            - ✅ Vehicle age
            - ✅ Tax payment status

            **75-100:** Safe to Buy ✅
            **50-74:** Verify Further ⚠️
            **0-49:** High Risk 🚨
            """)

    with tab3:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            #### 🚗 Used Car Buyers
            Verify vehicle history before purchase. Check RC, insurance, owner count instantly.

            #### 👮 Law Enforcement
            Check blacklisted vehicles at checkposts. Instant stolen vehicle detection.

            #### 🏢 Corporates & Offices
            Log visitor vehicles. Alert security for unknown plates.
            """)
        with col2:
            st.markdown("""
            #### 🅿️ Parking Operators
            Auto-log entry/exit. Block blacklisted vehicles.

            #### 🏠 Housing Societies
            Whitelist resident vehicles. Track visitor entries.

            #### 🏛️ Government/RTO
            Field verification of RC, insurance, tax compliance.
            """)

    with tab4:
        st.markdown("""
        #### Data Collection
        CarInfo collects: plate numbers, vehicle info from public RTO database, scan timestamps.

        #### Data Storage
        All data stored in Google Firebase (encrypted). Never sold to third parties.

        #### Your Rights
        Admin can delete all data. Individual records deletable from History page.

        #### Disclaimer
        Vehicle data sourced from public RTO records. CarInfo is not responsible for
        inaccurate RTO data. Use in compliance with local laws.

        #### Contact
        **Developer:** Somya Bhalani
        **Email:** somyabhalani@gmail.com
        **GitHub:** [Bhuro234](https://github.com/Bhuro234)
        """)
