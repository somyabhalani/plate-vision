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
from PIL import Image
import io
import firebase_admin
from firebase_admin import credentials, firestore

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="ANPR Pro — India",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# INJECT CSS
# ─────────────────────────────────────────────
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
RAPIDAPI_HOST = "vehicle-rc-information-v2.p.rapidapi.com"
PLATE_PATTERN = re.compile(r'^[A-Z]{2}[0-9]{2}[A-Z]{2}[0-9]{4}$')

# Session state defaults
if "rapidapi_key" not in st.session_state:
    st.session_state.rapidapi_key = st.secrets.get("RAPIDAPI_KEY", "")
if "admin_authenticated" not in st.session_state:
    st.session_state.admin_authenticated = False

# ─────────────────────────────────────────────
# FIREBASE INIT
# ─────────────────────────────────────────────
@st.cache_resource
def init_firebase():
    try:
        if not firebase_admin._apps:
            fb_cfg = st.secrets.get("firebase", None)
            if fb_cfg:
                cred = credentials.Certificate(dict(fb_cfg))
                firebase_admin.initialize_app(cred)
        return firestore.client()
    except Exception as e:
        st.warning(f"Firebase not connected: {e}")
        return None

db = init_firebase()

# ─────────────────────────────────────────────
# FIREBASE DB FUNCTIONS
# ─────────────────────────────────────────────
def save_scan(plate, info, source="image", speed=None):
    if not db:
        return
    import json
    ts = datetime.now().isoformat()
    try:
        db.collection("scans").add({
            "plate": plate,
            "timestamp": ts,
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
            "speed_kmh": speed,
        })
    except Exception as e:
        st.error(f"Error saving scan: {e}")

def get_all_scans():
    if not db:
        return pd.DataFrame()
    try:
        docs = db.collection("scans").order_by(
            "timestamp", direction=firestore.Query.DESCENDING
        ).limit(500).stream()
        rows = [doc.to_dict() for doc in docs]
        return pd.DataFrame(rows) if rows else pd.DataFrame()
    except Exception as e:
        st.error(f"Error fetching scans: {e}")
        return pd.DataFrame()

def get_blacklist():
    if not db:
        return []
    try:
        docs = db.collection("blacklist").stream()
        return [doc.to_dict() for doc in docs]
    except Exception:
        return []

def add_to_blacklist(plate, reason):
    if not db:
        return False
    try:
        db.collection("blacklist").document(plate).set({
            "plate": plate,
            "reason": reason,
            "added_at": datetime.now().isoformat(),
        })
        return True
    except Exception as e:
        st.error(f"Error adding to blacklist: {e}")
        return False

def remove_from_blacklist(plate):
    if not db:
        return False
    try:
        db.collection("blacklist").document(plate).delete()
        return True
    except Exception:
        return False

def is_blacklisted(plate):
    if not db:
        return None
    try:
        doc = db.collection("blacklist").document(plate).get()
        if doc.exists:
            return doc.to_dict()
        return None
    except Exception:
        return None

# ─────────────────────────────────────────────
# EMAIL ALERTS
# ─────────────────────────────────────────────
def send_email_alert(subject, body):
    gmail = st.secrets.get("GMAIL_ADDRESS", "")
    app_pwd = st.secrets.get("GMAIL_APP_PASSWORD", "")
    if not gmail or not app_pwd:
        return False
    try:
        msg = MIMEMultipart()
        msg["From"] = gmail
        msg["To"] = gmail
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "html"))
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(gmail, app_pwd)
            server.send_message(msg)
        return True
    except Exception as e:
        st.warning(f"Email not sent: {e}")
        return False

def send_scan_alert(plate, info, source, blacklist_info=None):
    ts = datetime.now().strftime("%d-%b-%Y %I:%M %p")
    is_black = blacklist_info is not None
    color = "#ff3b5c" if is_black else "#00e5ff"
    title = "🚨 BLACKLISTED PLATE DETECTED" if is_black else "🚗 New Plate Scanned"
    reason = f"<p><b>⚠️ Blacklist Reason:</b> {blacklist_info.get('reason','N/A')}</p>" if is_black else ""

    body = f"""
    <div style="font-family:Arial;max-width:600px;margin:auto;background:#0a0e1a;color:#e8eaf0;padding:24px;border-radius:12px;border:2px solid {color}">
        <h2 style="color:{color}">{title}</h2>
        <div style="background:#ffd60a;border-radius:8px;padding:12px 24px;display:inline-block;margin:12px 0">
            <span style="font-family:monospace;font-size:24px;font-weight:bold;color:#111;letter-spacing:4px">{plate}</span>
        </div>
        {reason}
        <table style="width:100%;border-collapse:collapse;margin-top:16px">
            <tr><td style="padding:8px;color:#6b7280">👤 Owner</td><td style="padding:8px">{info.get('owner','N/A')}</td></tr>
            <tr><td style="padding:8px;color:#6b7280">🚗 Vehicle</td><td style="padding:8px">{info.get('maker','N/A')} {info.get('model','N/A')}</td></tr>
            <tr><td style="padding:8px;color:#6b7280">⛽ Fuel</td><td style="padding:8px">{info.get('fuel_type','N/A')}</td></tr>
            <tr><td style="padding:8px;color:#6b7280">🎨 Color</td><td style="padding:8px">{info.get('color','N/A')}</td></tr>
            <tr><td style="padding:8px;color:#6b7280">📋 RC Status</td><td style="padding:8px">{info.get('rc_status','N/A')}</td></tr>
            <tr><td style="padding:8px;color:#6b7280">🛡️ Insurance</td><td style="padding:8px">{info.get('insurance','N/A')} (expires {info.get('insurance_expiry','N/A')})</td></tr>
            <tr><td style="padding:8px;color:#6b7280">📍 Source</td><td style="padding:8px">{source}</td></tr>
            <tr><td style="padding:8px;color:#6b7280">🕐 Time</td><td style="padding:8px">{ts}</td></tr>
        </table>
        <p style="color:#6b7280;font-size:12px;margin-top:24px">ANPR Pro — somya-anpr-one.streamlit.app</p>
    </div>
    """
    subject = f"{'🚨 BLACKLISTED' if is_black else '🚗 New Scan'}: {plate} — {ts}"
    return send_email_alert(subject, body)

# ─────────────────────────────────────────────
# OCR MODEL
# ─────────────────────────────────────────────
@st.cache_resource
def load_ocr():
    return easyocr.Reader(['en'], gpu=False)

# ─────────────────────────────────────────────
# PLATE UTILS
# ─────────────────────────────────────────────
def fix_ocr_char(char, expect_digit):
    char = char.upper()
    if expect_digit:
        corrections = {'O':'0','I':'1','Z':'2','S':'5','B':'8','G':'6','Q':'0','D':'0'}
        return corrections.get(char, char)
    else:
        rev = {'0':'O','1':'I','2':'Z','5':'S','8':'B','6':'G'}
        return rev.get(char, char)

def normalize_plate(raw: str) -> str:
    raw = re.sub(r'[^A-Z0-9]', '', raw.upper())
    if len(raw) == 10:
        expected = ['L','L','D','D','L','L','D','D','D','D']
        corrected = ''
        for i, ch in enumerate(raw):
            corrected += fix_ocr_char(ch, expected[i] == 'D')
        return corrected
    return raw

def validate_plate(plate: str) -> bool:
    return bool(PLATE_PATTERN.match(plate))

def preprocess_plate(img):
    try:
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            gray = img.copy()
        h, w = gray.shape
        if h < 100:
            gray = cv2.resize(gray, (w*3, h*3), interpolation=cv2.INTER_CUBIC)
        gray = cv2.bilateralFilter(gray, 11, 17, 17)
        gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY, 11, 2)
        kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
        gray = cv2.filter2D(gray, -1, kernel)
        return gray
    except Exception:
        return img

def detect_plates(image_bytes, reader):
    results = []
    annotated = None
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        # Resize large images to max 1024px to prevent crash
        w, h = img.size
        max_dim = 1024
        if max(w, h) > max_dim:
            scale = max_dim / max(w, h)
            img = img.resize((int(w*scale), int(h*scale)), Image.LANCZOS)

        img_np = np.array(img)
        annotated = img_np.copy()

        # Try 1: OCR on full image
        ocr_result = reader.readtext(
            img_np,
            allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
            batch_size=1,
            workers=0,
            paragraph=False,
        )
        for (bbox, text, conf) in ocr_result:
            if conf < 0.15:
                continue
            plate = normalize_plate(text)
            if validate_plate(plate):
                pts = np.array(bbox, dtype=np.int32)
                cv2.polylines(annotated, [pts], True, (0, 255, 80), 3)
                x, y = pts[0]
                cv2.putText(annotated, plate, (x, max(y-10, 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 80), 2)
                results.append({"plate": plate, "confidence": conf})

        # Try 2: preprocessed image if nothing found
        if not results:
            gray = preprocess_plate(img_np)
            ocr_result2 = reader.readtext(
                gray,
                allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
                batch_size=1,
                workers=0,
            )
            for (bbox, text, conf) in ocr_result2:
                plate = normalize_plate(text)
                if validate_plate(plate):
                    results.append({"plate": plate, "confidence": conf})

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
# RAPIDAPI LOOKUP
# ─────────────────────────────────────────────
def fetch_vehicle_info(plate: str) -> dict:
    import http.client
    import json
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
        data = res.read()
        return json.loads(data.decode("utf-8"))
    except Exception as e:
        return {"error": str(e)}

def parse_vehicle_info(raw: dict) -> dict:
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
        "raw": raw,
    }

# ─────────────────────────────────────────────
# ADMIN AUTH
# ─────────────────────────────────────────────
def check_admin():
    if st.session_state.admin_authenticated:
        return True
    st.markdown('<h2 class="page-title">🔐 Admin Access Required</h2>', unsafe_allow_html=True)
    st.warning("This page is restricted to admins only.")
    pwd = st.text_input("Enter Admin Password", type="password", placeholder="Password")
    if st.button("🔓 Login", type="primary"):
        correct = st.secrets.get("ADMIN_PASSWORD", "admin123")
        if pwd == correct:
            st.session_state.admin_authenticated = True
            st.rerun()
        else:
            st.error("❌ Wrong password. Try again.")
    return False

# ─────────────────────────────────────────────
# UI HELPERS
# ─────────────────────────────────────────────
def plate_badge(plate, is_blacklisted=False):
    color = "#ff3b5c" if is_blacklisted else "#ffd60a"
    label = "⚠️ BLACKLISTED" if is_blacklisted else ""
    return f"""
    <div style="display:inline-flex;flex-direction:column;align-items:center;margin:1rem 0">
        <div style="background:{color};border:3px solid #111;border-radius:8px;padding:0.5rem 1.5rem;
                    box-shadow:0 4px 20px rgba(255,214,10,0.4)">
            <span style="font-family:'Courier New',monospace;font-size:2rem;font-weight:bold;
                         color:#111;letter-spacing:4px">{plate}</span>
        </div>
        {f'<span style="color:#ff3b5c;font-weight:bold;margin-top:4px">{label}</span>' if is_blacklisted else ''}
    </div>
    """

def info_card(info):
    fields = [
        ("👤 Owner", info.get("owner","N/A")),
        ("🚗 Class", info.get("vehicle_class","N/A")),
        ("⛽ Fuel", info.get("fuel_type","N/A")),
        ("🏭 Maker", info.get("maker","N/A")),
        ("🚙 Model", info.get("model","N/A")),
        ("🎨 Color", info.get("color","N/A")),
        ("📋 RC Status", info.get("rc_status","N/A")),
        ("🛡️ Insurance", info.get("insurance","N/A")),
        ("📅 Ins. Expiry", info.get("insurance_expiry","N/A")),
        ("📅 Reg. Date", info.get("registration_date","N/A")),
        ("👥 Owner Count", info.get("owner_count","N/A")),
    ]
    rows = "".join(f'<tr><td class="info-label">{k}</td><td class="info-val">{v}</td></tr>' for k,v in fields)
    return f'<table class="info-table">{rows}</table>'

def process_and_show(image_bytes, source="image"):
    """Process image, show results, check blacklist, send alert."""
    with st.spinner("🔍 Detecting plates…"):
        plates, annotated = detect_plates(image_bytes, ocr_reader)

    if annotated is not None:
        st.image(annotated, caption="Detection Result", use_container_width=True)

    if not plates:
        st.warning("⚠️ No valid plate detected. Try a clearer, well-lit image with plate filling the frame.")
        return

    for det in plates:
        plate = det["plate"]
        conf = det["confidence"]

        # Check blacklist
        black_info = is_blacklisted(plate)
        st.markdown(plate_badge(plate, is_blacklisted=bool(black_info)), unsafe_allow_html=True)

        if black_info:
            st.error(f"🚨 **BLACKLISTED VEHICLE!** Reason: {black_info.get('reason','N/A')}")

        st.caption(f"Confidence: {conf:.0%}")

        with st.spinner("Fetching vehicle info…"):
            raw = fetch_vehicle_info(plate)
            info = parse_vehicle_info(raw)

        st.markdown(info_card(info), unsafe_allow_html=True)
        save_scan(plate, info, source=source)

        # Send email alert
        with st.spinner("Sending alert…"):
            sent = send_scan_alert(plate, info, source, blacklist_info=black_info)
            if sent:
                st.success("✅ Saved & alert sent!")
            else:
                st.success("✅ Saved to database")

        if black_info:
            with st.expander("Raw API Response"):
                st.json(raw)

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-logo">🚗 ANPR Pro</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-sub">India License Plate Recognition</div>', unsafe_allow_html=True)
    st.divider()
    page = st.radio("Navigate", [
        "🏠 Dashboard",
        "📸 Image / Camera",
        "🎥 Video & Speed",
        "🔍 Manual Lookup",
        "🚫 Blacklist",
        "🗄️ Database",
        "📤 Export",
        "⚙️ Settings",
    ], label_visibility="collapsed")
    st.divider()
    st.markdown("**Format:** `AB12CD1234`")
    st.markdown("2 letters · 2 digits · 2 letters · 4 digits")
    if st.session_state.admin_authenticated:
        st.success("🔓 Admin logged in")
        if st.button("🔒 Logout"):
            st.session_state.admin_authenticated = False
            st.rerun()

# ─────────────────────────────────────────────
# LOAD OCR MODEL
# ─────────────────────────────────────────────
with st.spinner("Loading OCR model…"):
    ocr_reader = load_ocr()

# ─────────────────────────────────────────────
# PAGES
# ─────────────────────────────────────────────

# ── DASHBOARD ──
if page == "🏠 Dashboard":
    st.markdown('<h1 class="page-title">Dashboard</h1>', unsafe_allow_html=True)
    df = get_all_scans()
    total = len(df)
    today = 0
    if total and "timestamp" in df.columns:
        today = df[df["timestamp"].str.startswith(datetime.now().strftime("%Y-%m-%d"), na=False)].shape[0]
    unique = df["plate"].nunique() if total else 0
    blacklist_count = len(get_blacklist())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Scans", total)
    c2.metric("Today", today)
    c3.metric("Unique Plates", unique)
    c4.metric("Blacklisted", f"🚫 {blacklist_count}")

    if total:
        st.markdown("### Recent Scans")
        display_cols = [c for c in ["plate","timestamp","owner","maker","color","source","speed_kmh"] if c in df.columns]
        st.dataframe(df[display_cols].head(20), use_container_width=True)
        st.markdown("### Scans Over Time")
        try:
            df["date"] = pd.to_datetime(df["timestamp"]).dt.date
            chart = df.groupby("date").size().reset_index(name="count")
            st.line_chart(chart.set_index("date"))
        except Exception:
            pass
    else:
        st.info("No scans yet. Start by scanning an image or video!")

# ── IMAGE / CAMERA ──
elif page == "📸 Image / Camera":
    st.markdown('<h1 class="page-title">Image & Camera Scan</h1>', unsafe_allow_html=True)
    tab1, tab2 = st.tabs(["📁 Upload Image", "📷 Camera Capture"])

    with tab1:
        st.info("💡 Tips: Good lighting, plate fills 40%+ of frame, hold camera steady")
        uploaded = st.file_uploader("Upload image", type=["jpg","jpeg","png","webp","bmp"])
        if uploaded:
            process_and_show(uploaded.read(), source="image")

    with tab2:
        st.markdown("""
        <style>
        [data-testid="stCameraInput"] video { width:100%!important;max-height:70vh!important;border-radius:12px!important; }
        [data-testid="stCameraInput"] { width:100%!important; }
        [data-testid="stCameraInput"] button { width:100%!important;padding:1rem!important;font-size:1.2rem!important; }
        </style>""", unsafe_allow_html=True)
        st.info("💡 Tip: Tap the plate area on screen before capturing for better focus")
        cam = st.camera_input(" ", label_visibility="collapsed")
        if cam:
            process_and_show(cam.getvalue(), source="camera")

# ── VIDEO & SPEED ──
elif page == "🎥 Video & Speed":
    st.markdown('<h1 class="page-title">Video Analysis & Speed</h1>', unsafe_allow_html=True)
    st.info("Upload a video to detect plates. Speed estimation uses plate position tracking.")

    col1, col2 = st.columns([2,1])
    with col1:
        video_file = st.file_uploader("Upload video", type=["mp4","avi","mov","mkv"])
    with col2:
        speed_limit = st.number_input("Speed limit (km/h)", 20, 200, 80)

    if video_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(video_file.read())
            tmp_path = tmp.name

        st.video(tmp_path)

        if st.button("🚀 Analyse Video", type="primary"):
            cap = cv2.VideoCapture(tmp_path)
            fps = cap.get(cv2.CAP_PROP_FPS) or 30
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
            plate_frames = {}
            frame_no = 0
            progress = st.progress(0, text="Analysing video…")

            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_no % 10 == 0:
                    _, frame_bytes = cv2.imencode('.jpg', frame)
                    detections, _ = detect_plates(frame_bytes.tobytes(), ocr_reader)
                    for d in detections:
                        p = d["plate"]
                        if p not in plate_frames:
                            plate_frames[p] = []
                        plate_frames[p].append(frame_no)
                progress.progress(min(frame_no / total_frames, 1.0))
                frame_no += 1

            cap.release()
            os.unlink(tmp_path)
            progress.empty()

            if not plate_frames:
                st.warning("No plates detected in video.")
            else:
                st.markdown("### Results")
                for plate, frames in plate_frames.items():
                    elapsed_s = (frames[-1] - frames[0]) / fps if len(frames) > 1 else 0
                    speed = round((10 / elapsed_s) * 3.6, 1) if elapsed_s > 0 else 0

                    black_info = is_blacklisted(plate)
                    col_a, col_b = st.columns([1,1])
                    with col_a:
                        st.markdown(plate_badge(plate, is_blacklisted=bool(black_info)), unsafe_allow_html=True)
                        if black_info:
                            st.error(f"🚨 BLACKLISTED: {black_info.get('reason','N/A')}")
                    with col_b:
                        icon = "🔴" if speed > speed_limit else "🟢"
                        st.metric("Speed", f"{icon} {speed} km/h")
                        if speed > speed_limit:
                            st.error(f"⚠️ OVERSPEED! Limit: {speed_limit} km/h")

                    with st.spinner("Fetching info…"):
                        raw = fetch_vehicle_info(plate)
                        info = parse_vehicle_info(raw)
                    st.markdown(info_card(info), unsafe_allow_html=True)
                    save_scan(plate, info, source="video", speed=speed)
                    send_scan_alert(plate, info, "video", blacklist_info=black_info)
                    st.divider()

# ── MANUAL LOOKUP ──
elif page == "🔍 Manual Lookup":
    st.markdown('<h1 class="page-title">Manual Plate Lookup</h1>', unsafe_allow_html=True)
    st.markdown("Enter plate in format **`AB12CD1234`** (2 letters · 2 digits · 2 letters · 4 digits)")

    raw_input = st.text_input("License Plate Number", max_chars=10, placeholder="MH12AB1234").upper().strip()
    raw_input = re.sub(r'[^A-Z0-9]', '', raw_input)

    if raw_input:
        if not validate_plate(raw_input):
            st.error(f"❌ Invalid format. Got: `{raw_input}` — must be like `MH12AB1234`")
        else:
            black_info = is_blacklisted(raw_input)
            st.markdown(plate_badge(raw_input, is_blacklisted=bool(black_info)), unsafe_allow_html=True)
            if black_info:
                st.error(f"🚨 **BLACKLISTED!** Reason: {black_info.get('reason','N/A')}")

            if st.button("🔍 Lookup Vehicle", type="primary"):
                with st.spinner("Querying RapidAPI…"):
                    raw = fetch_vehicle_info(raw_input)
                    info = parse_vehicle_info(raw)
                st.markdown(info_card(info), unsafe_allow_html=True)
                save_scan(raw_input, info, source="manual")
                sent = send_scan_alert(raw_input, info, "manual", blacklist_info=black_info)
                if sent:
                    st.success("✅ Saved & alert sent!")
                else:
                    st.success("✅ Saved to database")
                with st.expander("Raw API Response"):
                    st.json(raw)

# ── BLACKLIST ──
elif page == "🚫 Blacklist":
    st.markdown('<h1 class="page-title">Blacklist Manager</h1>', unsafe_allow_html=True)
    if not check_admin():
        st.stop()

    st.markdown("### Add Plate to Blacklist")
    col1, col2 = st.columns([1,2])
    with col1:
        new_plate = st.text_input("Plate Number", max_chars=10, placeholder="MH12AB1234").upper().strip()
        new_plate = re.sub(r'[^A-Z0-9]', '', new_plate)
    with col2:
        reason = st.text_input("Reason", placeholder="e.g. Stolen vehicle, Criminal record, etc.")

    if st.button("🚫 Add to Blacklist", type="primary"):
        if not validate_plate(new_plate):
            st.error("❌ Invalid plate format!")
        elif not reason.strip():
            st.error("❌ Please provide a reason!")
        else:
            if add_to_blacklist(new_plate, reason):
                st.success(f"✅ `{new_plate}` added to blacklist!")
                st.rerun()

    st.divider()
    st.markdown("### Current Blacklist")
    blacklist = get_blacklist()
    if not blacklist:
        st.info("No plates blacklisted yet.")
    else:
        for item in blacklist:
            col_a, col_b, col_c = st.columns([2,3,1])
            with col_a:
                st.markdown(f"**`{item.get('plate','')}`**")
            with col_b:
                st.markdown(f"⚠️ {item.get('reason','N/A')}")
            with col_c:
                if st.button("🗑️ Remove", key=f"del_{item.get('plate','')}"):
                    if remove_from_blacklist(item.get('plate','')):
                        st.success("Removed!")
                        st.rerun()

# ── DATABASE ──
elif page == "🗄️ Database":
    st.markdown('<h1 class="page-title">Scan Database</h1>', unsafe_allow_html=True)
    if not check_admin():
        st.stop()

    df = get_all_scans()
    if df.empty:
        st.info("No records yet.")
    else:
        col1, col2, col3 = st.columns(3)
        search = col1.text_input("🔍 Search plate")
        source_filter = col2.selectbox("Source", ["All","image","camera","video","manual"])
        date_filter = col3.date_input("Date", value=None)

        filtered = df.copy()
        if search and "plate" in filtered.columns:
            filtered = filtered[filtered["plate"].str.contains(search.upper(), na=False)]
        if source_filter != "All" and "source" in filtered.columns:
            filtered = filtered[filtered["source"] == source_filter]
        if date_filter and "timestamp" in filtered.columns:
            filtered = filtered[filtered["timestamp"].str.startswith(str(date_filter), na=False)]

        display_cols = [c for c in ["plate","timestamp","owner","maker","color","rc_status","source","speed_kmh"] if c in filtered.columns]
        st.dataframe(filtered[display_cols], use_container_width=True, height=450)
        st.caption(f"{len(filtered)} records")

    st.divider()
    if st.button("🔒 Logout Admin", type="secondary"):
        st.session_state.admin_authenticated = False
        st.rerun()

# ── EXPORT ──
elif page == "📤 Export":
    st.markdown('<h1 class="page-title">Export Data</h1>', unsafe_allow_html=True)
    if not check_admin():
        st.stop()

    df = get_all_scans()
    if df.empty:
        st.info("No data to export.")
    else:
        st.markdown(f"**{len(df)} records** available for export.")
        col1, col2 = st.columns(2)
        with col1:
            st.download_button("📥 Download CSV",
                               df.to_csv(index=False).encode("utf-8"),
                               "anpr_scans.csv", "text/csv")
        with col2:
            st.download_button("📥 Download JSON",
                               df.to_json(orient="records", indent=2),
                               "anpr_scans.json", "application/json")
        st.dataframe(df.head(50), use_container_width=True)

# ── SETTINGS ──
elif page == "⚙️ Settings":
    st.markdown('<h1 class="page-title">Settings</h1>', unsafe_allow_html=True)

    st.markdown("### 🔑 RapidAPI Key")
    st.caption("Each user can enter their own key — stored in session only.")
    new_key = st.text_input("RapidAPI Key", value=st.session_state.rapidapi_key,
                            type="password", placeholder="Paste your RapidAPI key here")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("💾 Save Key", type="primary"):
            if new_key.strip():
                st.session_state.rapidapi_key = new_key.strip()
                st.success("✅ Key saved!")
            else:
                st.error("Please enter a valid key.")
    with col2:
        if st.button("🧪 Test Key"):
            test = fetch_vehicle_info("MH12AB1234")
            if "error" in test or ("message" in test and "failed" in str(test.get("message","")).lower()):
                st.error("❌ Key not working!")
            else:
                st.success("✅ Key working!")

    st.info("💡 Get free key at [rapidapi.com](https://rapidapi.com) → search `vehicle-rc-information-v2`")

    st.divider()
    st.markdown("### 📧 Email Alerts")
    gmail = st.secrets.get("GMAIL_ADDRESS", "")
    if gmail:
        st.success(f"✅ Gmail configured: {gmail}")
    else:
        st.warning("⚠️ Gmail not configured. Add `GMAIL_ADDRESS` and `GMAIL_APP_PASSWORD` to Streamlit secrets.")

    st.divider()
    st.markdown("### ℹ️ About")
    st.info("ANPR Pro v2.0 — Indian License Plate Recognition\nBuilt with EasyOCR + Streamlit + Firebase")
