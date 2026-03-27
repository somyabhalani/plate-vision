%%writefile /content/ANPR-Project/app.py
import streamlit as st
import cv2
import numpy as np
import easyocr
import re
import json
import http.client
from PIL import Image

st.set_page_config(page_title="ANPR System", page_icon="🚗", layout="wide",
                   initial_sidebar_state="collapsed")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@700;900&family=Syne:wght@400;600&display=swap');
* { font-family: 'Syne', sans-serif; }
.stApp { background: #050810; }
@media (max-width: 768px) {
    .block-container { padding: 1rem !important; }
    .hero { font-size: 1.8rem !important; }
    .plate { font-size: 1.8rem !important; padding: 0.8rem 1rem !important; }
    .owner { font-size: 1.3rem !important; }
    .field { flex-direction: column !important; gap: 0.2rem !important; }
    .val { text-align: left !important; }
    .stButton>button { min-height: 3.5rem !important; font-size: 1.1rem !important; margin-top: 0.5rem !important; }
    .stTextInput input { min-height: 3rem !important; font-size: 1.1rem !important; }
    .stTabs [data-baseweb="tab-list"] { overflow-x: auto !important; flex-wrap: nowrap !important; }
}
.hero { font-family:'Orbitron',monospace; font-size:2.5rem; font-weight:900;
        background:linear-gradient(135deg,#00f5ff,#fff);
        -webkit-background-clip:text; -webkit-text-fill-color:transparent; line-height:1.2; }
.subtitle { color:#4a5a6a; letter-spacing:0.08em; font-size:0.8rem; text-transform:uppercase; margin-top:0.3rem; }
.plate { font-family:'Orbitron',monospace; font-size:2.5rem; font-weight:900;
         color:#00f5ff; text-align:center; background:#0d1a2a;
         border:2px solid #00f5ff44; border-radius:12px;
         padding:1rem 2rem; letter-spacing:0.2em; margin:1rem 0; word-break:break-all; }
.card { background:#0b1120; border:1px solid rgba(0,245,255,0.12); border-radius:16px; padding:1.5rem; margin:0.5rem 0; }
.owner { font-family:'Orbitron',monospace; font-size:1.8rem; font-weight:700; color:#fff; word-break:break-word; }
.field { display:flex; justify-content:space-between; padding:0.5rem 0; border-bottom:1px solid #1a1a28; font-size:0.9rem; gap:1rem; }
.field:last-child { border-bottom:none; }
.lbl { color:#555; white-space:nowrap; }
.val { color:#ccc; font-weight:500; text-align:right; }
.active { color:#00ff88; font-weight:700; }
.inactive { color:#ff4444; font-weight:700; }
.stButton>button { background:linear-gradient(135deg,#00f5ff,#0088aa) !important;
                   color:#000 !important; font-family:'Orbitron',monospace !important;
                   font-size:0.95rem !important; font-weight:900 !important;
                   border:none !important; border-radius:10px !important;
                   width:100% !important; padding:0.8rem !important; }
.stTabs [data-baseweb="tab-list"] { background:#0b1120; border-radius:12px; padding:0.3rem; gap:0.3rem; }
.stTabs [data-baseweb="tab"] { border-radius:8px !important; color:#4a5a6a !important; font-size:0.85rem !important; padding:0.5rem 1rem !important; }
.stTabs [aria-selected="true"] { background:rgba(0,245,255,0.1) !important; color:#00f5ff !important; }
.stTextInput input { background:#0b1120 !important; border:1px solid rgba(0,245,255,0.2) !important;
                     border-radius:10px !important; color:#fff !important; }
[data-testid="stFileUploader"] { background:#0b1120 !important; border:1px dashed rgba(0,245,255,0.2) !important; border-radius:12px !important; }
[data-testid="stSidebar"] { background:#080d18 !important; border-right:1px solid rgba(0,245,255,0.1) !important; }
[data-testid="stMetricValue"] { color:#00f5ff !important; font-family:'Orbitron',monospace !important; }
#MainMenu { visibility:hidden; } footer { visibility:hidden; }
</style>
""", unsafe_allow_html=True)

DEFAULT_API_KEY = "8f4cee7e45msh121ea9043c13d42p190a3ejsn0801860ee586"

@st.cache_resource
def load_reader():
    return easyocr.Reader(['en'], gpu=False, verbose=False)

def fix_plate(text):
    text = ''.join(c for c in text.upper() if c.isalnum())
    text = re.sub(r'^(IND|IN|BH)', '', text)
    L2D = {'O':'0','I':'1','S':'5','B':'8','Z':'2','G':'6','Q':'0','D':'0'}
    D2L = {'0':'O','1':'I','5':'S','8':'B','2':'Z','6':'G'}
    r = list(text); n = len(r)
    try:
        for i in [0,1]:
            if i<n and r[i].isdigit(): r[i]=D2L.get(r[i],r[i])
        for i in [2,3]:
            if i<n and r[i].isalpha(): r[i]=L2D.get(r[i],r[i])
        for i in range(max(4,n-4),n):
            if r[i].isalpha(): r[i]=L2D.get(r[i],r[i])
        for i in range(4,max(4,n-4)):
            if r[i].isdigit(): r[i]=D2L.get(r[i],r[i])
    except: pass
    fixed = ''.join(r)
    m = re.search(r'([A-Z]{2})(\d{1,2})([A-Z]{1,3})(\d{1,4})', fixed)
    return ''.join(m.groups()) if m else fixed

def detect_plate(img, reader):
    H,W = img.shape[:2]; found=[]
    PATTERN = re.compile(r'^[A-Z]{2}\d{1,2}[A-Z]{1,3}\d{1,4}$')
    for (bbox,text,conf) in reader.readtext(img,detail=1,paragraph=False,
                                            contrast_ths=0.1,adjust_contrast=0.8):
        cleaned = fix_plate(text)
        if PATTERN.match(cleaned) and conf>0.2:
            pts = np.array(bbox,dtype=np.int32)
            x1,y1=max(0,pts[:,0].min()-10),max(0,pts[:,1].min()-10)
            x2,y2=min(W,pts[:,0].max()+10),min(H,pts[:,1].max()+10)
            found.append({'bbox':(x1,y1,x2,y2),'plate':cleaned,
                          'conf':conf,'crop':img[y1:y2,x1:x2],'method':'Direct'})
    if not found:
        bot=img[int(H*0.4):H,:]
        for (bbox,text,conf) in reader.readtext(bot,detail=1,paragraph=False,
                                                contrast_ths=0.1,adjust_contrast=0.8):
            cleaned=fix_plate(text)
            if len(cleaned)>=6 and conf>0.15:
                pts=np.array(bbox,dtype=np.int32)
                x1,y1=pts[:,0].min(),pts[:,1].min()+int(H*0.4)
                x2,y2=pts[:,0].max(),pts[:,1].max()+int(H*0.4)
                found.append({'bbox':(x1,y1,x2,y2),'plate':cleaned,
                              'conf':conf,'crop':img[y1:y2,x1:x2],'method':'Bottom'})
    seen,unique={},[]
    for d in sorted(found,key=lambda x:x['conf'],reverse=True):
        k=d['plate'][:6]
        if k not in seen: seen[k]=True; unique.append(d)
    return unique

def get_owner(plate, api_key):
    try:
        conn=http.client.HTTPSConnection("vehicle-rc-information-v2.p.rapidapi.com")
        headers={'x-rapidapi-key':api_key,
                 'x-rapidapi-host':"vehicle-rc-information-v2.p.rapidapi.com",
                 'Content-Type':"application/json"}
        conn.request("POST","/",json.dumps({"vehicle_number":plate}),headers)
        res=conn.getresponse()
        data=json.loads(res.read().decode("utf-8"))
        for k in ["data","result","response","vehicle","rc_details"]:
            if k in data and isinstance(data[k],dict): return data[k]
        return data
    except Exception as e:
        return {"error":str(e)}

def annotate(img, found):
    out=img.copy()
    for d in found:
        x1,y1,x2,y2=d['bbox']
        lbl=f"{d['plate']} {d['conf']:.0%}"
        cv2.rectangle(out,(x1,y1),(x2,y2),(0,212,255),3)
        (tw,th),_=cv2.getTextSize(lbl,cv2.FONT_HERSHEY_DUPLEX,0.9,2)
        cv2.rectangle(out,(x1,y1-th-14),(x1+tw+10,y1),(0,212,255),-1)
        cv2.putText(out,lbl,(x1+5,y1-7),cv2.FONT_HERSHEY_DUPLEX,0.9,(0,0,0),2)
    return out

def show_owner_details(data, plate):
    if "error" in data or "message" in data:
        msg=data.get('error') or data.get('message')
        if "quota" in msg.lower() or "limit" in msg.lower():
            st.error("⚠️ Daily API limit reached. Try tomorrow or update API key.")
        elif "subscri" in msg.lower():
            st.error("⚠️ API not subscribed. Go to RapidAPI and subscribe free.")
        else:
            st.error(f"❌ {msg}")
        return
    owner=data.get("owner_name","Unknown")
    status=data.get("rc_status","—")
    father=data.get("father_name","")
    addr=data.get("permanent_address") or data.get("present_address","—")
    scolor="active" if status=="ACTIVE" else "inactive"
    st.markdown(f'<div class="plate">{plate}</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="card">
        <div class="owner">{owner}</div>
        {f'<div style="color:#555;font-size:0.85rem;margin-top:0.3rem">S/O {father}</div>' if father else ''}
        <div class="{scolor}" style="margin-top:0.5rem;font-size:1rem">● RC: {status}</div>
        <div style="color:#666;font-size:0.82rem;margin-top:0.4rem">📍 {addr}</div>
    </div>""", unsafe_allow_html=True)
    m1,m2,m3=st.columns(3)
    m1.metric("👥 Owners", data.get("owner_count","—"))
    m2.metric("💨 CC",     data.get("cubic_capacity","—"))
    m3.metric("⛽ Fuel",   data.get("fuel_type","—"))
    fields=[
        ("brand_name","🏭 Brand"),("brand_model","📋 Model"),
        ("color","🎨 Color"),("registration_date","📅 Reg. Date"),
        ("insurance_company","🛡️ Insurance"),("insurance_expiry","🛡️ Ins. Expiry"),
        ("pucc_upto","🌿 PUC Until"),("tax_paid_upto","💰 Tax Until"),
        ("chassis_number","🔩 Chassis"),("engine_number","⚙️ Engine"),
        ("office_name","🏢 RTO"),("state","📍 State"),
        ("blacklist_status","🚫 Blacklist"),("norms","🌿 Norms"),
    ]
    rows=""
    for key,label in fields:
        val=str(data.get(key,"")).strip()
        if val and val.lower() not in ("","null","none","n/a","na","0"):
            rows+=f'<div class="field"><span class="lbl">{label}</span><span class="val">{val}</span></div>'
    if rows:
        st.markdown(f'<div class="card">{rows}</div>', unsafe_allow_html=True)

def save_history(plate, data):
    if "anpr_history" not in st.session_state:
        st.session_state.anpr_history=[]
    if "error" not in data and "message" not in data:
        existing=[h['plate'] for h in st.session_state.anpr_history]
        if plate not in existing:
            st.session_state.anpr_history.insert(0,{
                "plate":plate,
                "owner":data.get("owner_name","Unknown"),
                "status":data.get("rc_status","—"),
                "data":data
            })

with st.sidebar:
    st.markdown("### ⚙️ Settings")
    api_key=st.text_input("RapidAPI Key", value=DEFAULT_API_KEY, type="password",
                           help="Default key provided. Replace if quota exceeded.")
    st.caption("🔑 Default key provided. Replace if quota exceeded.")
    st.markdown("---")
    st.markdown("""
    **How to use:**
    - 📸 **Tab 1:** Upload photo → auto detect
    - ⌨️ **Tab 2:** Type plate manually
    - 🔍 **Tab 3:** View search history

    **Get free API key:**
    [RapidAPI →](https://rapidapi.com/fatehbrar92/api/vehicle-rc-information-v2)
    """)
    st.markdown("---")
    st.markdown("**Built by:** Somya Bhalani  \n**College:** Parul University  \n**Sem:** 2nd CSE-AI")

st.markdown('<div class="hero">🚗 ANPR System</div>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Automatic Number Plate Recognition · Owner Lookup</p>',
            unsafe_allow_html=True)
st.markdown("---")

tab1,tab2,tab3=st.tabs(["📸  Upload Image","⌨️  Enter Plate Manually","🔍  Search History"])

with tab1:
    st.markdown("#### Upload a vehicle photo")
    uploaded=st.file_uploader("Upload Image", type=["jpg","jpeg","png"],
                               label_visibility="collapsed")
    st.markdown("##### ✏️ Plate not detected correctly? Override:")
    override=st.text_input("Override Plate",
                            placeholder="Type correct plate e.g. AB12CD3456",
                            key="override1",
                            label_visibility="collapsed")
    override=override.upper().replace(" ","")
    if uploaded:
        img_pil=Image.open(uploaded).convert("RGB")
        img_np=np.array(img_pil)
        img_bgr=cv2.cvtColor(img_np,cv2.COLOR_RGB2BGR)
        st.image(img_pil, use_container_width=True)
    btn1=st.button("🔍  DETECT PLATE & GET OWNER", key="btn1",
                   disabled=not uploaded)
    if uploaded and btn1:
        reader=load_reader()
        with st.spinner("🔍 Scanning plate..."):
            found=detect_plate(img_bgr, reader)
        if found:
            ann=annotate(img_bgr, found)
            st.image(cv2.cvtColor(ann,cv2.COLOR_BGR2RGB),
                     use_container_width=True,
                     caption=f"Detected: {found[0]['plate']} ({found[0]['conf']:.0%})")
        PATTERN=re.compile(r'^[A-Z]{2}\d{1,2}[A-Z]{1,3}\d{1,4}$')
        if override:
            plate=override
            st.success(f"✅ Using manual plate: **{plate}**")
        elif found:
            plate=next((d['plate'] for d in found if PATTERN.match(d['plate'])),
                       found[0]['plate'])
            st.success(f"✅ Auto detected: **{plate}** ({found[0]['conf']:.0%})")
        else:
            st.warning("⚠️ No plate detected. Type it in the override box above.")
            st.stop()
        with st.spinner(f"📡 Fetching owner for {plate}..."):
            data=get_owner(plate, api_key)
        show_owner_details(data, plate)
        save_history(plate, data)

with tab2:
    st.markdown("#### Enter any vehicle number")
    st.markdown("")
    manual_plate=st.text_input("Vehicle Number Plate",
                                placeholder="Enter plate  e.g.  AB12CD3456",
                                key="manual_plate")
    manual_plate=manual_plate.upper().replace(" ","")
    if manual_plate and len(manual_plate)>=4:
        st.markdown(f"""
        <div style="background:#0d1a2a;border:1px solid #00f5ff33;
                    border-radius:10px;padding:0.8rem;text-align:center;
                    font-family:'Orbitron',monospace;font-size:1.5rem;
                    color:#00f5ff;letter-spacing:0.2em;margin:0.5rem 0">
            {manual_plate}
        </div>""", unsafe_allow_html=True)
    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
    btn2=st.button("🔍  GET OWNER DETAILS", key="btn2", use_container_width=True)
    st.markdown("**Quick examples:**")
    c1,c2,c3,c4=st.columns(4)
    ex_plate=None
    with c1:
        if st.button("GJ11CQ3300"): ex_plate="GJ11CQ3300"
    with c2:
        if st.button("MH12AB1234"): ex_plate="MH12AB1234"
    with c3:
        if st.button("DL3CAB1234"): ex_plate="DL3CAB1234"
    with c4:
        if st.button("KA01MF0001"): ex_plate="KA01MF0001"
    if ex_plate: manual_plate=ex_plate
    st.markdown("---")
    if btn2 or ex_plate:
        search_plate=ex_plate if ex_plate else manual_plate
        if not search_plate or len(search_plate)<6:
            st.warning("⚠️ Please enter a valid plate number (min 6 characters)")
        else:
            with st.spinner(f"📡 Looking up {search_plate}..."):
                data=get_owner(search_plate, api_key)
            show_owner_details(data, search_plate)
            save_history(search_plate, data)

with tab3:
    st.markdown("#### Your recent searches")
    if "anpr_history" not in st.session_state or not st.session_state.anpr_history:
        st.info("🔍 No searches yet. Go to Tab 1 or Tab 2 to search!")
    else:
        for i,h in enumerate(st.session_state.anpr_history[:10]):
            with st.container():
                c1,c2,c3,c4=st.columns([2,3,2,1])
                with c1: st.markdown(f"**{h['plate']}**")
                with c2: st.markdown(f"{h['owner']}")
                with c3:
                    color="🟢" if h['status']=="ACTIVE" else "🔴"
                    st.markdown(f"{color} {h['status']}")
                with c4:
                    if st.button("View", key=f"view_{i}"):
                        show_owner_details(h['data'], h['plate'])
                st.divider()
        if st.button("🗑️ Clear History"):
            st.session_state.anpr_history=[]
            st.rerun()
