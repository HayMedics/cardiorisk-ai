"""
CardioRisk AI — HayMedics Academy
Coronary Artery Disease Risk Assessment
Research Use Only
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib, json, os, base64, warnings
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
warnings.filterwarnings("ignore")

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="CardioRisk AI | HayMedics Academy",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Logo loader ───────────────────────────────────────────────
def img_b64(folder, names):
    for n in names:
        p = os.path.join(folder, n)
        if os.path.exists(p):
            ext = p.rsplit(".", 1)[-1].lower()
            mime = "jpeg" if ext in ("jpg","jpeg") else "png"
            with open(p, "rb") as f:
                return mime, base64.b64encode(f.read()).decode()
    return None, ""

BASE         = os.path.dirname(os.path.abspath(__file__))
logo_mime, logo_b64 = img_b64(BASE, [
    "HayMedics_Academy06.jpg","HayMedics_Academy05.jpg",
    "HMA.jpg","HMA_PNG.png","HMA_ICON.jpg","HMA_ICON.jpeg"])
icon_mime, icon_b64 = img_b64(BASE, [
    "HMA_ICON.jpg","HMA_ICON.jpeg","HMA_ICON_PNG.png"])
wm_mime,   wm_b64   = img_b64(BASE, [
    "HMA_ICON_PNG.png","HMA_ICON.jpg","HMA_ICON.jpeg"])

logo_tag = f'<img src="data:image/{logo_mime};base64,{logo_b64}" style="height:52px;"/>' if logo_b64 else '<span style="font-size:1.4rem;font-weight:900;color:#fff;">Hay<span style="color:#F5A623;">Medics</span> Academy</span>'
icon_tag = f'<img src="data:image/{icon_mime};base64,{icon_b64}" style="height:44px;border-radius:8px;"/>' if icon_b64 else "🫀"
wm_tag   = f'<img src="data:image/{wm_mime};base64,{wm_b64}" style="width:380px;"/>' if wm_b64 else ""

# ── CSS ───────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

*, html, body, [class*="css"] {
  font-family: 'Inter', sans-serif;
  box-sizing: border-box;
}

/* ── App background ── */
.stApp {
  background: #F4F6FB;
  color: #0D1B4B;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
  background: #0D1B4B !important;
  border-right: 4px solid #F5A623;
  padding-top: 0 !important;
}
section[data-testid="stSidebar"] > div {
  padding: 0 !important;
}
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] div {
  color: #CBD5F0 !important;
}
section[data-testid="stSidebar"] .stSlider label {
  font-size: 0.78rem !important;
  color: #94A3C8 !important;
  font-weight: 500 !important;
}
section[data-testid="stSidebar"] .stSelectbox label {
  font-size: 0.78rem !important;
  color: #94A3C8 !important;
  font-weight: 500 !important;
}

/* ── Sidebar logo area ── */
.sb-logo {
  background: linear-gradient(135deg, #0D1B4B 0%, #162256 100%);
  padding: 20px 20px 16px;
  border-bottom: 2px solid rgba(245,166,35,0.4);
  margin-bottom: 4px;
}
.sb-tagline {
  font-size: 0.65rem;
  color: #F5A623 !important;
  letter-spacing: 2px;
  text-transform: uppercase;
  font-weight: 600;
  margin-top: 6px;
}

/* ── Nav section headings ── */
.sb-section {
  font-size: 0.58rem;
  font-weight: 700;
  letter-spacing: 2.5px;
  text-transform: uppercase;
  color: #F5A623 !important;
  padding: 14px 20px 6px;
  border-bottom: 1px solid rgba(245,166,35,0.2);
  margin-bottom: 4px;
}

/* ── Watermark ── */
.watermark {
  position: fixed;
  bottom: 0px;
  right: 0px;
  opacity: 0.04;
  z-index: 0;
  pointer-events: none;
}

/* ── Top header bar ── */
.header-bar {
  background: linear-gradient(90deg, #0D1B4B 0%, #1B3080 60%, #0D1B4B 100%);
  border-radius: 14px;
  padding: 18px 28px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 6px;
  border-bottom: 3px solid #F5A623;
  box-shadow: 0 4px 20px rgba(13,27,75,0.2);
}
.header-left { display:flex; align-items:center; gap:16px; }
.header-app-name {
  font-size: 1.5rem;
  font-weight: 800;
  color: white;
  letter-spacing: -0.5px;
  line-height: 1.1;
}
.header-app-name em { color:#F5A623; font-style:normal; }
.header-sub {
  font-size: 0.72rem;
  color: #94A3C8;
  margin-top: 3px;
  letter-spacing: 0.2px;
}
.header-right { display:flex; gap:10px; align-items:center; }
.badge {
  padding: 5px 14px;
  border-radius: 20px;
  font-size: 0.68rem;
  font-weight: 600;
  letter-spacing: 0.8px;
  text-transform: uppercase;
  white-space: nowrap;
}
.badge-warn {
  background: rgba(245,166,35,0.15);
  border: 1.5px solid #F5A623;
  color: #F5A623;
}
.badge-blue {
  background: rgba(99,152,255,0.15);
  border: 1.5px solid #6398FF;
  color: #6398FF;
}

/* ── Metric strip ── */
.metric-strip {
  display: flex;
  gap: 8px;
  flex-wrap: wrap;
  margin: 10px 0 18px;
}
.metric-chip {
  background: white;
  border: 1.5px solid #DDE3F5;
  border-radius: 10px;
  padding: 8px 14px;
  display: flex;
  flex-direction: column;
  align-items: center;
  min-width: 80px;
  box-shadow: 0 1px 6px rgba(13,27,75,0.06);
}
.metric-chip-label {
  font-size: 0.6rem;
  font-weight: 600;
  letter-spacing: 1px;
  text-transform: uppercase;
  color: #8894B8;
}
.metric-chip-val {
  font-size: 1.1rem;
  font-weight: 800;
  color: #0D1B4B;
  font-family: 'JetBrains Mono', monospace;
  margin-top: 2px;
}

/* ── Section heading ── */
.sec-head {
  font-size: 0.65rem;
  font-weight: 700;
  letter-spacing: 2.5px;
  text-transform: uppercase;
  color: #1B3080;
  padding: 14px 0 7px;
  border-bottom: 2px solid #DDE3F5;
  margin-bottom: 10px;
  display: flex;
  align-items: center;
  gap: 7px;
}

/* ── Cards ── */
.card {
  background: white;
  border-radius: 14px;
  padding: 22px 24px;
  border: 1px solid #DDE3F5;
  box-shadow: 0 2px 10px rgba(13,27,75,0.05);
  margin-bottom: 14px;
}
.card-title {
  font-size: 0.9rem;
  font-weight: 700;
  color: #0D1B4B;
  margin-bottom: 10px;
  display: flex;
  align-items: center;
  gap: 7px;
}

/* ── Risk result cards ── */
.result-high {
  background: linear-gradient(135deg,#FFF2F2,#FFE8E8);
  border: 2px solid #DC2626;
  border-radius: 16px;
  padding: 28px 32px;
  text-align: center;
  box-shadow: 0 6px 28px rgba(220,38,38,0.12);
}
.result-moderate {
  background: linear-gradient(135deg,#FFFCF0,#FFF5D6);
  border: 2px solid #D97706;
  border-radius: 16px;
  padding: 28px 32px;
  text-align: center;
  box-shadow: 0 6px 28px rgba(217,119,6,0.12);
}
.result-low {
  background: linear-gradient(135deg,#F0FFF8,#E6FFF2);
  border: 2px solid #059669;
  border-radius: 16px;
  padding: 28px 32px;
  text-align: center;
  box-shadow: 0 6px 28px rgba(5,150,105,0.1);
}
.big-num {
  font-family: 'JetBrains Mono', monospace;
  font-size: 3.8rem;
  font-weight: 700;
  letter-spacing: -3px;
  line-height: 1;
}
.num-high     { color: #DC2626; }
.num-moderate { color: #D97706; }
.num-low      { color: #059669; }
.risk-tag {
  display: inline-block;
  margin-top: 10px;
  padding: 5px 18px;
  border-radius: 20px;
  font-size: 0.75rem;
  font-weight: 700;
  letter-spacing: 2px;
  text-transform: uppercase;
}
.tag-high     { background:#DC2626; color:white; }
.tag-moderate { background:#D97706; color:white; }
.tag-low      { background:#059669; color:white; }
.result-caption {
  font-size: 0.78rem;
  color: #6B7BB5;
  margin-top: 10px;
}

/* ── Info / warn / danger alerts ── */
.alert {
  display: flex;
  gap: 10px;
  align-items: flex-start;
  padding: 10px 14px;
  border-radius: 10px;
  margin: 6px 0;
  font-size: 0.82rem;
  line-height: 1.55;
}
.alert-info    { background:#EEF4FF; border-left:3px solid #1B3080; color:#1B3080; }
.alert-warn    { background:#FFFBEC; border-left:3px solid #D97706; color:#92600A; }
.alert-danger  { background:#FFF5F5; border-left:3px solid #DC2626; color:#B91C1C; }
.alert-success { background:#F0FFF8; border-left:3px solid #059669; color:#065F46; }

/* ── Data rows ── */
.data-row {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 7px 0;
  border-bottom: 1px solid #F1F3FB;
  font-size: 0.82rem;
}
.data-row:last-child { border-bottom: none; }
.data-label { color: #8894B8; font-weight: 500; }
.data-val   { color: #0D1B4B; font-weight: 600; font-family:'JetBrains Mono',monospace; font-size:0.79rem; }

/* ── Predict button ── */
div.stButton > button {
  background: linear-gradient(135deg, #0D1B4B 0%, #1B3080 100%);
  color: white;
  border: none;
  border-radius: 12px;
  padding: 15px 0;
  width: 100%;
  font-family: 'Inter', sans-serif;
  font-size: 0.95rem;
  font-weight: 700;
  letter-spacing: 0.3px;
  box-shadow: 0 4px 18px rgba(13,27,75,0.28);
  transition: all 0.2s;
  cursor: pointer;
}
div.stButton > button:hover {
  background: linear-gradient(135deg, #F5A623 0%, #E09010 100%);
  box-shadow: 0 6px 22px rgba(245,166,35,0.32);
  transform: translateY(-1px);
}

/* ── About page ── */
.about-section {
  background: white;
  border-radius: 14px;
  padding: 26px 28px;
  border: 1px solid #DDE3F5;
  box-shadow: 0 2px 10px rgba(13,27,75,0.05);
  margin-bottom: 16px;
}
.about-section h3 {
  font-size: 1rem;
  font-weight: 700;
  color: #0D1B4B;
  margin: 0 0 14px;
  padding-bottom: 8px;
  border-bottom: 2px solid #F4F6FB;
}
.about-section p, .about-section li {
  font-size: 0.86rem;
  color: #4A5580;
  line-height: 1.85;
}
.about-section ul { padding-left: 18px; margin: 6px 0; }

/* ── Footer ── */
.footer {
  background: linear-gradient(90deg, #0D1B4B, #1B3080);
  border-radius: 12px;
  padding: 14px 24px;
  text-align: center;
  font-size: 0.7rem;
  color: #94A3C8;
  margin-top: 28px;
  border-top: 2px solid #F5A623;
}
.footer strong { color: white; }
.footer em { color: #F5A623; font-style: normal; }

/* Hide Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
div[data-testid="stDecoration"] { display:none; }
</style>
""", unsafe_allow_html=True)

# ── Watermark ─────────────────────────────────────────────────
if wm_b64:
    st.markdown(
        f'<div class="watermark">{wm_tag}</div>',
        unsafe_allow_html=True)

# ── Load model ────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    d = os.path.dirname(os.path.abspath(__file__))
    m = joblib.load(os.path.join(d, "streamlit_app", "model.pkl"))
    p = joblib.load(os.path.join(d, "streamlit_app", "preprocessor.pkl"))
    with open(os.path.join(d, "streamlit_app", "feature_config.json")) as f:
        c = json.load(f)
    return m, p, c

try:
    model, preprocessor, cfg = load_model()
    THRESHOLD = cfg["OPTIMAL_THRESHOLD"]
    ok = True
except Exception as e:
    ok = False
    err = str(e)

# ── Feature engineering ───────────────────────────────────────
def engineer(df):
    d = df.copy()
    d['age_sex_interaction'] = d['age'] * d['sex']
    d['age_risk']  = (((d['sex']==1)&(d['age']>=45))|((d['sex']==0)&(d['age']>=55))).astype(int)
    d['hr_reserve']         = (220-d['age'])-d['thalach']
    d['pct_max_hr']         = d['thalach']/(220-d['age'])
    d['chrono_incompetence']= (d['pct_max_hr']<0.85).astype(int)
    d['high_risk_ett']      = ((d['oldpeak']>=2.0)&(d['slope']==0)&(d['exang']==1)).astype(int)
    d['bp_elevated']        = (d['trestbps']>=130).astype(int)
    d['chol_elevated']      = (d['chol']>=200).astype(int)
    d['metabolic_score']    = d['fbs']+d['bp_elevated']+d['chol_elevated']
    d['framingham_proxy']   = 0.04*d['age']+0.32*d['sex']+0.18*(d['chol']/50)+0.13*(d['trestbps']/20)+0.11*d['fbs']
    return d

# ── Gauge ─────────────────────────────────────────────────────
def gauge(prob):
    fig, ax = plt.subplots(figsize=(5, 2.6))
    fig.patch.set_facecolor('none'); ax.set_facecolor('none')
    segs = [('#059669','#68D391',(0,.25)),('#68D391','#FBBF24',(.25,.45)),
            ('#FBBF24','#F97316',(.45,.65)),('#F97316','#DC2626',(.65,1.0))]
    for c1,c2,(a,b) in segs:
        th = np.linspace(np.pi*(1-b), np.pi*(1-a), 80)
        ax.fill_between(np.cos(th),np.sin(th),0.6*np.cos(th),0.6*np.sin(th),color=c1,alpha=0.9,zorder=2)
    ang = np.pi*(1-prob)
    ax.annotate("", xy=(0.5*np.cos(ang),0.5*np.sin(ang)), xytext=(0,0),
                arrowprops=dict(arrowstyle="-|>",color="#0D1B4B",lw=2.6,mutation_scale=15))
    ax.plot(0,0,'o',color='#0D1B4B',ms=9,zorder=5)
    for v,l in [(0,'0%'),(0.25,'25%'),(0.5,'50%'),(0.75,'75%'),(1,'100%')]:
        a=np.pi*(1-v)
        ax.text(1.13*np.cos(a),1.13*np.sin(a),l,ha='center',va='center',
                fontsize=7,color='#8894B8',fontfamily='monospace')
    ax.set_xlim(-1.2,1.2); ax.set_ylim(-0.2,1.2); ax.set_aspect('equal'); ax.axis('off')
    plt.tight_layout(pad=0)
    return fig

# ── Risk bar ─────────────────────────────────────────────────
def riskbar(prob):
    fig, ax = plt.subplots(figsize=(6, 0.45))
    fig.patch.set_alpha(0); ax.set_facecolor('none')
    ax.imshow(np.linspace(0,1,300).reshape(1,-1), aspect='auto',
              extent=[0,1,0,1], cmap=plt.cm.RdYlGn_r, vmin=0, vmax=1, alpha=0.8)
    ax.axvline(prob, color='#0D1B4B', lw=2.5, zorder=5)
    ax.plot(prob, 0.5, 'o', color='#0D1B4B', ms=9, zorder=6)
    ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis('off')
    plt.tight_layout(pad=0)
    return fig

# ══════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    # Logo block
    st.markdown(f"""
    <div class="sb-logo">
      {logo_tag}
      <div class="sb-tagline">Data · Research · Innovation</div>
    </div>""", unsafe_allow_html=True)

    # Navigation
    st.markdown('<div class="sb-section">Navigation</div>', unsafe_allow_html=True)
    page = st.radio("", ["🫀  Risk Assessment", "📖  About the Project"],
                    label_visibility="collapsed")

    if "Risk" in page:
        # ── Demographics ──
        st.markdown('<div class="sb-section">Demographics</div>', unsafe_allow_html=True)
        age = st.slider("Age (years)", 20, 90, 55)
        sex = st.selectbox("Sex", [0,1],
                           format_func=lambda x:"Female" if x==0 else "Male", index=1)

        # ── Symptoms ──
        st.markdown('<div class="sb-section">Symptoms & History</div>', unsafe_allow_html=True)
        cp = st.selectbox("Chest Pain Type", [0,1,2,3],
                          format_func=lambda x:{0:"Typical Angina",1:"Atypical Angina",
                                                2:"Non-anginal Pain",3:"Asymptomatic"}[x])
        exang = st.selectbox("Exercise-Induced Angina",[0,1],
                             format_func=lambda x:"No" if x==0 else "Yes")
        fbs   = st.selectbox("Fasting Blood Sugar >120 mg/dL",[0,1],
                             format_func=lambda x:"No" if x==0 else "Yes")

        # ── Vitals ──
        st.markdown('<div class="sb-section">Vitals & Labs</div>', unsafe_allow_html=True)
        trestbps = st.slider("Resting BP (mm Hg)", 80, 200, 130)
        chol     = st.slider("Cholesterol (mg/dL)", 100, 600, 240)

        # ── ECG ──
        st.markdown('<div class="sb-section">ECG & Stress Test</div>', unsafe_allow_html=True)
        restecg = st.selectbox("Resting ECG",[0,1,2],
                               format_func=lambda x:{0:"Normal",
                                                     1:"ST-T Abnormality",
                                                     2:"LVH (Estes)"}[x])
        thalach = st.slider(f"Max Heart Rate (bpm) · Pred: {220-age}", 60, 220, 150)
        oldpeak = st.slider("ST Depression (mm)", 0.0, 6.5, 1.5, 0.1)
        slope   = st.selectbox("ST Slope",[0,1,2],
                               format_func=lambda x:{0:"Downsloping ↓",
                                                     1:"Flat →",
                                                     2:"Upsloping ↑"}[x])

        # ── Invasive ──
        st.markdown('<div class="sb-section">Invasive Results (optional)</div>',
                    unsafe_allow_html=True)
        ca_sel = st.selectbox("Stenosed Vessels (ca)",
                              [0.0,1.0,2.0,3.0,float('nan')],
                              format_func=lambda x:"Unknown" if (isinstance(x,float) and np.isnan(x))
                                          else f"{int(x)} vessel{'s' if x!=1 else ''}")
        thal_sel = st.selectbox("Thalassemia (thal)",
                                [1.0,2.0,3.0,float('nan')],
                                format_func=lambda x:"Unknown" if (isinstance(x,float) and np.isnan(x))
                                            else {1.0:"Fixed Defect",2.0:"Normal",
                                                  3.0:"Reversible Defect"}.get(x,""),
                                index=1)
        st.markdown("<br>", unsafe_allow_html=True)
        predict = st.button("🫀  Analyse CAD Risk", use_container_width=True)
    else:
        predict = False

    # Sidebar footer
    st.markdown("""
    <div style='padding:16px 20px;margin-top:20px;border-top:1px solid rgba(245,166,35,0.25);'>
      <p style='font-size:0.65rem;color:rgba(148,163,200,0.6);margin:0;text-align:center;'>
        HayMedics Academy<br>Learn with Ease · Achieve with Confidence
      </p>
    </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════
st.markdown(f"""
<div class="header-bar">
  <div class="header-left">
    {icon_tag}
    <div>
      <div class="header-app-name">CardioRisk<em>AI</em></div>
      <div class="header-sub">
        HayMedics Academy &nbsp;·&nbsp; Coronary Artery Disease Risk Assessment
        &nbsp;·&nbsp; TRIPOD-AI Standards &nbsp;·&nbsp; Cleveland Dataset
      </div>
    </div>
  </div>
  <div class="header-right">
    <span class="badge badge-blue">TRIPOD-AI</span>
    <span class="badge badge-warn">⚠ Research Only</span>
  </div>
</div>
""", unsafe_allow_html=True)

if not ok:
    st.error(f"**Model files not found.** Run the notebook first to create `streamlit_app/` folder.\n\n_{err}_")
    st.stop()

# ── Metric strip ──────────────────────────────────────────────
metrics = [
    ("AUC-ROC",      cfg.get('model_auc','—')),
    ("Sensitivity",  cfg.get('model_sensitivity','—')),
    ("Specificity",  cfg.get('model_specificity','—')),
    ("PPV",          cfg.get('model_ppv','—')),
    ("NPV",          cfg.get('model_npv','—')),
    ("Threshold",    f"{THRESHOLD:.2f}"),
    ("Dataset",      "n=302"),
    ("Model",        "Stacking"),
]
chips = "".join([f"""
<div class="metric-chip">
  <span class="metric-chip-label">{l}</span>
  <span class="metric-chip-val">{v}</span>
</div>""" for l,v in metrics])
st.markdown(f'<div class="metric-strip">{chips}</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# PAGE — ABOUT
# ══════════════════════════════════════════════════════════════
if "About" in page:
    st.markdown('<div class="sec-head">📖 About the Project</div>',
                unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3, gap="medium")

    with c1:
        st.markdown("""
        <div class="about-section">
          <h3>🎯 Project Overview</h3>
          <p>CardioRisk AI classifies the risk of Coronary Artery Disease (CAD)
          using the Cleveland Heart Disease dataset — one of the most studied
          cardiac ML benchmarks worldwide.</p>
          <p>Built following <strong>TRIPOD-AI</strong> reporting standards with
          clinical cost optimisation (FN penalised 10× FP).</p>
        </div>
        <div class="about-section">
          <h3>🗃️ Dataset Details</h3>
          <ul>
            <li><strong>Source:</strong> UCI ML Repository</li>
            <li><strong>Origin:</strong> Cleveland Clinic Foundation</li>
            <li><strong>Period:</strong> 1981–1984</li>
            <li><strong>Size:</strong> 302 patients, 13 features</li>
            <li><strong>Outcome:</strong> CAD ≥50% stenosis</li>
            <li><strong>Prevalence:</strong> ~54% positive</li>
            <li><strong>Citation:</strong> Detrano R et al. Am J Cardiol. 1989</li>
          </ul>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="about-section">
          <h3>⚙️ Model Architecture</h3>
          <ul>
            <li><strong>Type:</strong> Stacking Ensemble</li>
            <li><strong>Base 1:</strong> Gradient Boosting</li>
            <li><strong>Base 2:</strong> Logistic Regression</li>
            <li><strong>Base 3:</strong> Random Forest</li>
            <li><strong>Base 4:</strong> Extra Trees</li>
            <li><strong>Meta-learner:</strong> Logistic Regression</li>
            <li><strong>Calibration:</strong> Isotonic (cv=3)</li>
            <li><strong>Validation:</strong> 5-fold nested CV</li>
            <li><strong>Threshold:</strong> Cost-based FN=10×FP</li>
          </ul>
        </div>
        <div class="about-section">
          <h3>🧠 Engineered Features</h3>
          <ul>
            <li>HR Reserve (chronotropic capacity)</li>
            <li>% Age-predicted Max HR</li>
            <li>Chronotropic Incompetence flag</li>
            <li>Framingham Risk Proxy score</li>
            <li>Metabolic Syndrome score</li>
            <li>High-risk ETT composite</li>
            <li>Age × Sex interaction</li>
          </ul>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown("""
        <div class="about-section">
          <h3>📋 Clinical Features</h3>
          <ul>
            <li><strong>age</strong> — Age in years</li>
            <li><strong>sex</strong> — Male / Female</li>
            <li><strong>cp</strong> — Chest pain type (0–3)</li>
            <li><strong>trestbps</strong> — Resting BP (mm Hg)</li>
            <li><strong>chol</strong> — Cholesterol (mg/dL)</li>
            <li><strong>fbs</strong> — Fasting blood sugar</li>
            <li><strong>restecg</strong> — Resting ECG</li>
            <li><strong>thalach</strong> — Max heart rate</li>
            <li><strong>exang</strong> — Exercise angina</li>
            <li><strong>oldpeak</strong> — ST depression</li>
            <li><strong>slope</strong> — ST slope</li>
            <li><strong>ca</strong> — Vessels stenosed</li>
            <li><strong>thal</strong> — Thalassemia scan</li>
          </ul>
        </div>
        <div class="about-section">
          <h3>⚠️ Key Limitations</h3>
          <ul>
            <li>Single centre, 1981–1984 only</li>
            <li>n=302 — small for subgroups</li>
            <li>No medication data</li>
            <li>No ethnicity information</li>
            <li>FBS threshold discrepancy (>120 vs ADA ≥126)</li>
            <li>No external validation cohort</li>
            <li>AUC ceiling ~0.90 without biomarkers</li>
          </ul>
        </div>
        """, unsafe_allow_html=True)

    # References row
    st.markdown("""
    <div class="about-section">
      <h3>📚 Standards & References</h3>
      <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:12px;">
        <div>
          <p><strong>Reporting:</strong> TRIPOD-AI (Collins et al. 2024)</p>
          <p><strong>BP:</strong> ACC/AHA 2019 Hypertension Guidelines</p>
        </div>
        <div>
          <p><strong>Lipids:</strong> ATPIII Cholesterol Classification</p>
          <p><strong>Diabetes:</strong> ADA Diagnostic Criteria 2024</p>
        </div>
        <div>
          <p><strong>Dataset:</strong> Detrano R et al. Am J Cardiol. 1989;64(5):304-310</p>
          <p><strong>Fairness:</strong> Equal opportunity difference threshold &lt;0.10</p>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# PAGE — RISK ASSESSMENT
# ══════════════════════════════════════════════════════════════
elif "Risk" in page:

    if not predict:
        # ── Welcome state ─────────────────────────────────────
        col_w, col_ref = st.columns([3, 2], gap="large")

        with col_w:
            st.markdown('<div class="sec-head">👋 How to Use This Tool</div>',
                        unsafe_allow_html=True)
            st.markdown("""
            <div class="card">
              <p style="color:#4A5580;font-size:0.9rem;line-height:2;">
                CardioRisk AI uses a research-grade stacking ensemble model to estimate
                the probability of Coronary Artery Disease from clinical measurements.
              </p>
              <ol style="color:#4A5580;font-size:0.87rem;line-height:2.3;padding-left:20px;">
                <li>Complete the patient data fields in the <strong style="color:#0D1B4B;">left sidebar</strong></li>
                <li>Invasive results (ca, thal) are optional — leave as Unknown if unavailable</li>
                <li>Click <strong style="color:#F5A623;">Analyse CAD Risk</strong> to generate the prediction</li>
                <li>Review the probability score, risk category, and clinical flags</li>
              </ol>
            </div>
            <div class="alert alert-warn">
              ⚠ <strong>Research Use Only.</strong> Based on Cleveland Clinic data (1981–1984, n=302).
              Not validated for clinical decision-making. Always consult a qualified cardiologist.
            </div>
            """, unsafe_allow_html=True)

            st.markdown('<div class="sec-head">📊 Key Predictors (by Importance)</div>',
                        unsafe_allow_html=True)
            predictors = [
                ("🔬","ca — Vessels stenosed","Strongest predictor. Direct anatomical evidence of disease burden."),
                ("💓","thalach — Max heart rate","Lower HR achieved correlates with greater ischaemia."),
                ("📉","oldpeak — ST depression","Each +1 mm raises CAD odds ~2.3× (95% CI: 1.8–2.9)."),
                ("🫁","cp — Chest pain type","Asymptomatic (cp=3) has paradoxically high CAD prevalence."),
                ("📈","slope — ST slope","Downsloping = worst prognosis; upsloping = best."),
                ("👤","age × sex","Sex-specific risk threshold per ACC/AHA 2019."),
            ]
            for icon, feat, desc in predictors:
                st.markdown(f"""
                <div class="data-row">
                  <span class="data-label">{icon} {feat}</span>
                  <span style="color:#8894B8;font-size:0.76rem;max-width:55%;text-align:right;">{desc}</span>
                </div>""", unsafe_allow_html=True)

        with col_ref:
            st.markdown('<div class="sec-head">🤖 Model Card</div>',
                        unsafe_allow_html=True)
            st.markdown('<div class="card">', unsafe_allow_html=True)
            for k, v in [
                ("Type","Stacking Ensemble (Calibrated)"),
                ("Base models","GB · LR · RF · ExtraTrees"),
                ("Meta-learner","Logistic Regression"),
                ("Calibration","Isotonic regression (cv=3)"),
                ("Optimised for","Average Precision (PR-AUC)"),
                ("Training set","n=180 patients"),
                ("Test set","n=61 patients"),
                ("AUC-ROC",str(cfg.get('model_auc','—'))),
                ("Sensitivity",str(cfg.get('model_sensitivity','—'))),
                ("Decision threshold",f"{THRESHOLD:.2f}"),
                ("Cost matrix","FN = 10 × FP"),
                ("Standards","TRIPOD-AI"),
            ]:
                st.markdown(f"""
                <div class="data-row">
                  <span class="data-label">{k}</span>
                  <span class="data-val">{v}</span>
                </div>""", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

    else:
        # ══════════════════════════════════════════════════════
        # PREDICTION RESULTS
        # ══════════════════════════════════════════════════════
        ca_in   = np.nan if (isinstance(ca_sel,float) and np.isnan(ca_sel)) else ca_sel
        thal_in = np.nan if (isinstance(thal_sel,float) and np.isnan(thal_sel)) else thal_sel

        patient = pd.DataFrame([{
            'age':age,'sex':sex,'cp':cp,'trestbps':trestbps,'chol':chol,
            'fbs':fbs,'restecg':restecg,'thalach':thalach,'exang':exang,
            'oldpeak':oldpeak,'slope':slope,'ca':ca_in,'thal':thal_in
        }])
        X    = preprocessor.transform(engineer(patient))
        prob = float(model.predict_proba(X)[0,1])
        pred = int(prob >= THRESHOLD)

        if prob >= 0.50:
            rc,rl,rm = "high","HIGH RISK","num-high"
            ri,rt    = "🔴","tag-high"
        elif prob >= THRESHOLD:
            rc,rl,rm = "moderate","MODERATE RISK","num-moderate"
            ri,rt    = "🟡","tag-moderate"
        else:
            rc,rl,rm = "low","LOW RISK","num-low"
            ri,rt    = "🟢","tag-low"

        # ── Layout ────────────────────────────────────────────
        col_res, col_detail = st.columns([5, 4], gap="large")

        with col_res:
            st.markdown('<div class="sec-head">📊 Risk Assessment Result</div>',
                        unsafe_allow_html=True)

            # Result card
            st.markdown(f"""
            <div class="result-{rc}">
              <div class="{rm} big-num">{prob:.1%}</div>
              <span class="risk-tag {rt}">{ri} {rl}</span>
              <div class="result-caption">
                CAD probability &nbsp;·&nbsp; Decision threshold = {THRESHOLD:.0%}<br>
                {"Disease likely — refer for further cardiac evaluation"
                 if pred==1 else "Disease less likely — continue monitoring & primary prevention"}
              </div>
            </div>""", unsafe_allow_html=True)

            # Gauge
            st.markdown("<br>", unsafe_allow_html=True)
            fig_g, ax_g = plt.subplots(figsize=(5,2.6))
            fig_g.patch.set_facecolor('none'); ax_g.set_facecolor('none')
            gauge(prob)
            fg = gauge(prob)
            st.pyplot(fg, use_container_width=True)
            plt.close(fg)

            # Risk bar
            fb = riskbar(prob)
            st.pyplot(fb, use_container_width=True)
            plt.close(fb)
            st.markdown("""
            <div style="display:flex;justify-content:space-between;
                        font-size:0.68rem;color:#8894B8;font-family:monospace;margin-top:3px;">
              <span>◀ Low Risk</span><span>Moderate</span><span>High Risk ▶</span>
            </div>""", unsafe_allow_html=True)

            # Clinical action
            st.markdown('<div class="sec-head" style="margin-top:18px;">⚕ Suggested Clinical Action</div>',
                        unsafe_allow_html=True)
            if prob >= 0.50:
                st.markdown("""<div class="alert alert-danger">
                <strong>🚨 Urgent:</strong> Cardiology referral recommended. Consider stress echo,
                CT coronary angiography, or invasive catheterisation. Initiate guideline-directed
                therapy — aspirin, statin, beta-blocker (ACC/AHA 2019).</div>""",
                            unsafe_allow_html=True)
            elif prob >= THRESHOLD:
                st.markdown("""<div class="alert alert-warn">
                <strong>⚠ Action needed:</strong> Cardiology review within 2–4 weeks.
                Consider non-invasive cardiac imaging. Optimise BP, lipids, and glycaemia.
                Reassess in 3–6 months.</div>""", unsafe_allow_html=True)
            else:
                st.markdown("""<div class="alert alert-success">
                <strong>✓ Continue monitoring:</strong> Low-risk profile. Primary prevention:
                lifestyle modification, BP and cholesterol targets. Routine follow-up
                at next scheduled primary care visit.</div>""", unsafe_allow_html=True)

        with col_detail:
            # ── Clinical flags ────────────────────────────────
            st.markdown('<div class="sec-head">⚑ Clinical Flags</div>',
                        unsafe_allow_html=True)
            exp = 220 - age
            flags = []
            if thalach < 0.85*exp:
                flags.append(("warn",f"Chronotropic incompetence: {thalach} bpm < 85% predicted ({int(0.85*exp)} bpm)"))
            if oldpeak >= 2.0:
                flags.append(("danger",f"Severe ST depression: {oldpeak:.1f} mm ≥ 2.0 mm — severe ischaemia threshold"))
            elif oldpeak >= 1.0:
                flags.append(("info",f"Positive stress test: ST depression {oldpeak:.1f} mm ≥ 1.0 mm"))
            if slope==0 and oldpeak>=2.0 and exang==1:
                flags.append(("danger","High-risk ETT composite: downsloping + severe ST depression + exertional angina"))
            if trestbps >= 180:
                flags.append(("danger",f"Hypertensive crisis: {trestbps} mm Hg ≥ 180 mm Hg"))
            elif trestbps >= 140:
                flags.append(("warn",f"Stage 2 hypertension: {trestbps} mm Hg (AHA 2019)"))
            elif trestbps >= 130:
                flags.append(("info",f"Stage 1 hypertension: {trestbps} mm Hg (AHA 2019)"))
            if chol >= 240:
                flags.append(("info",f"High cholesterol: {chol} mg/dL ≥ 240 (ATPIII High)"))
            if (sex==1 and age>=45) or (sex==0 and age>=55):
                flags.append(("info",f"Age-sex threshold met: {'Male ≥45' if sex==1 else 'Female ≥55'} yrs (ACC/AHA 2019)"))
            if cp==3:
                flags.append(("info","Asymptomatic cp=3: paradoxically higher CAD prevalence (silent ischaemia)"))

            if flags:
                for ftype, fmsg in flags:
                    icon = "🔴" if ftype=="danger" else "⚠" if ftype=="warn" else "ℹ"
                    st.markdown(f'<div class="alert alert-{ftype}">{icon} {fmsg}</div>',
                                unsafe_allow_html=True)
            else:
                st.markdown('<div class="alert alert-success">✓ No major clinical flags for this patient profile.</div>',
                            unsafe_allow_html=True)

            # ── Computed features ─────────────────────────────
            st.markdown('<div class="sec-head" style="margin-top:16px;">🔧 Computed Clinical Features</div>',
                        unsafe_allow_html=True)
            st.markdown('<div class="card">', unsafe_allow_html=True)
            hr_res  = (220-age)-thalach
            pct_hr  = thalach/(220-age)
            fram    = 0.04*age+0.32*sex+0.18*(chol/50)+0.13*(trestbps/20)+0.11*fbs
            metab   = fbs+int(trestbps>=130)+int(chol>=200)
            hi_ett  = int(oldpeak>=2.0 and slope==0 and exang==1)
            for nm, vl in [
                ("HR Reserve",         f"{hr_res:.0f} bpm"),
                ("% Max HR achieved",  f"{pct_hr:.1%}"),
                ("Chrono incompetence","Yes ⚠" if pct_hr<0.85 else "No ✓"),
                ("Framingham proxy",   f"{fram:.3f}"),
                ("Metabolic score",    f"{metab} / 3"),
                ("High-risk ETT",      "Yes 🔴" if hi_ett else "No ✓"),
                ("BP elevated",        "Yes" if trestbps>=130 else "No"),
                ("Chol elevated",      "Yes" if chol>=200 else "No"),
            ]:
                st.markdown(f"""
                <div class="data-row">
                  <span class="data-label">{nm}</span>
                  <span class="data-val">{vl}</span>
                </div>""", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

            # ── Input summary ─────────────────────────────────
            st.markdown('<div class="sec-head" style="margin-top:4px;">📋 Input Summary</div>',
                        unsafe_allow_html=True)
            st.markdown('<div class="card">', unsafe_allow_html=True)
            ca_d   = "NaN" if np.isnan(ca_in) else str(int(ca_in))
            thal_d = "NaN" if np.isnan(thal_in) else {1.0:"Fixed",2.0:"Normal",3.0:"Reversible"}.get(thal_in,"?")
            for nm, vl in [
                ("Age",         f"{age} yrs"),
                ("Sex",         "Male" if sex==1 else "Female"),
                ("Chest Pain",  {0:"Typical",1:"Atypical",2:"Non-anginal",3:"Asymptomatic"}[cp]),
                ("Resting BP",  f"{trestbps} mmHg"),
                ("Cholesterol", f"{chol} mg/dL"),
                ("FBS >120",    "Yes" if fbs else "No"),
                ("Resting ECG", {0:"Normal",1:"ST-T abnl",2:"LVH"}[restecg]),
                ("Max HR",      f"{thalach} bpm"),
                ("Ex. Angina",  "Yes" if exang else "No"),
                ("ST Depr.",    f"{oldpeak:.1f} mm"),
                ("ST Slope",    {0:"Downsloping",1:"Flat",2:"Upsloping"}[slope]),
                ("ca",          ca_d),
                ("thal",        thal_d),
            ]:
                st.markdown(f"""
                <div class="data-row">
                  <span class="data-label">{nm}</span>
                  <span class="data-val">{vl}</span>
                </div>""", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────
st.markdown(f"""
<div class="footer">
  <strong>CardioRisk AI</strong> &nbsp;·&nbsp; HayMedics Academy
  &nbsp;·&nbsp; Cleveland Clinic Foundation Dataset (Detrano 1989)
  &nbsp;·&nbsp; Stacking Ensemble · TRIPOD-AI Compliant
  &nbsp;·&nbsp; <em>⚠ Research Use Only — Not for Clinical Decision-Making</em>
</div>
""", unsafe_allow_html=True)
