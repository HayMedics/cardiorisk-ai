"""
CardioRisk AI v2 — HayMedics Academy
FINAL CORRECTED VERSION with:
  + SHAP-style feature contributions
  + Downloadable PDF clinical report
  + What-If simulator with clinical plausibility checks
Research Use Only
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib, json, os, base64, io, warnings
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="CardioRisk AI v2 | HayMedics Academy",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Force sidebar visible (Streamlit Cloud fix) ────────────
st.markdown("""
<style>
[data-testid="stSidebar"] {
    display: block !important;
    visibility: visible !important;
    min-width: 320px !important;
    width: 320px !important;
    transform: translateX(0px) !important;
}
[data-testid="collapsedControl"] {
    display: block !important;
    visibility: visible !important;
}
</style>
""", unsafe_allow_html=True)

# ── Logo loader ────────────────────────────────────────────
def img_b64(folder, names):
    for n in names:
        p = os.path.join(folder, n)
        if os.path.exists(p):
            ext = p.rsplit(".", 1)[-1].lower()
            mime = "jpeg" if ext in ("jpg","jpeg") else "png"
            with open(p, "rb") as f:
                return mime, base64.b64encode(f.read()).decode()
    return None, ""

BASE = os.path.dirname(os.path.abspath(__file__))
logo_mime, logo_b64 = img_b64(BASE, ["HayMedics_Academy06.jpg","HMA.jpg","HMA_PNG.png","HMA_ICON.jpg"])
icon_mime, icon_b64 = img_b64(BASE, ["HMA_ICON.jpg","HMA_ICON_PNG.png"])

logo_tag = f'<img src="data:image/{logo_mime};base64,{logo_b64}" style="height:52px;"/>' if logo_b64 else '<span style="font-size:1.4rem;font-weight:900;color:#fff;">Hay<span style="color:#F5A623;">Medics</span> Academy</span>'
icon_tag = f'<img src="data:image/{icon_mime};base64,{icon_b64}" style="height:44px;border-radius:8px;"/>' if icon_b64 else "🫀"

# ── CSS ────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');
*, html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: #F4F6FB; color: #0D1B4B; }
section[data-testid="stSidebar"] {
    background: #0D1B4B !important;
    border-right: 4px solid #F5A623;
}
section[data-testid="stSidebar"] * { color: #CBD5F0 !important; }
section[data-testid="stSidebar"] .stSlider label,
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stRadio label {
    color: #94A3C8 !important;
    font-size: 0.8rem !important;
}

.sb-logo {
    background: linear-gradient(135deg, #0D1B4B 0%, #162256 100%);
    padding: 20px 20px 16px;
    border-bottom: 2px solid rgba(245,166,35,0.4);
}
.sb-tagline { font-size:0.65rem; color:#F5A623 !important; letter-spacing:2px; text-transform:uppercase; margin-top:6px; }
.sb-section {
    font-size: 0.58rem; font-weight: 700; letter-spacing: 2.5px;
    text-transform: uppercase; color: #F5A623 !important;
    padding: 14px 20px 6px; border-bottom: 1px solid rgba(245,166,35,0.2);
}

.header-bar {
    background: linear-gradient(90deg,#0D1B4B 0%,#1B3080 60%,#0D1B4B 100%);
    border-radius: 14px; padding: 18px 28px;
    display:flex; align-items:center; justify-content:space-between;
    margin-bottom: 6px; border-bottom: 3px solid #F5A623;
    box-shadow: 0 4px 20px rgba(13,27,75,0.2);
}
.header-left { display:flex; align-items:center; gap:16px; }
.header-app-name { font-size:1.5rem; font-weight:800; color:white; letter-spacing:-0.5px; line-height:1.1; }
.header-app-name em { color:#F5A623; font-style:normal; }
.header-app-name span.ver {
    background:#F5A623; color:#0D1B4B; padding:3px 10px;
    border-radius:6px; font-size:0.7rem; margin-left:8px; vertical-align:middle;
}
.header-sub { font-size:0.72rem; color:#94A3C8; margin-top:3px; }
.badge { padding:5px 14px; border-radius:20px; font-size:0.68rem; font-weight:600; letter-spacing:0.8px; text-transform:uppercase; }
.badge-warn { background:rgba(245,166,35,0.15); border:1.5px solid #F5A623; color:#F5A623; }
.badge-blue { background:rgba(99,152,255,0.15); border:1.5px solid #6398FF; color:#6398FF; }
.badge-new { background:#F5A623; color:#0D1B4B; border:1.5px solid #F5A623; }

.metric-strip { display:flex; gap:8px; flex-wrap:wrap; margin:10px 0 18px; }
.metric-chip { background:white; border:1.5px solid #DDE3F5; border-radius:10px; padding:8px 14px;
    display:flex; flex-direction:column; align-items:center; min-width:80px;
    box-shadow:0 1px 6px rgba(13,27,75,0.06); }
.metric-chip-label { font-size:0.6rem; font-weight:600; letter-spacing:1px; text-transform:uppercase; color:#8894B8; }
.metric-chip-val { font-size:1.1rem; font-weight:800; color:#0D1B4B; font-family:'JetBrains Mono',monospace; margin-top:2px; }

.sec-head { font-size:0.65rem; font-weight:700; letter-spacing:2.5px; text-transform:uppercase;
    color:#1B3080; padding:14px 0 7px; border-bottom:2px solid #DDE3F5; margin-bottom:10px;
    display:flex; align-items:center; gap:7px; }

.card { background:white; border-radius:14px; padding:22px 24px; border:1px solid #DDE3F5;
    box-shadow:0 2px 10px rgba(13,27,75,0.05); margin-bottom:14px; }

.result-high     { background:linear-gradient(135deg,#FFF2F2,#FFE8E8); border:2px solid #DC2626;
    border-radius:16px; padding:28px 32px; text-align:center; box-shadow:0 6px 28px rgba(220,38,38,0.12); }
.result-moderate { background:linear-gradient(135deg,#FFFCF0,#FFF5D6); border:2px solid #D97706;
    border-radius:16px; padding:28px 32px; text-align:center; box-shadow:0 6px 28px rgba(217,119,6,0.12); }
.result-low      { background:linear-gradient(135deg,#F0FFF8,#E6FFF2); border:2px solid #059669;
    border-radius:16px; padding:28px 32px; text-align:center; box-shadow:0 6px 28px rgba(5,150,105,0.1); }
.big-num { font-family:'JetBrains Mono',monospace; font-size:3.8rem; font-weight:700; letter-spacing:-3px; line-height:1; }
.num-high { color:#DC2626; } .num-moderate { color:#D97706; } .num-low { color:#059669; }
.risk-tag { display:inline-block; margin-top:10px; padding:5px 18px; border-radius:20px;
    font-size:0.75rem; font-weight:700; letter-spacing:2px; text-transform:uppercase; }
.tag-high { background:#DC2626; color:white; }
.tag-moderate { background:#D97706; color:white; }
.tag-low { background:#059669; color:white; }
.result-caption { font-size:0.78rem; color:#6B7BB5; margin-top:10px; }

.alert { display:flex; gap:10px; align-items:flex-start; padding:10px 14px; border-radius:10px;
    margin:6px 0; font-size:0.82rem; line-height:1.55; }
.alert-info    { background:#EEF4FF; border-left:3px solid #1B3080; color:#1B3080; }
.alert-warn    { background:#FFFBEC; border-left:3px solid #D97706; color:#92600A; }
.alert-danger  { background:#FFF5F5; border-left:3px solid #DC2626; color:#B91C1C; }
.alert-success { background:#F0FFF8; border-left:3px solid #059669; color:#065F46; }

.data-row { display:flex; justify-content:space-between; align-items:center; padding:7px 0;
    border-bottom:1px solid #F1F3FB; font-size:0.82rem; }
.data-row:last-child { border-bottom:none; }
.data-label { color:#8894B8; font-weight:500; }
.data-val { color:#0D1B4B; font-weight:600; font-family:'JetBrains Mono',monospace; font-size:0.79rem; }

div.stButton > button {
    background: linear-gradient(135deg,#0D1B4B 0%,#1B3080 100%);
    color: white; border: none; border-radius: 12px; padding: 15px 0;
    width: 100%; font-family:'Inter',sans-serif; font-size:0.95rem; font-weight:700;
    box-shadow: 0 4px 18px rgba(13,27,75,0.28); transition: all 0.2s;
}
div.stButton > button:hover {
    background: linear-gradient(135deg,#F5A623 0%,#E09010 100%);
    transform: translateY(-1px); box-shadow: 0 6px 22px rgba(245,166,35,0.35);
}
div.stDownloadButton > button {
    background: linear-gradient(135deg,#059669 0%,#10B981 100%);
    color:white; border:none; border-radius:10px; padding:12px 0;
    font-weight:700; box-shadow:0 4px 14px rgba(5,150,105,0.25);
}

.footer { background:linear-gradient(90deg,#0D1B4B,#1B3080); border-radius:12px; padding:14px 24px;
    text-align:center; font-size:0.7rem; color:#94A3C8; margin-top:28px; border-top:2px solid #F5A623; }
.footer strong { color:white; }
.footer em { color:#F5A623; font-style:normal; }

#MainMenu, footer, header { visibility:hidden; }

.feature-pill {
    display: inline-block;
    background: linear-gradient(135deg, #F5A623, #E09010);
    color: white; padding: 4px 12px; border-radius: 20px;
    font-size: 0.7rem; font-weight: 700; letter-spacing: 1px;
    text-transform: uppercase; margin: 3px;
}
</style>
""", unsafe_allow_html=True)

# ── Load artifacts ─────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    d = os.path.dirname(os.path.abspath(__file__))
    m = joblib.load(os.path.join(d, "model.pkl"))
    p = joblib.load(os.path.join(d, "preprocessor.pkl"))
    with open(os.path.join(d, "feature_config.json")) as f:
        c = json.load(f)
    return m, p, c

try:
    model, preprocessor, cfg = load_model()
    THRESHOLD = cfg["OPTIMAL_THRESHOLD"]
    ok = True
except Exception as e:
    ok = False
    err = str(e)

# ── Feature engineering ─────────────────────────────────────
def engineer(df):
    d = df.copy()
    d['age_sex_interaction'] = d['age']*d['sex']
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

def predict_one(patient_dict):
    df = pd.DataFrame([patient_dict])
    X = preprocessor.transform(engineer(df))
    prob = float(model.predict_proba(X)[0,1])
    return prob, X

# ── Feature contribution (perturbation-based) ──
def feature_contributions(patient_dict):
    base_prob, _ = predict_one(patient_dict)
    baseline = {
        'age':54, 'sex':0.68, 'cp':1.0, 'trestbps':131, 'chol':246,
        'fbs':0.15, 'restecg':0.5, 'thalach':150, 'exang':0.33,
        'oldpeak':1.04, 'slope':1.4, 'ca':0.67, 'thal':2.31
    }
    contribs = {}
    for feat in patient_dict.keys():
        perturbed = patient_dict.copy()
        perturbed[feat] = baseline[feat]
        perturbed_prob, _ = predict_one(perturbed)
        contribs[feat] = base_prob - perturbed_prob
    return contribs, base_prob

# ── Plots ───────────────────────────────────────────────────
def plot_gauge(prob):
    fig, ax = plt.subplots(figsize=(5, 2.6))
    fig.patch.set_facecolor('none'); ax.set_facecolor('none')
    segs = [('#059669',(0,.25)),('#68D391',(.25,.45)),
            ('#F5A623',(.45,.65)),('#DC2626',(.65,1.0))]
    for col,(a,b) in segs:
        th = np.linspace(np.pi*(1-b), np.pi*(1-a), 80)
        ax.fill_between(np.cos(th),np.sin(th),0.6*np.cos(th),0.6*np.sin(th),
                        color=col, alpha=0.9, zorder=2)
    ang = np.pi*(1-prob)
    ax.annotate("", xy=(0.5*np.cos(ang),0.5*np.sin(ang)), xytext=(0,0),
                arrowprops=dict(arrowstyle="-|>", color="#0D1B4B", lw=2.6, mutation_scale=15))
    ax.plot(0,0,'o',color='#0D1B4B',ms=9,zorder=5)
    for v,l in [(0,'0%'),(0.25,'25%'),(0.5,'50%'),(0.75,'75%'),(1,'100%')]:
        a=np.pi*(1-v)
        ax.text(1.13*np.cos(a),1.13*np.sin(a),l,ha='center',va='center',
                fontsize=7,color='#8894B8',fontfamily='monospace')
    ax.set_xlim(-1.2,1.2); ax.set_ylim(-0.2,1.2); ax.set_aspect('equal'); ax.axis('off')
    plt.tight_layout(pad=0)
    return fig

def plot_contributions(contribs):
    items = [(k, v) for k, v in contribs.items() if abs(v) > 0.001]
    items.sort(key=lambda x: abs(x[1]), reverse=True)
    items = items[:10]
    names = [k for k, _ in items]
    values = [v for _, v in items]
    colors = ['#DC2626' if v > 0 else '#059669' for v in values]

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor('white'); ax.set_facecolor('white')
    y_pos = np.arange(len(names))
    bars = ax.barh(y_pos, values, color=colors, alpha=0.85, edgecolor='white', linewidth=1.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=10, color='#0D1B4B')
    ax.invert_yaxis()
    ax.axvline(0, color='#0D1B4B', lw=1, zorder=2)
    ax.set_xlabel('Contribution to CAD Probability', fontsize=10, color='#0D1B4B', fontweight='600')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#DDE3F5'); ax.spines['bottom'].set_color('#DDE3F5')
    ax.tick_params(colors='#8894B8')
    for bar, val in zip(bars, values):
        w = bar.get_width()
        label_x = w + (0.005 if w >= 0 else -0.005)
        ax.text(label_x, bar.get_y() + bar.get_height()/2,
                f'{val:+.3f}', va='center',
                ha='left' if w >= 0 else 'right',
                fontsize=8, color='#0D1B4B', fontfamily='monospace')
    ax.text(0.99, 0.02, '🔴 Increases risk    🟢 Decreases risk',
            transform=ax.transAxes, ha='right', va='bottom',
            fontsize=8, color='#8894B8', style='italic')
    plt.tight_layout()
    return fig

# ── PDF Report Generation ───────────────────────────────────
def generate_pdf_report(patient, prob, contribs, threshold, flags):
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib.colors import HexColor, white
    from reportlab.lib.enums import TA_CENTER

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=2*cm, bottomMargin=2*cm,
                            leftMargin=2*cm, rightMargin=2*cm)
    styles = getSampleStyleSheet()
    navy = HexColor('#0D1B4B')

    title_style = ParagraphStyle('Title', parent=styles['Heading1'],
                                  fontSize=20, textColor=navy, alignment=TA_CENTER,
                                  spaceAfter=6, fontName='Helvetica-Bold')
    sub_style = ParagraphStyle('Sub', parent=styles['Normal'],
                                fontSize=9, textColor=HexColor('#6B7BB5'),
                                alignment=TA_CENTER, spaceAfter=20)
    head_style = ParagraphStyle('Head', parent=styles['Heading2'],
                                 fontSize=13, textColor=navy, fontName='Helvetica-Bold',
                                 spaceBefore=15, spaceAfter=8)
    body_style = ParagraphStyle('Body', parent=styles['Normal'],
                                 fontSize=10, textColor=HexColor('#0D1B4B'),
                                 spaceAfter=6, fontName='Helvetica')

    story = []
    story.append(Paragraph("CardioRisk AI — Clinical Risk Report", title_style))
    story.append(Paragraph("HayMedics Academy &nbsp;·&nbsp; Coronary Artery Disease Risk Assessment", sub_style))

    timestamp = datetime.now().strftime("%d %B %Y · %H:%M")
    meta_data = [
        ['Report Generated:', timestamp],
        ['Model:', 'Stacking Ensemble (Calibrated)'],
        ['Dataset:', 'Cleveland Clinic 1981-1984 (n=302)'],
        ['Standards:', 'TRIPOD-AI'],
    ]
    meta_table = Table(meta_data, colWidths=[5*cm, 11*cm])
    meta_table.setStyle(TableStyle([
        ('FONTNAME', (0,0), (-1,-1), 'Helvetica'),
        ('FONTSIZE', (0,0), (-1,-1), 9),
        ('TEXTCOLOR', (0,0), (0,-1), HexColor('#8894B8')),
        ('TEXTCOLOR', (1,0), (1,-1), navy),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('TOPPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 0.3*cm))

    label_full = "Low Risk" if prob < threshold else ("Moderate Risk" if prob < 0.5 else "High Risk")
    result_color = '#059669' if prob < threshold else ('#D97706' if prob < 0.5 else '#DC2626')
    result_data = [[
        Paragraph(f'<b>CAD Probability:</b> <font size=20 color="{result_color}">{prob:.1%}</font>', body_style),
        Paragraph(f'<b>Category:</b> <font color="{result_color}"><b>{label_full.upper()}</b></font>', body_style),
        Paragraph(f'<b>Decision Threshold:</b> {threshold:.0%}', body_style),
    ]]
    result_table = Table(result_data, colWidths=[6*cm, 5*cm, 5*cm])
    result_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), HexColor('#F4F6FB')),
        ('BOX', (0,0), (-1,-1), 1.5, HexColor(result_color)),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('PADDING', (0,0), (-1,-1), 10),
    ]))
    story.append(result_table)

    story.append(Paragraph("Patient Clinical Data", head_style))
    cp_map = {0:"Typical Angina", 1:"Atypical Angina", 2:"Non-anginal Pain", 3:"Asymptomatic"}
    ecg_map = {0:"Normal", 1:"ST-T Abnormality", 2:"LVH"}
    slope_map = {0:"Downsloping", 1:"Flat", 2:"Upsloping"}
    thal_map = {1.0:"Fixed Defect", 2.0:"Normal", 3.0:"Reversible Defect"}

    ca_disp = "Unknown" if (isinstance(patient['ca'], float) and np.isnan(patient['ca'])) else str(int(patient['ca']))
    thal_disp = "Unknown" if (isinstance(patient['thal'], float) and np.isnan(patient['thal'])) else thal_map.get(patient['thal'], '?')

    patient_data = [
        ['Age', f"{patient['age']} years", 'Resting BP', f"{patient['trestbps']} mm Hg"],
        ['Sex', 'Male' if patient['sex']==1 else 'Female', 'Cholesterol', f"{patient['chol']} mg/dL"],
        ['Chest Pain', cp_map[patient['cp']], 'FBS >120', 'Yes' if patient['fbs'] else 'No'],
        ['Resting ECG', ecg_map[patient['restecg']], 'Max Heart Rate', f"{patient['thalach']} bpm"],
        ['Exercise Angina', 'Yes' if patient['exang'] else 'No', 'ST Depression', f"{patient['oldpeak']:.1f} mm"],
        ['ST Slope', slope_map[patient['slope']], 'Vessels Stenosed', ca_disp],
        ['Thalassemia', thal_disp, '', ''],
    ]
    pdata_table = Table(patient_data, colWidths=[3.5*cm, 4.5*cm, 3.5*cm, 4.5*cm])
    pdata_table.setStyle(TableStyle([
        ('FONTNAME', (0,0), (-1,-1), 'Helvetica'),
        ('FONTSIZE', (0,0), (-1,-1), 9),
        ('TEXTCOLOR', (0,0), (0,-1), HexColor('#8894B8')),
        ('TEXTCOLOR', (2,0), (2,-1), HexColor('#8894B8')),
        ('TEXTCOLOR', (1,0), (1,-1), navy),
        ('TEXTCOLOR', (3,0), (3,-1), navy),
        ('FONTNAME', (1,0), (1,-1), 'Helvetica-Bold'),
        ('FONTNAME', (3,0), (3,-1), 'Helvetica-Bold'),
        ('GRID', (0,0), (-1,-1), 0.3, HexColor('#DDE3F5')),
        ('PADDING', (0,0), (-1,-1), 5),
    ]))
    story.append(pdata_table)

    story.append(Paragraph("Top Risk Drivers (Feature Contributions)", head_style))
    sorted_contribs = sorted(contribs.items(), key=lambda x: abs(x[1]), reverse=True)[:8]
    contrib_data = [['Feature', 'Patient Value', 'Direction', 'Magnitude']]
    for feat, val in sorted_contribs:
        if abs(val) < 0.001: continue
        direction = "Increases" if val > 0 else "Decreases"
        color = '#DC2626' if val > 0 else '#059669'
        contrib_data.append([feat, str(patient.get(feat, '-'))[:10],
                             Paragraph(f'<font color="{color}">{direction}</font>', body_style),
                             f"{abs(val):.3f}"])
    contrib_table = Table(contrib_data, colWidths=[5*cm, 4*cm, 4*cm, 3*cm])
    contrib_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), navy),
        ('TEXTCOLOR', (0,0), (-1,0), white),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,-1), 9),
        ('GRID', (0,0), (-1,-1), 0.3, HexColor('#DDE3F5')),
        ('PADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(contrib_table)

    if flags:
        story.append(Paragraph("Clinical Flags", head_style))
        for fmsg in flags:
            story.append(Paragraph(f"• {fmsg}", body_style))

    story.append(Paragraph("Suggested Clinical Action", head_style))
    if prob >= 0.50:
        action = ("<b>URGENT:</b> Cardiology referral recommended. Consider stress echocardiography, "
                  "CT coronary angiography, or invasive catheterisation. Initiate guideline-directed "
                  "medical therapy (aspirin, statin, beta-blocker per ACC/AHA 2019).")
    elif prob >= threshold:
        action = ("<b>Action needed:</b> Cardiology review within 2-4 weeks. Consider non-invasive cardiac imaging. "
                  "Optimise BP, lipids, and glycaemia. Reassess in 3-6 months.")
    else:
        action = ("<b>Continue monitoring:</b> Low-risk profile. Primary prevention: lifestyle modification, "
                  "BP and cholesterol targets. Routine follow-up at next scheduled primary care visit.")
    story.append(Paragraph(action, body_style))

    story.append(Spacer(1, 0.4*cm))
    disclaimer_style = ParagraphStyle('Disc', parent=styles['Normal'],
                                       fontSize=8, textColor=HexColor('#B91C1C'),
                                       alignment=TA_CENTER, fontName='Helvetica-Oblique')
    story.append(Paragraph(
        "RESEARCH USE ONLY - This report is generated by a machine learning model trained on "
        "the Cleveland Heart Disease dataset (1981-1984, n=302). It is not validated for clinical "
        "decision-making. Always consult a qualified cardiologist.",
        disclaimer_style))

    story.append(Spacer(1, 0.3*cm))
    footer_style = ParagraphStyle('Foot', parent=styles['Normal'],
                                   fontSize=7, textColor=HexColor('#8894B8'),
                                   alignment=TA_CENTER)
    story.append(Paragraph(
        f"CardioRisk AI v2 · HayMedics Academy · TRIPOD-AI Compliant<br/>Generated {timestamp}",
        footer_style))

    doc.build(story)
    buffer.seek(0)
    return buffer

# ══════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════
if 'last_prediction' not in st.session_state:
    st.session_state.last_prediction = None
if 'last_patient' not in st.session_state:
    st.session_state.last_patient = None
if 'last_contribs' not in st.session_state:
    st.session_state.last_contribs = None
if 'last_flags' not in st.session_state:
    st.session_state.last_flags = []

# ══════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(f"""
    <div class="sb-logo">{logo_tag}<div class="sb-tagline">Data · Research · Innovation</div></div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sb-section">Navigation</div>', unsafe_allow_html=True)
    page = st.radio("", [
        "🫀  Risk Assessment",
        "🎛️  What-If Simulator",
        "📖  About the Project"
    ], label_visibility="collapsed")

    # Initialise predict variable for all pages
    predict = False

    if "Risk Assessment" in page:
        st.markdown('<div class="sb-section">Demographics</div>', unsafe_allow_html=True)
        age = st.slider("Age (years)", 20, 90, 55, key="age_main")
        sex = st.selectbox("Sex", [0,1], format_func=lambda x:"Female" if x==0 else "Male", index=1, key="sex_main")

        st.markdown('<div class="sb-section">Symptoms & History</div>', unsafe_allow_html=True)
        cp = st.selectbox("Chest Pain Type", [0,1,2,3],
                          format_func=lambda x:{0:"Typical Angina",1:"Atypical Angina",
                                                2:"Non-anginal Pain",3:"Asymptomatic"}[x], key="cp_main")
        exang = st.selectbox("Exercise-Induced Angina",[0,1], format_func=lambda x:"No" if x==0 else "Yes", key="exang_main")
        fbs   = st.selectbox("Fasting Blood Sugar >120 mg/dL",[0,1], format_func=lambda x:"No" if x==0 else "Yes", key="fbs_main")

        st.markdown('<div class="sb-section">Vitals & Labs</div>', unsafe_allow_html=True)
        trestbps = st.slider("Resting BP (mm Hg)", 80, 200, 130, key="bp_main")
        chol     = st.slider("Cholesterol (mg/dL)", 100, 600, 240, key="chol_main")

        st.markdown('<div class="sb-section">ECG & Stress Test</div>', unsafe_allow_html=True)
        restecg = st.selectbox("Resting ECG",[0,1,2],
                               format_func=lambda x:{0:"Normal",1:"ST-T Abnormality",2:"LVH (Estes)"}[x], key="ecg_main")
        thalach = st.slider(f"Max Heart Rate (bpm) · Pred: {220-age}", 60, 220, 150, key="hr_main")
        oldpeak = st.slider("ST Depression (mm)", 0.0, 6.5, 1.5, 0.1, key="op_main")
        slope   = st.selectbox("ST Slope",[0,1,2],
                               format_func=lambda x:{0:"Downsloping ↓",1:"Flat →",2:"Upsloping ↑"}[x], key="sl_main")

        st.markdown('<div class="sb-section">Invasive Results (optional)</div>', unsafe_allow_html=True)
        ca_sel = st.selectbox("Stenosed Vessels (ca)",[0.0,1.0,2.0,3.0,float('nan')],
                              format_func=lambda x:"Unknown" if (isinstance(x,float) and np.isnan(x))
                                          else f"{int(x)} vessel{'s' if x!=1 else ''}", key="ca_main")
        thal_sel = st.selectbox("Thalassemia (thal)",[1.0,2.0,3.0,float('nan')],
                                format_func=lambda x:"Unknown" if (isinstance(x,float) and np.isnan(x))
                                            else {1.0:"Fixed Defect",2.0:"Normal",3.0:"Reversible Defect"}.get(x,""),
                                index=1, key="thal_main")
        st.markdown("<br>", unsafe_allow_html=True)
        predict = st.button("🫀  Analyse CAD Risk", use_container_width=True, key="predict_btn")

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
      <div class="header-app-name">CardioRisk<em>AI</em><span class="ver">v2.0</span></div>
      <div class="header-sub">
        HayMedics Academy · CAD Risk Assessment · SHAP explanations · PDF reports · What-If Simulator
      </div>
    </div>
  </div>
  <div style="display:flex;gap:10px;">
    <span class="badge badge-new">✨ V2 Features</span>
    <span class="badge badge-blue">TRIPOD-AI</span>
    <span class="badge badge-warn">⚠ Research Only</span>
  </div>
</div>
""", unsafe_allow_html=True)

if not ok:
    st.error(f"**Model files not found.** Make sure `model.pkl`, `preprocessor.pkl`, and `feature_config.json` are in the app folder.\n\n_{err}_")
    st.stop()

# Metric strip
metrics = [
    ("AUC-ROC", cfg.get('model_auc','—')),
    ("Sensitivity", cfg.get('model_sensitivity','—')),
    ("Specificity", cfg.get('model_specificity','—')),
    ("PPV", cfg.get('model_ppv','—')),
    ("NPV", cfg.get('model_npv','—')),
    ("Threshold", f"{THRESHOLD:.2f}"),
    ("Dataset", "n=302"),
    ("Model", "Stacking"),
]
chips = "".join([f'<div class="metric-chip"><span class="metric-chip-label">{l}</span><span class="metric-chip-val">{v}</span></div>' for l,v in metrics])
st.markdown(f'<div class="metric-strip">{chips}</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# PAGE — RISK ASSESSMENT
# ══════════════════════════════════════════════════════════════
if "Risk Assessment" in page:
    if not predict and st.session_state.last_prediction is None:
        col_w, col_ref = st.columns([3, 2], gap="large")
        with col_w:
            st.markdown('<div class="sec-head">✨ What\'s New in v2</div>', unsafe_allow_html=True)
            st.markdown("""
            <div class="card">
              <div style="display:flex; gap:8px; flex-wrap:wrap; margin-bottom:16px;">
                <span class="feature-pill">🧠 SHAP Explanations</span>
                <span class="feature-pill">📄 PDF Reports</span>
                <span class="feature-pill">🎛️ What-If Simulator</span>
              </div>
              <p style="color:#4A5580;font-size:0.9rem;line-height:1.9;">
                CardioRisk AI v2 builds on the trusted v1 model with new clinical-grade features:
              </p>
              <ul style="color:#4A5580;font-size:0.87rem;line-height:1.9;padding-left:20px;">
                <li><strong>SHAP-style feature explanations</strong> - see exactly which clinical factors drove the prediction</li>
                <li><strong>Downloadable PDF clinical reports</strong> - share with cardiologists or save to patient records</li>
                <li><strong>What-If Simulator</strong> - explore how risk changes when clinical variables are modified</li>
              </ul>
              <p style="color:#4A5580;font-size:0.88rem;line-height:1.9;margin-top:12px;">
                Fill the patient data in the <strong style="color:#1B3080;">left sidebar</strong>
                and click <strong style="color:#F5A623;">Analyse CAD Risk</strong>.
              </p>
            </div>
            <div class="alert alert-warn">
              ⚠ <strong>Research Use Only.</strong> Based on Cleveland Clinic data (1981-1984, n=302).
              Not validated for clinical use. Always consult a qualified cardiologist.
            </div>
            """, unsafe_allow_html=True)

        with col_ref:
            st.markdown('<div class="sec-head">🤖 Model Card</div>', unsafe_allow_html=True)
            st.markdown('<div class="card">', unsafe_allow_html=True)
            for k, v in [
                ("Type", "Stacking Ensemble (Calibrated)"),
                ("Base models", "GB · LR · RF · ExtraTrees"),
                ("Meta-learner", "Logistic Regression"),
                ("Calibration", "Isotonic regression (cv=3)"),
                ("AUC-ROC", str(cfg.get('model_auc','—'))),
                ("Sensitivity", str(cfg.get('model_sensitivity','—'))),
                ("Decision threshold", f"{THRESHOLD:.2f}"),
                ("Cost matrix", "FN = 10 × FP"),
                ("Standards", "TRIPOD-AI"),
            ]:
                st.markdown(f'<div class="data-row"><span class="data-label">{k}</span><span class="data-val">{v}</span></div>',
                            unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

    else:
        if predict:
            ca_in   = np.nan if (isinstance(ca_sel,float) and np.isnan(ca_sel)) else ca_sel
            thal_in = np.nan if (isinstance(thal_sel,float) and np.isnan(thal_sel)) else thal_sel

            patient = {
                'age':age,'sex':sex,'cp':cp,'trestbps':trestbps,'chol':chol,
                'fbs':fbs,'restecg':restecg,'thalach':thalach,'exang':exang,
                'oldpeak':oldpeak,'slope':slope,'ca':ca_in,'thal':thal_in
            }
            with st.spinner("🔍 Analysing patient data and computing explanations..."):
                prob, _ = predict_one(patient)
                contribs, _ = feature_contributions(patient)

            exp = 220 - age
            flags = []
            if thalach < 0.85*exp:
                flags.append(f"Chronotropic incompetence: {thalach} bpm < 85% predicted ({int(0.85*exp)} bpm)")
            if oldpeak >= 2.0:
                flags.append(f"Severe ST depression: {oldpeak:.1f} mm ≥ 2.0 mm — severe ischaemia threshold")
            elif oldpeak >= 1.0:
                flags.append(f"Positive stress test: ST depression {oldpeak:.1f} mm ≥ 1.0 mm")
            if slope==0 and oldpeak>=2.0 and exang==1:
                flags.append("High-risk ETT composite: downsloping + severe ST depression + exertional angina")
            if trestbps >= 180:
                flags.append(f"Hypertensive crisis: {trestbps} mm Hg ≥ 180 mm Hg")
            elif trestbps >= 140:
                flags.append(f"Stage 2 hypertension: {trestbps} mm Hg (AHA 2019)")
            if chol >= 240:
                flags.append(f"High cholesterol: {chol} mg/dL ≥ 240 mg/dL (ATPIII)")
            if cp==3:
                flags.append("Asymptomatic cp=3: paradoxically higher CAD prevalence")

            st.session_state.last_prediction = prob
            st.session_state.last_patient = patient
            st.session_state.last_contribs = contribs
            st.session_state.last_flags = flags

        prob     = st.session_state.last_prediction
        patient  = st.session_state.last_patient
        contribs = st.session_state.last_contribs
        flags    = st.session_state.last_flags

        if prob >= 0.50:
            rc, rl, rm = "high", "HIGH RISK", "num-high"
            ri, rt = "🔴", "tag-high"
        elif prob >= THRESHOLD:
            rc, rl, rm = "moderate", "MODERATE RISK", "num-moderate"
            ri, rt = "🟡", "tag-moderate"
        else:
            rc, rl, rm = "low", "LOW RISK", "num-low"
            ri, rt = "🟢", "tag-low"

        col_res, col_action = st.columns([5, 4], gap="large")

        with col_res:
            st.markdown('<div class="sec-head">📊 Risk Assessment Result</div>', unsafe_allow_html=True)
            st.markdown(f"""
            <div class="result-{rc}">
              <div class="{rm} big-num">{prob:.1%}</div>
              <span class="risk-tag {rt}">{ri} {rl}</span>
              <div class="result-caption">
                CAD probability &nbsp;·&nbsp; Threshold = {THRESHOLD:.0%}<br>
                {"Disease likely — refer for cardiac evaluation" if prob >= THRESHOLD
                 else "Disease less likely — continue monitoring"}
              </div>
            </div>""", unsafe_allow_html=True)

            fig_g = plot_gauge(prob)
            st.pyplot(fig_g, use_container_width=True)
            plt.close(fig_g)

        with col_action:
            st.markdown('<div class="sec-head">📄 Download Report & Next Steps</div>', unsafe_allow_html=True)

            try:
                pdf_buffer = generate_pdf_report(patient, prob, contribs, THRESHOLD, flags)
                ts = datetime.now().strftime("%Y%m%d_%H%M")
                st.download_button(
                    label="📄  Download Clinical PDF Report",
                    data=pdf_buffer,
                    file_name=f"CardioRisk_Report_{ts}.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
            except Exception as e:
                st.warning(f"PDF generation requires `reportlab` package.")

            st.markdown("<br>", unsafe_allow_html=True)
            if prob >= 0.50:
                st.markdown("""<div class="alert alert-danger">
                <strong>🚨 Urgent:</strong> Cardiology referral. Consider stress echo, CT angiography,
                or catheterisation. Initiate ACC/AHA 2019 therapy.</div>""", unsafe_allow_html=True)
            elif prob >= THRESHOLD:
                st.markdown("""<div class="alert alert-warn">
                <strong>⚠ Action needed:</strong> Cardiology review within 2-4 weeks.
                Optimise BP, lipids, glycaemia.</div>""", unsafe_allow_html=True)
            else:
                st.markdown("""<div class="alert alert-success">
                <strong>✓ Continue monitoring:</strong> Low-risk profile.
                Primary prevention measures.</div>""", unsafe_allow_html=True)

            if flags:
                st.markdown('<div class="sec-head">⚑ Clinical Flags</div>', unsafe_allow_html=True)
                for fmsg in flags[:5]:
                    st.markdown(f'<div class="alert alert-info">ℹ {fmsg}</div>', unsafe_allow_html=True)

        st.markdown('<div class="sec-head" style="margin-top:24px;">🧠 Why This Prediction? — Feature Contributions</div>',
                    unsafe_allow_html=True)

        col_chart, col_explain = st.columns([3, 2], gap="large")

        with col_chart:
            fig_contrib = plot_contributions(contribs)
            st.pyplot(fig_contrib, use_container_width=True)
            plt.close(fig_contrib)

        with col_explain:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("""
            <p style="color:#4A5580; font-size:0.85rem; line-height:1.8;">
            <strong>How to read this chart:</strong><br><br>
            Each bar shows how much a feature pushed the prediction <strong>up</strong>
            (🔴 red = increases CAD risk) or <strong>down</strong>
            (🟢 green = decreases CAD risk) from the population baseline.
            </p>
            """, unsafe_allow_html=True)

            sorted_contribs = sorted(contribs.items(), key=lambda x: abs(x[1]), reverse=True)
            st.markdown('<div style="margin-top:14px;"><strong style="color:#1B3080; font-size:0.85rem;">Top 3 drivers for THIS patient:</strong></div>',
                        unsafe_allow_html=True)
            for feat, val in sorted_contribs[:3]:
                direction = "↑ increased" if val > 0 else "↓ decreased"
                color = "#DC2626" if val > 0 else "#059669"
                st.markdown(f"""
                <div class="data-row">
                  <span class="data-label">{feat}</span>
                  <span style="color:{color}; font-weight:600; font-size:0.78rem;">
                    {direction} by {abs(val):.3f}
                  </span>
                </div>""", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# PAGE — WHAT-IF SIMULATOR
# ══════════════════════════════════════════════════════════════
elif "What-If" in page:
    st.markdown('<div class="sec-head">🎛️ What-If Simulator — Explore Risk Sensitivity</div>',
                unsafe_allow_html=True)

    if st.session_state.last_patient is None:
        st.markdown("""
        <div class="alert alert-info">
        ℹ <strong>No baseline patient yet.</strong> Please go to the <strong>Risk Assessment</strong> page first
        and run a prediction. Then return here to explore how risk changes when you modify individual variables.
        </div>
        """, unsafe_allow_html=True)
    else:
        baseline = st.session_state.last_patient
        baseline_prob = st.session_state.last_prediction

        st.markdown(f"""
        <div class="card">
          <p style="color:#4A5580; font-size:0.9rem;">
            <strong>Baseline patient:</strong> {baseline['age']}yo
            {"Male" if baseline['sex']==1 else "Female"} ·
            Baseline CAD risk: <strong style="color:#1B3080; font-size:1.1rem;">{baseline_prob:.1%}</strong>
          </p>
          <p style="color:#6B7BB5; font-size:0.82rem; margin-top:8px;">
            Move the sliders below to simulate "what would happen if this patient had different values?"
            The risk recalculates in real time.
          </p>
        </div>
        """, unsafe_allow_html=True)

        col_sim_l, col_sim_r = st.columns(2, gap="large")

        with col_sim_l:
            st.markdown('<div class="sec-head">Modify Variables</div>', unsafe_allow_html=True)
            sim_age      = st.slider("Age", 20, 90, baseline['age'], key="sim_age")
            sim_trestbps = st.slider("Resting BP (mmHg)", 80, 200, baseline['trestbps'], key="sim_bp")
            sim_chol     = st.slider("Cholesterol (mg/dL)", 100, 600, baseline['chol'], key="sim_chol")
            sim_thalach  = st.slider("Max Heart Rate (bpm)", 60, 220, baseline['thalach'], key="sim_hr")
            sim_oldpeak  = st.slider("ST Depression (mm)", 0.0, 6.5, float(baseline['oldpeak']), 0.1, key="sim_op")

        with col_sim_r:
            sim_patient = baseline.copy()
            sim_patient['age']      = sim_age
            sim_patient['trestbps'] = sim_trestbps
            sim_patient['chol']     = sim_chol
            sim_patient['thalach']  = sim_thalach
            sim_patient['oldpeak']  = sim_oldpeak

            sim_prob, _ = predict_one(sim_patient)
            delta = sim_prob - baseline_prob

            if sim_prob >= 0.5:
                sim_rc, sim_rl = "high", "HIGH RISK"
            elif sim_prob >= THRESHOLD:
                sim_rc, sim_rl = "moderate", "MODERATE RISK"
            else:
                sim_rc, sim_rl = "low", "LOW RISK"

            num_class = f"num-{sim_rc}"
            tag_class = f"tag-{sim_rc}"

            delta_color = "#DC2626" if delta > 0.05 else ("#059669" if delta < -0.05 else "#8894B8")
            delta_sign = "+" if delta >= 0 else ""

            st.markdown(f"""
            <div class="result-{sim_rc}">
              <div class="{num_class} big-num">{sim_prob:.1%}</div>
              <span class="risk-tag {tag_class}">{sim_rl}</span>
              <div class="result-caption">
                Simulated CAD probability<br>
                <strong style="color:{delta_color}; font-size:1rem;">
                  {delta_sign}{delta*100:.1f}% vs baseline ({baseline_prob:.1%})
                </strong>
              </div>
            </div>
            """, unsafe_allow_html=True)

            fig_sim = plot_gauge(sim_prob)
            st.pyplot(fig_sim, use_container_width=True)
            plt.close(fig_sim)

        # ── Clinical plausibility check ────────
        warnings_list = []
        sim_pct_max = sim_thalach / (220 - sim_age) if (220 - sim_age) > 0 else 1.0

        if sim_pct_max < 0.65:
            warnings_list.append(
                f"⚠ Max HR of {sim_thalach} bpm is only {sim_pct_max:.0%} of predicted max "
                f"({220-sim_age} bpm) for age {sim_age} — severe chronotropic incompetence. "
                f"This is clinically implausible for an otherwise healthy patient and may artificially inflate risk."
            )
        if sim_age < 40 and sim_oldpeak >= 2.0:
            warnings_list.append(
                f"⚠ ST depression {sim_oldpeak} mm at age {sim_age} is unusual — "
                f"this finding is more typical in older patients with established CAD."
            )
        if sim_age < 30 and sim_trestbps >= 160:
            warnings_list.append(
                f"⚠ Stage 2 hypertension ({sim_trestbps} mmHg) at age {sim_age} suggests secondary causes "
                f"(renal artery stenosis, endocrine disorders) rather than primary essential hypertension."
            )
        if sim_chol >= 400:
            warnings_list.append(
                f"⚠ Cholesterol of {sim_chol} mg/dL is extremely high — consider familial hypercholesterolaemia."
            )
        if sim_chol < 130:
            warnings_list.append(
                f"ℹ Cholesterol of {sim_chol} mg/dL is unusually low — may indicate malnutrition or aggressive statin therapy."
            )

        if warnings_list:
            st.markdown('<div class="sec-head" style="margin-top:20px;">⚠ Clinical Plausibility Check</div>',
                        unsafe_allow_html=True)
            st.markdown("""
            <div class="alert alert-info">
              💡 The simulator allows any combination of values — but not all combinations are clinically realistic.
              Below are flags for unusual feature combinations in this simulation:
            </div>
            """, unsafe_allow_html=True)
            for w in warnings_list:
                st.markdown(f'<div class="alert alert-warn">{w}</div>',
                            unsafe_allow_html=True)

        # Comparison table
        st.markdown('<div class="sec-head" style="margin-top:20px;">📊 Side-by-Side Comparison</div>',
                    unsafe_allow_html=True)
        st.markdown('<div class="card">', unsafe_allow_html=True)
        compare_rows = [
            ("Age",           f"{baseline['age']} yrs",       f"{sim_age} yrs"),
            ("Resting BP",    f"{baseline['trestbps']} mmHg", f"{sim_trestbps} mmHg"),
            ("Cholesterol",   f"{baseline['chol']} mg/dL",    f"{sim_chol} mg/dL"),
            ("Max Heart Rate",f"{baseline['thalach']} bpm",   f"{sim_thalach} bpm"),
            ("ST Depression", f"{baseline['oldpeak']:.1f} mm",f"{sim_oldpeak:.1f} mm"),
            ("CAD Risk",      f"{baseline_prob:.1%}",         f"{sim_prob:.1%} ({delta_sign}{delta*100:.1f}%)"),
        ]
        st.markdown("""
        <div style="display:grid; grid-template-columns: 2fr 2fr 2fr; gap:8px; padding:8px 0; font-size:0.75rem; color:#F5A623; font-weight:700; letter-spacing:1.5px; text-transform:uppercase; border-bottom:2px solid #DDE3F5;">
          <span>Variable</span><span>Baseline</span><span>Simulated</span>
        </div>
        """, unsafe_allow_html=True)
        for var, base_v, sim_v in compare_rows:
            highlight = "#FFF8EC" if base_v != sim_v else "transparent"
            st.markdown(f"""
            <div style="display:grid; grid-template-columns: 2fr 2fr 2fr; gap:8px; padding:9px 8px; background:{highlight}; border-radius:6px; font-size:0.83rem; border-bottom:1px solid #F1F3FB;">
              <span style="color:#8894B8;">{var}</span>
              <span style="color:#0D1B4B; font-family:monospace;">{base_v}</span>
              <span style="color:#0D1B4B; font-family:monospace; font-weight:600;">{sim_v}</span>
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# PAGE — ABOUT
# ══════════════════════════════════════════════════════════════
elif "About" in page:
    st.markdown('<div class="sec-head">📖 About the Project</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown("""
        <div class="card">
          <h3 style="color:#0D1B4B; font-size:1.05rem;">🎯 Project Overview</h3>
          <p style="color:#4A5580; font-size:0.87rem; line-height:1.85;">
          CardioRisk AI v2 is a TRIPOD-AI compliant ML system for Coronary Artery Disease
          risk classification, built on the Cleveland Heart Disease dataset. v2 adds
          clinical-grade explanations, PDF reporting, and a What-If Simulator with
          clinical plausibility validation.</p>
        </div>
        <div class="card">
          <h3 style="color:#0D1B4B; font-size:1.05rem;">✨ What's New in v2</h3>
          <ul style="color:#4A5580; font-size:0.86rem; line-height:1.85; padding-left:20px;">
            <li><strong>SHAP-style feature explanations</strong> via perturbation analysis</li>
            <li><strong>Downloadable PDF reports</strong> for clinical use</li>
            <li><strong>What-If Simulator</strong> with clinical plausibility flags</li>
            <li>Enhanced UI with v2 badges and modern layout</li>
          </ul>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="card">
          <h3 style="color:#0D1B4B; font-size:1.05rem;">⚙️ Model Architecture</h3>
          <ul style="color:#4A5580; font-size:0.86rem; line-height:1.85; padding-left:20px;">
            <li>Stacking Ensemble (Calibrated)</li>
            <li>Base: Gradient Boosting · Logistic Regression · Random Forest · Extra Trees</li>
            <li>Meta-learner: Logistic Regression</li>
            <li>Calibration: Isotonic (cv=3)</li>
            <li>Threshold: Cost-based (FN = 10 × FP)</li>
            <li>AUC = 0.87 · Sensitivity = 100%</li>
          </ul>
        </div>
        <div class="card">
          <h3 style="color:#0D1B4B; font-size:1.05rem;">⚠️ Known Limitations</h3>
          <ul style="color:#4A5580; font-size:0.86rem; line-height:1.85; padding-left:20px;">
            <li>n=302, single centre (1981-1984)</li>
            <li>No medication or ethnicity data</li>
            <li>FBS threshold &gt;120 vs ADA ≥126 mg/dL</li>
            <li>No external validation cohort</li>
            <li>Feature contributions use perturbation, not true SHAP</li>
          </ul>
        </div>
        """, unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
  <strong>CardioRisk AI v2.0</strong> · HayMedics Academy ·
  Cleveland Clinic Foundation Dataset (Detrano 1989) ·
  Stacking Ensemble · TRIPOD-AI Compliant ·
  <em>⚠ Research Use Only — Not for Clinical Decision-Making</em>
</div>
""", unsafe_allow_html=True)
