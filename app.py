"""
Credit Ledger - an underwriting desk for consumer loan applications.
Streamlit front-end for the Explainable Credit Risk notebook.

Run:  streamlit run app.py
Needs credit_risk_dataset.csv next to this file on the first run (the model is
trained once, then cached in model_artifacts.joblib).
"""
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # noqa: F401
from xgboost import DMatrix, XGBClassifier

BASE = Path(__file__).parent
DATA_PATH = BASE / "credit_risk_dataset.csv"
ARTIFACT_PATH = BASE / "model_artifacts.joblib"

NUMERIC_COLS = ["person_age", "person_income", "person_emp_length", "loan_amnt",
                "loan_int_rate", "loan_percent_income", "cb_person_cred_hist_length"]
CATEGORICAL_COLS = ["person_home_ownership", "loan_intent", "loan_grade",
                    "cb_person_default_on_file"]

# Best parameters found by RandomizedSearchCV in the notebook (section 10)
BEST_PARAMS = dict(colsample_bytree=0.9374, gamma=2.3914, learning_rate=0.1317,
                   max_depth=5, min_child_weight=7, n_estimators=366, subsample=0.8453)


# --------------------------------------------------------------------------
# Model: same steps as the notebook (clean -> split -> XGBoost -> calibrate)
# --------------------------------------------------------------------------
def _train_and_save():
    df = pd.read_csv(DATA_PATH)

    # Section 3 - validation & outlier handling
    df = df.drop_duplicates()
    df = df[(df["person_age"] >= 18) & (df["person_age"] <= 100)]
    df = df[df["person_emp_length"] <= df["person_age"]]
    df = df[df["person_emp_length"] <= 60]
    df = df[df["loan_amnt"] > 0]

    # Section 4 - features / target / split
    X = df.drop("loan_status", axis=1)
    y = df["loan_status"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    # Section 5 - class imbalance
    neg, pos = np.bincount(y_train)
    scale_pos_weight = neg / pos

    # Section 6 - preprocessing for XGBoost
    preprocessor = ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), NUMERIC_COLS),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))]), CATEGORICAL_COLS),
    ])
    xgb_pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", XGBClassifier(scale_pos_weight=scale_pos_weight,
                                     random_state=42, **BEST_PARAMS)),
    ])

    # Section 13 - probability calibration (sigmoid / Platt)
    calibrated = CalibratedClassifierCV(xgb_pipe, method="sigmoid", cv=5)
    calibrated.fit(X_train, y_train)

    # Section 12 - decision threshold = best F1 on the precision-recall curve.
    # NOTE: the notebook computed F1 as 2*(p - r)/(p + r), which is wrong and gave a
    # threshold of 1.0. The correct F1 is 2*p*r/(p + r), used here.
    prob = calibrated.predict_proba(X_test)[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y_test, prob)
    f1 = 2 * precisions * recalls / (precisions + recalls + 1e-9)
    threshold = float(thresholds[np.argmax(f1[:-1])])

    # Separate fitted pipeline, used only to explain individual decisions
    explain_pipe = Pipeline(xgb_pipe.steps)
    explain_pipe.fit(X_train, y_train)

    artifacts = {"model": calibrated, "threshold": threshold, "explain_pipe": explain_pipe}
    joblib.dump(artifacts, ARTIFACT_PATH)
    return artifacts


@st.cache_resource(show_spinner=False)
def load_artifacts():
    if ARTIFACT_PATH.exists():
        try:
            return joblib.load(ARTIFACT_PATH)
        except Exception:
            pass  # e.g. library version mismatch -> retrain
    return _train_and_save()


def explain(artifacts, row_df, top_n=6):
    """Per-feature contribution to this applicant's score (XGBoost log-odds)."""
    pipe = artifacts["explain_pipe"]
    pre = pipe.named_steps["preprocessor"]
    Xt = pre.transform(row_df)
    if hasattr(Xt, "toarray"):
        Xt = Xt.toarray()
    booster = pipe.named_steps["classifier"].get_booster()
    contribs = booster.predict(DMatrix(np.asarray(Xt, dtype=float)), pred_contribs=True)[0][:-1]
    names = [n.split("__", 1)[-1].replace("_", " ") for n in pre.get_feature_names_out()]
    order = np.argsort(-np.abs(contribs))[:top_n]
    return [(names[i], float(contribs[i])) for i in order]


# --------------------------------------------------------------------------
# Page + styling
# --------------------------------------------------------------------------
st.set_page_config(page_title="Credit Ledger", page_icon="⊕", layout="centered")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&family=Source+Serif+4:ital,wght@0,500;0,600;1,600&display=swap');

html, body, [class*="css"], .stApp { font-family: 'IBM Plex Sans', sans-serif; }
.stApp { background: radial-gradient(ellipse at top, #0f1823 0%, #0a1017 55%); }
#MainMenu, footer, header { visibility: hidden; }
.block-container { max-width: 940px; padding-top: 3rem; padding-bottom: 5rem; }

/* header */
.brand { display:flex; justify-content:space-between; align-items:flex-start; }
.brand-l { display:flex; gap:1.1rem; align-items:flex-start; }
.brand h1 { font-family:'Source Serif 4',serif; font-weight:600; font-size:2.3rem;
            color:#f2f4f6; margin:0; padding:0; line-height:1.1; }
.brand p { color:#8b98a5; margin:.5rem 0 0; max-width:22rem; line-height:1.5; font-size:1.02rem; }
.status { font-family:'IBM Plex Mono',monospace; font-size:.85rem; color:#8b98a5; padding-top:.5rem; }
.dot { display:inline-block; width:.55rem; height:.55rem; border-radius:50%;
       background:#6fae8a; margin-right:.5rem; }
.rule { border-top:1px solid #1d2b3a; margin:1.6rem 0 2rem; }

/* cards */
div[data-testid="stVerticalBlockBorderWrapper"] {
  background:#0f1923; border:1px solid #1d2b3a !important; border-radius:4px;
  padding:1.1rem 1.6rem .6rem; }
.sec { font-family:'Source Serif 4',serif; color:#d4a85f; font-size:1.25rem; margin:0 0 1.4rem; }
.lbl { color:#9aa6b2; font-size:1.02rem; }
.unit { font-family:'IBM Plex Mono',monospace; color:#6b7886; font-size:.95rem; }
.unit.r { text-align:right; }
.hr { border-top:1px solid #182431; margin:.15rem 0 .55rem; }
.hint { color:#6b7886; font-size:.92rem; margin:-.2rem 0 1.2rem; }

/* inputs */
div[data-baseweb="input"], div[data-baseweb="select"] > div, div[data-baseweb="base-input"] {
  background:#131f2b !important; border-color:#223243 !important; border-radius:3px !important; }
div[data-baseweb="input"] input { font-family:'IBM Plex Mono',monospace; text-align:right;
  color:#e6eaee !important; -webkit-text-fill-color:#e6eaee !important; }
div[data-baseweb="select"] * { color:#e6eaee !important; }
[data-testid="stNumberInputStepUp"], [data-testid="stNumberInputStepDown"] { display:none; }
div[data-testid="stSelectbox"] label, div[data-testid="stNumberInput"] label { display:none; }

/* button */
div.stButton { display:flex; justify-content:center; margin:.4rem 0 .6rem; }
div.stButton > button { background:#d9ae65; color:#1b1509; font-weight:600; font-size:1.05rem;
  border:none; border-radius:3px; padding:.75rem 2.6rem; }
div.stButton > button:hover { background:#e4bb78; color:#1b1509; }

/* result */
.res { display:flex; align-items:center; justify-content:space-between; gap:1.5rem; flex-wrap:wrap; padding:.6rem 0 1rem; }
.stamp { width:10.5rem; height:10.5rem; border-radius:50%; display:flex; align-items:center;
  justify-content:center; border:2px solid var(--c); box-shadow:inset 0 0 0 6px #0f1923, inset 0 0 0 7px var(--c);
  transform:rotate(-6deg); }
.stamp span { font-family:'Source Serif 4',serif; font-style:italic; font-weight:600;
  letter-spacing:.06em; color:var(--c); font-size:1.25rem; }
.tbl { min-width:17rem; flex:1; max-width:20rem; }
.tbl div { display:flex; justify-content:space-between; padding:.7rem 0; border-bottom:1px solid #182431; }
.tbl span:first-child { color:#9aa6b2; }
.tbl span:last-child { font-family:'IBM Plex Mono',monospace; color:#e6eaee; }
.why { margin-top:.4rem; }
.why .t { color:#9aa6b2; font-size:.95rem; margin-bottom:.7rem; }
.bar { display:flex; align-items:center; gap:.8rem; margin:.4rem 0; font-size:.92rem; color:#c9d1d9; }
.bar .n { width:13rem; flex:none; }
.bar .b { height:.5rem; border-radius:2px; }
.bar .v { font-family:'IBM Plex Mono',monospace; color:#8b98a5; font-size:.85rem; }
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def row(label, render, prefix="", suffix="", wide=False):
    """One 'label ..... input unit' line, like the mock-up."""
    if wide:
        c1, c3 = st.columns([5, 3.5], vertical_alignment="center")
        c2 = c4 = None
    else:
        c1, c2, c3, c4 = st.columns([5, 0.55, 2.6, 0.9], vertical_alignment="center")
    c1.markdown(f'<div class="lbl">{label}</div>', unsafe_allow_html=True)
    if c2 is not None:
        c2.markdown(f'<div class="unit r">{prefix}</div>', unsafe_allow_html=True)
        c4.markdown(f'<div class="unit">{suffix}</div>', unsafe_allow_html=True)
    with c3:
        value = render()
    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
    return value


def sync_ratio():
    income = st.session_state.income
    if income and income > 0:
        st.session_state.ratio = round(st.session_state.loan_amnt / income, 2)


def ring_svg(prob, threshold, color):
    r, cx = 88, 100
    circ = 2 * math.pi * r
    arc = max(prob * circ, 7)  # always show a small cap
    ang = math.radians(-90 + 360 * threshold)
    x1, y1 = cx + (r - 9) * math.cos(ang), cx + (r - 9) * math.sin(ang)
    x2, y2 = cx + (r + 9) * math.cos(ang), cx + (r + 9) * math.sin(ang)
    return f"""
<svg width="210" height="210" viewBox="0 0 200 200">
  <circle cx="100" cy="100" r="{r}" fill="none" stroke="#1e2a38" stroke-width="7"/>
  <circle cx="100" cy="100" r="{r}" fill="none" stroke="#d4a85f" stroke-width="8"
          stroke-linecap="round" stroke-dasharray="{arc:.1f} {circ:.1f}"
          transform="rotate(-90 100 100)"/>
  <line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="#7b8794" stroke-width="3"/>
  <text x="100" y="108" text-anchor="middle" fill="#f2f4f6" font-size="44"
        font-family="IBM Plex Mono, monospace" font-weight="500">{prob*100:.1f}</text>
  <text x="100" y="134" text-anchor="middle" fill="#8b98a5" font-size="22"
        font-family="IBM Plex Mono, monospace">%</text>
  <text x="100" y="158" text-anchor="middle" fill="#6b7886" font-size="11.5"
        font-family="IBM Plex Sans, sans-serif">default probability</text>
</svg>"""


def render_result(res):
    prob, thr = res["prob"], res["threshold"]
    high = prob >= thr
    color = "#c8665b" if high else "#6fae8a"
    verdict = "High Risk" if high else "Low Risk"
    html = f"""
<div class="res" style="--c:{color}">
  {ring_svg(prob, thr, color)}
  <div class="stamp"><span>{verdict.upper()}</span></div>
  <div class="tbl">
    <div><span>Default probability</span><span>{prob*100:.1f}%</span></div>
    <div><span>Decision threshold</span><span>{thr*100:.1f}%</span></div>
    <div><span>Model verdict</span><span>{verdict}</span></div>
  </div>
</div>"""
    st.markdown(html, unsafe_allow_html=True)

    if res.get("why"):
        peak = max(abs(v) for _, v in res["why"]) or 1
        bars = "".join(
            f'<div class="bar"><span class="n">{n}</span>'
            f'<span class="b" style="width:{max(abs(v)/peak*180, 3):.0f}px;'
            f'background:{"#c8665b" if v > 0 else "#6fae8a"}"></span>'
            f'<span class="v">{v:+.2f}</span></div>' for n, v in res["why"])
        st.markdown(
            '<div class="why"><div class="t">What drove this score '
            '(red raises default risk, green lowers it)</div>' + bars + '</div>',
            unsafe_allow_html=True)


# --------------------------------------------------------------------------
# Load model
# --------------------------------------------------------------------------
if not ARTIFACT_PATH.exists() and not DATA_PATH.exists():
    st.error("credit_risk_dataset.csv not found. Put it next to app.py so the model can be trained.")
    st.stop()

if not ARTIFACT_PATH.exists():
    with st.spinner("Training the model for the first time (about a minute)..."):
        artifacts = load_artifacts()
else:
    artifacts = load_artifacts()

# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------
st.markdown("""
<div class="brand">
  <div class="brand-l">
    <svg width="54" height="54" viewBox="0 0 54 54"><circle cx="27" cy="27" r="25" fill="none" stroke="#d4a85f" stroke-width="2"/>
      <circle cx="27" cy="27" r="19" fill="none" stroke="#d4a85f" stroke-width="1" opacity=".6"/>
      <path d="M27 14v26M14 27h26" stroke="#d4a85f" stroke-width="2"/></svg>
    <div><h1>Credit Ledger</h1><p>An underwriting desk for consumer loan applications</p></div>
  </div>
  <div class="status"><span class="dot"></span>service ready</div>
</div>
<div class="rule"></div>
""", unsafe_allow_html=True)

# defaults (match the mock-up)
for k, v in dict(age=30, income=600000, emp_length=5, loan_amnt=100000,
                 int_rate=11.5, ratio=0.17, hist=6).items():
    st.session_state.setdefault(k, v)

# --------------------------------------------------------------------------
# 01 - Applicant
# --------------------------------------------------------------------------
HOME = {"Renting": "RENT", "Owning": "OWN", "Mortgage": "MORTGAGE", "Other": "OTHER"}
INTENT = {"Education": "EDUCATION", "Medical": "MEDICAL", "Venture": "VENTURE",
          "Personal": "PERSONAL", "Debt consolidation": "DEBTCONSOLIDATION",
          "Home improvement": "HOMEIMPROVEMENT"}

with st.container(border=True):
    st.markdown('<div class="sec">01 — Applicant</div>', unsafe_allow_html=True)
    age = row("Age", lambda: st.number_input("Age", 18, 100, step=1, key="age"), suffix="years")
    income = row("Annual income", lambda: st.number_input(
        "Annual income", 1, 100_000_000, step=10_000, key="income", on_change=sync_ratio), prefix="₹")
    home = row("Home ownership", lambda: st.selectbox(
        "Home ownership", list(HOME), key="home"), wide=True)
    emp = row("Employment length", lambda: st.number_input(
        "Employment length", 0, 60, step=1, key="emp_length"), suffix="years")

st.write("")

# --------------------------------------------------------------------------
# 02 - Loan
# --------------------------------------------------------------------------
with st.container(border=True):
    st.markdown('<div class="sec">02 — Loan</div>', unsafe_allow_html=True)
    intent = row("Loan intent", lambda: st.selectbox(
        "Loan intent", list(INTENT), key="intent"), wide=True)
    grade = row("Loan grade", lambda: st.selectbox(
        "Loan grade", list("ABCDEFG"), key="grade"), wide=True)
    loan_amnt = row("Loan amount", lambda: st.number_input(
        "Loan amount", 1, 100_000_000, step=10_000, key="loan_amnt", on_change=sync_ratio), prefix="₹")
    int_rate = row("Interest rate", lambda: st.number_input(
        "Interest rate", 0.0, 40.0, step=0.1, format="%.1f", key="int_rate"), suffix="%")
    ratio = row("Loan as % of income", lambda: st.number_input(
        "Loan as % of income", 0.0, 5.0, step=0.01, format="%.2f", key="ratio"), suffix="ratio")
    st.markdown('<div class="hint">Calculated from income and loan amount — edit if you need to override it.</div>',
                unsafe_allow_html=True)

st.write("")

# --------------------------------------------------------------------------
# 03 - Credit bureau file
# --------------------------------------------------------------------------
with st.container(border=True):
    st.markdown('<div class="sec">03 — Credit bureau file</div>', unsafe_allow_html=True)
    prior = row("Prior default on file", lambda: st.selectbox(
        "Prior default on file", ["No", "Yes"], key="prior"), wide=True)
    hist = row("Credit history length", lambda: st.number_input(
        "Credit history length", 0, 60, step=1, key="hist"), suffix="years")

    clicked = st.button("Assess risk")

    if clicked:
        if emp > age:
            st.session_state.result = None
            st.warning("Employment length can't be greater than age.")
        else:
            applicant = pd.DataFrame([{
                "person_age": age,
                "person_income": income,
                "person_home_ownership": HOME[home],
                "person_emp_length": emp,
                "loan_intent": INTENT[intent],
                "loan_grade": grade,
                "loan_amnt": loan_amnt,
                "loan_int_rate": int_rate,
                "loan_percent_income": ratio,
                "cb_person_default_on_file": "Y" if prior == "Yes" else "N",
                "cb_person_cred_hist_length": hist,
            }])
            prob = float(artifacts["model"].predict_proba(applicant)[0, 1])
            try:
                why = explain(artifacts, applicant)
            except Exception:
                why = None
            st.session_state.result = dict(prob=prob, threshold=artifacts["threshold"], why=why)

if st.session_state.get("result"):
    st.write("")
    with st.container(border=True):
        render_result(st.session_state.result)
