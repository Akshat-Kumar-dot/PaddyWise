"""
Paddy Yield Prediction & Variety Recommendation — Streamlit front end.
Run:  streamlit run app.py
Needs paddy_model.pkl (created by Paddy_Yield_Prediction.ipynb) and paddydataset.csv in this folder.
"""
import os
import warnings

os.environ["LOKY_MAX_CPU_COUNT"] = "4"
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="PaddyWise | Yield & Variety Advisor", page_icon=":material/agriculture:", layout="wide",
                   initial_sidebar_state="collapsed")

GREEN = "#1f5f45"
LEAF = "#52b788"
MINT = "#b7e4c7"
GOLD = "#e0a030"
INK = "#14261c"
MUTED = "#64756a"
PALETTE = [GREEN, GOLD, LEAF, "#95d5b2", "#b08968", "#74c69d"]

# ---------------------------------------------------------------- styling
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Poppins:wght@600;700&display=swap');
html, body, [class*="css"], .stMarkdown, button, input, select, textarea, label { font-family: 'Inter', sans-serif; }

/* full-width canvas for 16:9 screens */
header[data-testid="stHeader"], [data-testid="stToolbar"], #MainMenu, footer { display: none !important; }
.block-container { max-width: 100% !important; padding: 1.1rem 2.4rem 2rem 2.4rem !important; }
[data-testid="stHeaderActionElements"], .stMarkdown a.anchor-link { display: none !important; }
div[data-testid="stVerticalBlock"] { gap: 0.9rem; }

/* top bar */
.topbar { display:flex; align-items:center; justify-content:space-between; gap:24px; flex-wrap:wrap;
  background: linear-gradient(100deg, #123d2c 0%, #1f5f45 60%, #2f7d5b 100%);
  border-radius: 16px; padding: 18px 28px; color:#fff; box-shadow: 0 8px 28px rgba(18,61,44,.22); }
.brand .name { font-family:'Poppins',sans-serif; font-size:1.65rem; font-weight:700; letter-spacing:-.3px; line-height:1.1; }
.brand .name em { font-style:normal; color:#f2c46d; }
.brand .sub { color:#cde8d8; font-size:.88rem; margin-top:3px; }
.kpis { display:flex; gap:10px; flex-wrap:wrap; }
.kpi { background: rgba(255,255,255,.10); border:1px solid rgba(255,255,255,.18); border-radius:12px; padding:8px 16px; min-width:118px; }
.kpi b { display:block; font-size:1.15rem; font-weight:700; color:#fff; }
.kpi span { font-size:.72rem; color:#cde8d8; text-transform:uppercase; letter-spacing:.6px; }

/* tabs */
.stTabs [data-baseweb="tab-list"] { gap: 4px; background:#fff; border:1px solid #e2eadc; border-radius:14px;
  padding:5px; width:fit-content; box-shadow: 0 2px 10px rgba(20,38,28,.04); margin-bottom: 4px; }
.stTabs [data-baseweb="tab"] { height:auto; padding: 9px 26px; border-radius:10px; background:transparent; }
.stTabs [data-baseweb="tab"] p { font-weight:600; color:#4b5e52; font-size:.92rem; }
.stTabs [data-baseweb="tab"]:hover { background:#eef5ea; }
.stTabs [aria-selected="true"] { background:#1f5f45 !important; }
.stTabs [aria-selected="true"] p { color:#fff !important; }
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display:none !important; }

div[data-testid="stVerticalBlockBorderWrapper"]:has(> div > div[data-testid="stVerticalBlock"]) { background:#fff; border-radius:16px; }

/* every chart sits in the same white card */
[data-testid="stPlotlyChart"] { background:#fff; border:1px solid #e2eadc; border-radius:16px; padding:10px 0 6px;
  overflow:hidden; box-shadow: 0 2px 10px rgba(20,38,28,.04); }

/* test-farm table */
table.tt { width:100%; border-collapse:collapse; font-size:.84rem; table-layout:fixed; border:none !important; }
table.tt th, table.tt td { border-left:none !important; border-right:none !important; border-top:none !important; background:#fff; }
table.tt th { text-align:left; color:#64756a; font-weight:600; font-size:.74rem; text-transform:uppercase; letter-spacing:.5px;
  padding:6px 6px; border-bottom:1px solid #e2eadc; }
table.tt th:first-child { width:38%; }
table.tt td { padding:8px 6px; border-bottom:1px dashed #e2eadc; color:#14261c; vertical-align:middle; }
table.tt td span { display:block; color:#64756a; font-size:.74rem; }
table.tt .n { text-align:right; }
table.tt .err { display:inline-block; font-weight:700; font-size:.78rem; padding:2px 8px; border-radius:8px; }
table.tt .err.ok { background:#d8f3dc; color:#1f5f45; } table.tt .err.warn { background:#fdebc8; color:#8a5a12; }

/* Streamlit pulls every markdown block up by 1rem; remove that so cards never overlap */
[data-testid="stMarkdownContainer"] { margin-bottom: 0 !important; }

/* panels */
.panel, [data-testid="stForm"] { background:#fff; border:1px solid #e2eadc !important; border-radius:16px !important;
  padding: 18px 20px; box-shadow: 0 2px 10px rgba(20,38,28,.04); }
.ptitle { font-family:'Poppins',sans-serif; font-size:1.02rem; font-weight:600; color:#14261c; margin:0 0 2px 0; }
.psub { color:#64756a; font-size:.82rem; margin-bottom:10px; }
.eyebrow { font-size:.7rem; font-weight:700; letter-spacing:1.2px; text-transform:uppercase; color:#b7791f; }

/* recommendation hero */
.reco { display:flex; justify-content:space-between; align-items:center; gap:24px; flex-wrap:wrap;
  background: linear-gradient(110deg,#fffaf0 0%,#ffffff 70%); border:1.5px solid #ecc274; border-radius:16px; padding:20px 24px; }
.reco .vname { font-family:'Poppins',sans-serif; font-size:2rem; font-weight:700; color:#123d2c; line-height:1.15; margin:2px 0; }
.reco .blurb { color:#64756a; font-size:.88rem; }
.reco .stats { display:flex; gap:12px; flex-wrap:wrap; }
.reco .stat { background:#f4f8f1; border-radius:12px; padding:10px 16px; min-width:120px; }
.reco .stat b { display:block; font-size:1.25rem; color:#1f5f45; }
.reco .stat span { font-size:.74rem; color:#64756a; }

/* ranking cards */
.rank-row { display:grid; grid-template-columns: repeat(3, 1fr); gap:12px; }
.vcard { background:#fff; border:1px solid #e2eadc; border-radius:14px; padding:14px 16px; }
.vcard.first { border-color:#ecc274; box-shadow: 0 4px 14px rgba(224,160,48,.15); }
.vhead { display:flex; align-items:center; justify-content:space-between; }
.vhead .t { font-weight:700; color:#14261c; }
.badge { font-size:.7rem; font-weight:700; padding:3px 9px; border-radius:20px; background:#edf2ea; color:#4b5e52; }
.first .badge { background:#e0a030; color:#fff; }
.bar { height:7px; background:#edf2ea; border-radius:6px; margin:10px 0 4px; overflow:hidden; }
.bar > div { height:100%; background: linear-gradient(90deg,#52b788,#1f5f45); border-radius:6px; }
.kv { display:flex; justify-content:space-between; font-size:.84rem; padding:3px 0; color:#2b3d31; gap:12px; }
.kv span:first-child { color:#64756a; }

/* how-it-works steps */
.how { counter-reset: s; }
.how .s { display:flex; gap:12px; padding:8px 0; border-bottom:1px dashed #e2eadc; font-size:.85rem; color:#2b3d31; }
.how .s:last-child { border-bottom:none; }
.how .s:before { counter-increment:s; content: counter(s); min-width:24px; height:24px; border-radius:50%;
  background:#1f5f45; color:#fff; font-weight:700; font-size:.75rem; display:flex; align-items:center; justify-content:center; }
.tag { display:inline-block; font-size:.68rem; font-weight:700; padding:2px 8px; border-radius:6px; margin-left:4px; vertical-align:middle; }
.tag.ml { background:#d8f3dc; color:#1f5f45; } .tag.rule { background:#fdebc8; color:#8a5a12; }
.wrow { display:flex; align-items:center; gap:10px; font-size:.82rem; margin:5px 0; }
.wrow .l { width:110px; color:#4b5e52; } .wrow .b { flex:1; height:8px; background:#edf2ea; border-radius:6px; overflow:hidden; }
.wrow .b > div { height:100%; background:#e0a030; } .wrow .v { width:34px; text-align:right; font-weight:600; }

/* input plan */
.step { display:flex; gap:12px; align-items:center; padding:7px 0; border-bottom:1px dashed #e2eadc; font-size:.84rem; }
.step:last-child { border-bottom:none; }
.step .day { min-width:84px; background:#eef5ea; color:#1f5f45; font-weight:600; border-radius:8px; text-align:center; padding:4px 0; font-size:.74rem; }
.step .what { flex:1; color:#2b3d31; } .step .amt { font-weight:600; color:#14261c; white-space:nowrap; }

.clim { display:grid; grid-template-columns: 1fr 1fr; gap:8px; }
.clim div { background:#f4f8f1; border-radius:10px; padding:8px 12px; }
.clim b { display:block; font-size:1.05rem; color:#1f5f45; } .clim span { font-size:.72rem; color:#64756a; }

div[data-testid="stMetric"] { background:#fff; border:1px solid #e2eadc; border-radius:14px; padding:12px 18px; }
.stFormSubmitButton > button { background:#1f5f45; color:#fff; border-radius:10px; border:none; padding:.6rem 1rem; font-weight:600; width:100%; }
.stFormSubmitButton > button:hover { background:#123d2c; color:#fff; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------- data & model
HERE = os.path.dirname(os.path.abspath(__file__))


@st.cache_resource
def load_bundle():
    return joblib.load(os.path.join(HERE, "paddy_model.pkl"))


@st.cache_data
def load_data():
    raw = pd.read_csv(os.path.join(HERE, "paddydataset.csv"))
    raw.columns = raw.columns.str.strip()
    for c in raw.select_dtypes("object"):
        raw[c] = raw[c].str.strip()
    raw = raw.drop_duplicates().reset_index(drop=True)
    raw = raw.rename(columns={"Soil Types": "Soil", "Paddy yield(in Kg)": "Yield"})
    raw["Yield_per_ha"] = raw["Yield"] / raw["Hectares"]
    return raw


bundle = load_bundle()
model, COLUMNS = bundle["model"], bundle["columns"]
variety_info, dose_per_ha = bundle["variety_info"], bundle["dose_per_ha"]
df = load_data()
FEATURES = ["Hectares", "Agriblock", "Variety", "Soil", "Nursery"]

LABELS = {"CO_43": "CO 43", "ponmani": "Ponmani", "delux ponni": "Delux Ponni"}
BLURB = {
    "CO_43": "Medium-duration white rice, dependable for double cropping.",
    "ponmani": "Long-duration Samba crop with bold grains, good for parboiling.",
    "delux ponni": "Fine-grain table rice that earns a market premium.",
}
PREFS = {
    "Maximum profit": "profit",
    "Maximum yield": "yield",
    "Short duration crop": "short_duration",
    "Fine-grain (premium) rice": "fine_grain",
}
CRITERIA = ["Yield", "Income", "Shorter duration", "Fine grain"]
WEIGHTS = {   # (yield, income, shortness, fine grain) — same as the notebook
    "yield": (0.7, 0.2, 0.1, 0.0),
    "profit": (0.2, 0.7, 0.1, 0.0),
    "short_duration": (0.3, 0.2, 0.5, 0.0),
    "fine_grain": (0.2, 0.3, 0.0, 0.5),
}
PLAN = [  # (column, day, description, unit)
    ("Seedrate", "Day 0", "Seed for the nursery", "kg"),
    ("Nursery_area", "Day 0", "Nursery area to prepare", "cents"),
    ("LP_Nursery", "Day 0", "Land preparation, nursery", "t"),
    ("LP_Mainfield", "Transplant", "Land preparation, main field", "t"),
    ("DAP_20days", "Day 20", "DAP fertilizer", "kg"),
    ("Weed28D_thiobencarb", "Day 28", "Thiobencarb weedicide", "units"),
    ("Urea_40Days", "Day 40", "Urea top dressing", "kg"),
    ("Potassh_50Days", "Day 50", "Potash", "kg"),
    ("Pest_60Day", "Day 60", "Pesticide spray", "ml"),
    ("Micronutrients_70Days", "Day 70", "Micronutrient mixture", "kg"),
]


def encode(rows: pd.DataFrame) -> pd.DataFrame:
    return pd.get_dummies(rows, dtype=int).reindex(columns=COLUMNS, fill_value=0)


def recommend(hectares, agriblock, soil, nursery, preference, prices):
    # Step 1 (ML): the hybrid model predicts yield for every variety under the same conditions
    farms = pd.DataFrame([{"Hectares": hectares, "Agriblock": agriblock, "Variety": v, "Soil": soil, "Nursery": nursery}
                          for v in variety_info.index])
    preds = model.predict(encode(farms))
    t = pd.DataFrame({
        "Variety": variety_info.index,
        "Yield": preds,
        "Per_ha": preds / hectares,
        "Duration": variety_info["Duration_days"].values,
        "Grain": variety_info["Grain_type"].values,
        "Price": [prices[v] for v in variety_info.index],
    })
    # Step 2: income = predicted yield x market price
    t["Income"] = t["Yield"] * t["Price"]
    # Step 3 (decision rule): scale each criterion to 0-1, weight it by the farmer's priority, add up
    norm = lambda s: (s - s.min()) / (s.max() - s.min() + 1e-9)
    w = WEIGHTS[preference]
    t["c_Yield"] = w[0] * norm(t["Yield"])
    t["c_Income"] = w[1] * norm(t["Income"])
    t["c_Shorter duration"] = w[2] * norm(-t["Duration"])
    t["c_Fine grain"] = w[3] * (t["Grain"] == "Fine")
    t["Score"] = t[[f"c_{c}" for c in CRITERIA]].sum(axis=1)
    return t.sort_values("Score", ascending=False).reset_index(drop=True)


@st.cache_data
def test_metrics():
    """Re-create the notebook's exact 80/20 split; the saved model never saw this test part."""
    X = encode(df[FEATURES])
    _, X_te, _, y_te = train_test_split(X, df["Yield"], test_size=0.2, random_state=42)
    p = model.predict(X_te)
    return {
        "r2": r2_score(y_te, p), "mae": mean_absolute_error(y_te, p),
        "rmse": np.sqrt(mean_squared_error(y_te, p)), "mape": mean_absolute_percentage_error(y_te, p) * 100,
        "actual": y_te.values, "pred": p, "test_idx": X_te.index.values,
        # 80% of test-set errors fall inside this relative band -> used as the "likely range"
        "band": np.percentile((y_te.values - p) / p, [10, 90]),
    }


def style_fig(fig, h=340, title=None):
    fig.update_layout(height=h, margin=dict(l=18, r=22, t=46 if title else 12, b=8), paper_bgcolor="#ffffff",
                      plot_bgcolor="#ffffff", font=dict(family="Inter", color=INK, size=12),
                      title=dict(text=title, font=dict(family="Poppins", size=15, color=INK), x=0.01, y=0.97,
                                 yanchor="top") if title else None,
                      legend=dict(orientation="h", title_text="", x=0, xanchor="left", y=-0.3, yanchor="top",
                                  font=dict(size=11.5)))
    fig.update_xaxes(gridcolor="#edf2ea", zeroline=False); fig.update_yaxes(gridcolor="#edf2ea", zeroline=False)
    return fig


def chart(fig, h=340, title=None, legend=None, top=None):
    fig = style_fig(fig, h, title)
    if legend:
        fig.update_layout(legend=legend)
    if top:
        fig.update_layout(margin_t=top)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


m = test_metrics()

# ---------------------------------------------------------------- top bar
st.markdown(f"""
<div class="topbar">
  <div class="brand">
    <div class="name">Paddy<em>Wise</em></div>
    <div class="sub">Hybrid, feature-optimized decision support for paddy yield prediction and variety selection</div>
  </div>
  <div class="kpis">
    <div class="kpi"><b>{m['r2']:.3f}</b><span>Test R²</span></div>
    <div class="kpi"><b>{100 - m['mape']:.1f}%</b><span>Accuracy</span></div>
    <div class="kpi"><b>{m['mae']:,.0f} kg</b><span>Avg error</span></div>
    <div class="kpi"><b>44 → 5</b><span>Features</span></div>
    <div class="kpi"><b>{len(df):,}</b><span>Farm records</span></div>
  </div>
</div>
""", unsafe_allow_html=True)

tab0, tab1, tab2, tab3, tab4 = st.tabs(["Yield Predictor", "Variety Advisor", "Data Insights", "Model Performance", "About"])

# ---------------------------------------------------------------- tab 0: live yield predictor
with tab0:
    p_in, p_out, p_side = st.columns([1.05, 2.0, 1.75], gap="medium")

    with p_in:
        with st.container(border=True):
            st.markdown('<div class="ptitle">Enter field conditions</div>'
                        '<div class="psub">The prediction updates instantly as you change any value.</div>', unsafe_allow_html=True)
            ph = st.slider("Farm area (hectares)", 1, 6, 3, 1, format="%d ha", key="p_ha",
                           help="The dataset only has whole-hectare farms (1–6 ha).")
            pblock = st.selectbox("Agri-block (location)", sorted(df["Agriblock"].unique()), key="p_block")
            pvar = st.selectbox("Paddy variety", list(variety_info.index), format_func=LABELS.get, key="p_var")
            a, b = st.columns(2)
            psoil = a.radio("Soil type", ["clay", "alluvial"], format_func=str.title, key="p_soil")
            pnur = b.radio("Nursery", ["wet", "dry"], format_func=str.title, key="p_nur")

    row = pd.DataFrame([{"Hectares": ph, "Agriblock": pblock, "Variety": pvar, "Soil": psoil, "Nursery": pnur}])
    y_hat = float(model.predict(encode(row))[0])
    lo, hi = y_hat * (1 + m["band"][0]), y_hat * (1 + m["band"][1])
    same = df[(df["Hectares"] == ph) & (df["Agriblock"] == pblock) & (df["Variety"] == pvar)
              & (df["Soil"] == psoil) & (df["Nursery"] == pnur)]

    with p_out:
        st.markdown(f"""
        <div class="reco" style="display:block">
          <div class="eyebrow">Predicted paddy yield · hybrid stacking model</div>
          <div style="display:flex; align-items:baseline; gap:14px; flex-wrap:wrap">
            <div class="vname" style="font-size:3rem">{y_hat:,.0f} kg</div>
            <div class="blurb" style="font-size:1rem">≈ {y_hat / 1000:,.2f} tonnes</div>
          </div>
          <div class="stats" style="margin-top:12px">
            <div class="stat"><b>{y_hat / ph:,.0f} kg</b><span>Yield per hectare</span></div>
            <div class="stat"><b>{lo:,.0f} – {hi:,.0f}</b><span>Likely range (kg, 80%)</span></div>
            <div class="stat"><b>{len(same)}</b><span>Real farms with these conditions</span></div>
            <div class="stat"><b>{same['Yield'].mean():,.0f} kg</b><span>Their actual average</span></div>
          </div>
        </div>""".replace("nan kg", "–"), unsafe_allow_html=True)

        # where does this prediction sit among real farms of the same size?
        peers = df[df["Hectares"] == ph] if (df["Hectares"] == ph).any() else df
        fig = px.histogram(peers, x="Yield", nbins=30, color_discrete_sequence=[MINT])
        fig.add_vline(x=y_hat, line=dict(color=GREEN, width=3))
        fig.add_annotation(x=y_hat, y=1, yref="paper", text=f"Prediction {y_hat:,.0f} kg", showarrow=False,
                           xanchor="left", xshift=6, font=dict(color=GREEN, size=12))
        fig.update_xaxes(title="Actual yield of real farms (kg)"); fig.update_yaxes(title="Number of farms")
        chart(fig, 330, f"Prediction vs real {ph:g}-hectare farms in the dataset")

    with p_side:
        with st.container(border=True):
            st.markdown('<div class="ptitle">Live check on unseen farms</div>'
                        '<div class="psub">Random farms from the 20% test set. The model never saw them while training.</div>',
                        unsafe_allow_html=True)
            if "seed" not in st.session_state:
                st.session_state.seed = 7
            if st.button("Pick 6 new test farms", use_container_width=True):
                st.session_state.seed += 1
            rng = np.random.default_rng(st.session_state.seed)
            pick = rng.choice(len(m["test_idx"]), 6, replace=False)
            sample = df.loc[m["test_idx"][pick], FEATURES].copy()
            sample["Actual (kg)"] = m["actual"][pick]
            sample["Predicted (kg)"] = m["pred"][pick]
            sample["Error"] = (sample["Predicted (kg)"] - sample["Actual (kg)"]) / sample["Actual (kg)"] * 100
            rows = "".join(
                f'<tr><td><b>{r.Agriblock}</b><span>{r.Hectares} ha · {LABELS[r.Variety]}</span></td>'
                f'<td class="n">{r._6:,.0f}</td><td class="n">{r._7:,.0f}</td>'
                f'<td class="n"><span class="err {"ok" if abs(r.Error) < 4 else "warn"}">{r.Error:+.1f}%</span></td></tr>'
                for r in sample.itertuples())
            st.markdown(f'<table class="tt"><thead><tr><th>Test farm</th><th class="n">Actual kg</th>'
                        f'<th class="n">Predicted kg</th><th class="n">Error</th></tr></thead><tbody>{rows}</tbody></table>',
                        unsafe_allow_html=True)
            avg_err = sample["Error"].abs().mean()
            st.markdown(f'<div class="kv"><span>Average error on these 6 farms</span><span><b>{avg_err:.1f}%</b></span></div>'
                        f'<div class="kv"><span>Average error on all 468 test farms</span><span><b>{m["mape"]:.1f}%</b></span></div>',
                        unsafe_allow_html=True)

# ---------------------------------------------------------------- tab 1: advisor
with tab1:
    c_in, c_main, c_side = st.columns([1.05, 2.5, 1.25], gap="medium")

    with c_in:
        with st.form("farm", border=False):
            st.markdown('<div class="ptitle">Farm details</div><div class="psub">Describe the field and your priority.</div>',
                        unsafe_allow_html=True)
            hectares = st.slider("Farm area (hectares)", 1, 6, 4, 1, format="%d ha",
                                 help="The dataset only has whole-hectare farms (1–6 ha).")
            agriblock = st.selectbox("Agri-block (location)", sorted(df["Agriblock"].unique()))
            a, b = st.columns(2)
            soil = a.radio("Soil type", ["clay", "alluvial"], format_func=str.title)
            nursery = b.radio("Nursery", ["wet", "dry"], format_func=str.title)
            pref_label = st.selectbox("Your priority", list(PREFS))
            with st.expander("Market price (₹ per kg paddy)"):
                prices = {v: st.number_input(LABELS[v], 5.0, 100.0, float(variety_info.loc[v, "Price_Rs_per_kg"]), 0.5)
                          for v in variety_info.index}
            st.form_submit_button("Get recommendation")

    pref = PREFS[pref_label]
    table = recommend(hectares, agriblock, soil, nursery, pref, prices)
    best = table.iloc[0]

    with c_main:
        st.markdown(f"""
        <div class="reco">
          <div>
            <div class="eyebrow">Recommended variety · {pref_label}</div>
            <div class="vname">{LABELS[best.Variety]}</div>
            <div class="blurb">{BLURB[best.Variety]}</div>
          </div>
          <div class="stats">
            <div class="stat"><b>{best.Yield:,.0f} kg</b><span>Expected yield</span></div>
            <div class="stat"><b>{best.Per_ha:,.0f} kg</b><span>Per hectare</span></div>
            <div class="stat"><b>₹{best.Income:,.0f}</b><span>Estimated income</span></div>
            <div class="stat"><b>{best.Duration} days</b><span>Crop duration</span></div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        cards = "".join(f"""
          <div class="vcard {'first' if i == 0 else ''}">
            <div class="vhead"><span class="t">{LABELS[r.Variety]}</span><span class="badge">Rank {i + 1}</span></div>
            <div class="bar"><div style="width:{max(r.Score, 0.02) * 100:.0f}%"></div></div>
            <div class="kv" style="margin-bottom:6px"><span>Match score</span><span><b>{r.Score * 100:.0f}</b> / 100</span></div>
            <div class="kv"><span>Predicted yield</span><span>{r.Yield:,.0f} kg</span></div>
            <div class="kv"><span>Per hectare</span><span>{r.Per_ha:,.0f} kg</span></div>
            <div class="kv"><span>Income</span><span>₹{r.Income:,.0f}</span></div>
            <div class="kv"><span>Duration</span><span>{r.Duration} days</span></div>
            <div class="kv"><span>Grain</span><span>{r.Grain}</span></div>
          </div>""" for i, r in enumerate(table.itertuples()))
        st.markdown(f'<div class="rank-row">{cards}</div>', unsafe_allow_html=True)

        g1, g2 = st.columns(2, gap="medium")
        with g1:
            # score breakdown: how much each criterion added to each variety's score
            names = table["Variety"].map(LABELS)
            fig = go.Figure()
            for crit, col in zip(CRITERIA, [GREEN, GOLD, LEAF, "#b08968"]):
                fig.add_bar(y=names, x=table[f"c_{crit}"] * 100, name=crit, orientation="h", marker_color=col)
            fig.add_scatter(y=names, x=table["Score"] * 100 + 2, mode="text", text=(table["Score"] * 100).round().astype(int),
                            textposition="middle right", textfont=dict(size=13, color=INK), showlegend=False, hoverinfo="skip")
            fig.update_layout(barmode="stack", bargap=0.35, xaxis=dict(title="Match score (0–100)", range=[0, 108]),
                              yaxis=dict(autorange="reversed"))
            chart(fig, 340, "Why this variety: score breakdown", top=84,
                  legend=dict(orientation="h", x=0, xanchor="left", y=1.02, yanchor="bottom"))
        with g2:
            sizes = range(1, 7)
            curve = pd.DataFrame([{"Hectares": h, "Agriblock": agriblock, "Variety": v, "Soil": soil, "Nursery": nursery}
                                  for v in variety_info.index for h in sizes])
            curve["Yield"] = model.predict(encode(curve))
            curve["Variety"] = curve["Variety"].map(LABELS)
            fig = px.line(curve, x="Hectares", y="Yield", color="Variety", markers=True,
                          color_discrete_map={"CO 43": LEAF, "Ponmani": "#b08968", "Delux Ponni": GOLD})
            fig.add_scatter(x=[hectares], y=[best.Yield], mode="markers", name="Your farm",
                            marker=dict(size=14, color=GREEN, line=dict(color="#fff", width=2)))
            fig.update_yaxes(title="Predicted yield (kg)")
            fig.update_xaxes(title="Farm area (ha)")
            chart(fig, 340, "Predicted yield vs farm size",
                  legend=dict(orientation="v", x=0.02, xanchor="left", y=0.98, yanchor="top",
                              bgcolor="rgba(255,255,255,.85)", bordercolor="#e2eadc", borderwidth=1))

    with c_side:
        wts = WEIGHTS[pref]
        wrows = "".join(f'<div class="wrow"><span class="l">{c}</span><span class="b"><div style="width:{w * 100:.0f}%"></div></span>'
                        f'<span class="v">{w * 100:.0f}%</span></div>' for c, w in zip(CRITERIA, wts))
        st.markdown(f"""
        <div class="panel">
          <div class="ptitle">How the recommendation is made</div>
          <div class="how">
            <div class="s"><div>The hybrid ML model predicts the yield of <b>each variety</b> for your farm conditions.<span class="tag ml">ML</span></div></div>
            <div class="s"><div>Income = predicted yield × market price.<span class="tag rule">Formula</span></div></div>
            <div class="s"><div>Each criterion is scaled to 0–1 across the three varieties, multiplied by the weight for your priority, and added up.<span class="tag rule">Decision rule</span></div></div>
            <div class="s"><div>The variety with the highest total score is recommended.</div></div>
          </div>
          <div class="psub" style="margin:10px 0 4px">Weights for “{pref_label}”</div>
          {wrows}
        </div>
        """, unsafe_allow_html=True)

        w = df[df["Agriblock"] == agriblock].iloc[0]
        rain = sum(w[c] for c in df.columns if "Rain" in c)
        tmax = np.mean([w[c] for c in df.columns if c.startswith("Max temp")])
        tmin = np.mean([w[c] for c in df.columns if c.startswith("Min temp")])
        rh = np.mean([w[c] for c in df.columns if c.startswith("Relative Humidity")])
        st.markdown(f"""
        <div class="panel">
          <div class="ptitle">{agriblock} season climate</div>
          <div class="psub">Recorded over the 120-day crop season.</div>
          <div class="clim">
            <div><b>{rain:.0f} mm</b><span>Total rainfall</span></div>
            <div><b>{rh:.0f}%</b><span>Avg humidity</span></div>
            <div><b>{tmax:.1f} °C</b><span>Avg max temp</span></div>
            <div><b>{tmin:.1f} °C</b><span>Avg min temp</span></div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    steps = [f'<div class="step"><div class="day">{day}</div><div class="what">{what}</div>'
             f'<div class="amt">{dose_per_ha[col] * hectares:,.1f} {unit}</div></div>'
             for col, day, what, unit in PLAN]
    half = len(steps) // 2
    st.markdown(f"""
    <div class="panel">
      <div class="ptitle">Input plan for {hectares:g} ha</div>
      <div class="psub">Standard practice recovered from the dataset (per-hectare dose × your farm area).</div>
      <div style="display:grid; grid-template-columns: 1fr 1fr; column-gap:36px">
        <div>{''.join(steps[:half])}</div><div>{''.join(steps[half:])}</div>
      </div>
    </div>
    """, unsafe_allow_html=True)
    st.caption("Duration, grain type and prices are general variety information (prices are indicative and editable). "
               "Yields are predicted by the hybrid ML model trained on the UCI Paddy Dataset.")

# ---------------------------------------------------------------- tab 2: insights
with tab2:
    k = st.columns(4)
    k[0].metric("Records after cleaning", f"{len(df):,}", "451 duplicates removed", delta_color="off")
    k[1].metric("Avg yield per hectare", f"{df['Yield_per_ha'].mean():,.0f} kg")
    k[2].metric("Agri-blocks", df["Agriblock"].nunique())
    k[3].metric("Varieties", df["Variety"].nunique())

    c1, c2, c3 = st.columns(3, gap="medium")
    with c1:
        chart(px.box(df, x="Hectares", y="Yield", color_discrete_sequence=[LEAF]), 360, "Yield grows with farm area")
    with c2:
        imp = pd.DataFrame({"Factor": ["Farm area", "Agri-block (weather)", "Variety", "Nursery", "Soil"],
                            "Importance": [77.9, 8.0, 7.0, 3.6, 3.5]})
        fig = px.bar(imp.iloc[::-1], x="Importance", y="Factor", orientation="h", text="Importance",
                     color="Importance", color_continuous_scale=[MINT, GREEN])
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside", cliponaxis=False).update_coloraxes(showscale=False)
        fig.update_xaxes(range=[0, 92])
        chart(fig, 360, "Key factors for yield per hectare (%)")
    with c3:
        fig = px.histogram(df, x="Yield_per_ha", nbins=35, color_discrete_sequence=[GOLD])
        fig.update_xaxes(title="kg / ha")
        chart(fig, 360, "Distribution of yield per hectare")

    c4, c5 = st.columns([1.6, 1], gap="medium")
    with c4:
        dim = st.radio("Average yield per hectare by", ["Variety", "Agriblock", "Soil", "Nursery"], horizontal=True)
        g = df.groupby(dim)["Yield_per_ha"].mean().sort_values(ascending=False).reset_index()
        fig = px.bar(g, x=dim, y="Yield_per_ha", text="Yield_per_ha", color=dim, color_discrete_sequence=PALETTE)
        fig.update_traces(texttemplate="%{text:,.0f}", textposition="outside", showlegend=False, cliponaxis=False)
        fig.update_yaxes(range=[g["Yield_per_ha"].min() - 120, g["Yield_per_ha"].max() + 60], title="kg / ha")
        chart(fig, 330)
    with c5:
        st.markdown("""
        <div class="panel"><div class="ptitle">How the features were optimized</div>
        <div class="psub">44 raw columns → 5 model features, with no loss of accuracy.</div>
        <div class="kv"><span>Trash (straw bundles)</span><span>Removed: known only after harvest</span></div>
        <div class="kv"><span>10 input columns</span><span>Removed: each = hectares × fixed dose</span></div>
        <div class="kv"><span>28 weather columns</span><span>Removed: fixed by the agri-block</span></div>
        <div class="kv" style="border-top:1px solid #e2eadc; margin-top:8px; padding-top:8px"><span><b>Final features</b></span>
        <span><b>Hectares, Agriblock, Variety, Soil, Nursery</b></span></div>
        </div>""", unsafe_allow_html=True)

# ---------------------------------------------------------------- tab 3: model performance
with tab3:
    k = st.columns(4)
    k[0].metric("R² score", f"{m['r2']:.4f}")
    k[1].metric("MAE", f"{m['mae']:,.0f} kg")
    k[2].metric("RMSE", f"{m['rmse']:,.0f} kg")
    k[3].metric("MAPE", f"{m['mape']:.2f}%")
    st.caption("Measured live on the 20% hidden test set (468 farms) that the model never saw during training.")

    comp = pd.DataFrame({
        "Model": ["Hybrid Stacking", "Gradient Boosting", "Decision Tree", "Random Forest", "Linear Regression"],
        "Test R²": [0.9903, 0.9902, 0.9893, 0.9891, 0.9875],
        "MAE (kg)": [658, 665, 687, 694, 770],
        "Accuracy (%)": [97.11, 97.07, 96.97, 96.93, 96.34],
        "CV Mean R²": [0.9916, 0.9915, 0.9904, 0.9901, 0.9887],
    })
    c1, c2, c3 = st.columns([1.2, 1.2, 1], gap="medium")
    with c1:
        fig = px.scatter(x=m["actual"], y=m["pred"], opacity=0.55, color_discrete_sequence=[LEAF],
                         labels={"x": "Actual yield (kg)", "y": "Predicted yield (kg)"})
        lo, hi = m["actual"].min(), m["actual"].max()
        fig.add_scatter(x=[lo, hi], y=[lo, hi], mode="lines", line=dict(color=GOLD, dash="dash", width=2), name="Perfect prediction")
        chart(fig, 380, "Actual vs predicted (test set)",
              legend=dict(x=0.02, xanchor="left", y=0.98, yanchor="top", bgcolor="rgba(255,255,255,.85)"))
    with c2:
        fig = px.bar(comp.iloc[::-1], x="MAE (kg)", y="Model", orientation="h", text="MAE (kg)",
                     color="Model", color_discrete_sequence=[MINT] * 4 + [GREEN])
        fig.update_traces(textposition="outside", showlegend=False, cliponaxis=False)
        fig.update_xaxes(range=[0, 880])
        chart(fig, 380, "Average error by model (lower is better)")
    with c3:
        st.markdown("""
        <div class="panel"><div class="ptitle">Hybrid stacking architecture</div>
        <div class="psub">Three base models learn the data; a final linear model learns how much to trust each one.</div>
        <div class="kv"><span>Base model 1</span><span>Gradient Boosting, depth 3</span></div>
        <div class="kv"><span>Base model 2</span><span>Gradient Boosting, depth 2</span></div>
        <div class="kv"><span>Base model 3</span><span>Random Forest</span></div>
        <div class="kv"><span>Meta model</span><span>Linear Regression (weights ≥ 0)</span></div>
        <div class="kv"><span>Validation</span><span>80/20 split + 5-fold CV</span></div>
        </div>""", unsafe_allow_html=True)
    st.dataframe(comp, hide_index=True, use_container_width=True)

# ---------------------------------------------------------------- tab 4: about
with tab4:
    a, b, c = st.columns([1.4, 1, 1], gap="medium")
    with a:
        st.markdown("""
        <div class="panel"><div class="ptitle">Problem statement</div>
        <div style="font-size:.9rem; color:#2b3d31; line-height:1.6">Develop a hybrid, feature-optimized and decision-support
        machine learning framework that identifies the key factors influencing paddy growth, predicts rice yield, and
        recommends the most suitable paddy variety based on agricultural conditions and farmers' preferences.</div></div>
        <div class="panel"><div class="ptitle">Dataset</div>
        <div style="font-size:.9rem">M. Subramaniyan, "Paddy Dataset," UCI Machine Learning Repository, 2023.
        <a href="https://doi.org/10.24432/C55W3J" target="_blank">doi.org/10.24432/C55W3J</a></div></div>
        """, unsafe_allow_html=True)
    with b:
        st.markdown("""
        <div class="panel"><div class="ptitle">Pipeline</div>
        <div class="kv"><span>1. Data cleaning</span><span>names, nulls, duplicates</span></div>
        <div class="kv"><span>2. EDA</span><span>graphs, correlation</span></div>
        <div class="kv"><span>3. Feature optimization</span><span>44 → 5</span></div>
        <div class="kv"><span>4. Modelling</span><span>5 models, hybrid stacking</span></div>
        <div class="kv"><span>5. Evaluation</span><span>R², MAE, RMSE, MAPE, CV</span></div>
        <div class="kv"><span>6. Decision support</span><span>weighted variety ranking</span></div>
        </div>""", unsafe_allow_html=True)
    with c:
        st.markdown("""
        <div class="panel"><div class="ptitle">Group 2 · B1_Sec_A_Sem_V</div>
        <div class="kv"><span>2427010043</span><span><b>Akshat Kumar</b></span></div>
        <div class="kv"><span>2427010049</span><span><b>Kushal Bajaj</b></span></div>
        <div class="kv"><span>2427010336</span><span><b>Chirag Sahu</b></span></div>
        </div>
        <div class="panel"><div class="ptitle">Tech stack</div>
        <div class="kv"><span>ML</span><span>scikit-learn</span></div>
        <div class="kv"><span>Data</span><span>pandas, NumPy</span></div>
        <div class="kv"><span>Charts</span><span>Plotly, Matplotlib, Seaborn</span></div>
        <div class="kv"><span>Web app</span><span>Streamlit</span></div>
        </div>""", unsafe_allow_html=True)
