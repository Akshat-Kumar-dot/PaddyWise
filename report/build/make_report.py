"""Builds report/Project_Report.html (then printed to PDF with headless Chrome)."""
import base64
import datetime as dt
import os
import warnings

os.environ["LOKY_MAX_CPU_COUNT"] = "4"
warnings.filterwarnings("ignore")

import joblib
import nbformat
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.join(ROOT, "report", "build", "Project_Report.html")

# ------------------------------------------------------------------ data, model, numbers
raw = pd.read_csv(os.path.join(ROOT, "paddydataset.csv"))
N_RAW, C_RAW = raw.shape
raw.columns = raw.columns.str.strip()
for c in raw.select_dtypes("object"):
    raw[c] = raw[c].str.strip()
N_DUP = int(raw.duplicated().sum())
df = raw.drop_duplicates().reset_index(drop=True).rename(columns={"Soil Types": "Soil", "Paddy yield(in Kg)": "Yield"})
df["Yield_per_ha"] = df["Yield"] / df["Hectares"]

bundle = joblib.load(os.path.join(ROOT, "paddy_model.pkl"))
model, COLS, vinfo, dose = bundle["model"], bundle["columns"], bundle["variety_info"], bundle["dose_per_ha"]
FEATURES = ["Hectares", "Agriblock", "Variety", "Soil", "Nursery"]
enc = lambda d: pd.get_dummies(d, dtype=int).reindex(columns=COLS, fill_value=0)

X = enc(df[FEATURES])
X_tr, X_te, y_tr, y_te = train_test_split(X, df["Yield"], test_size=0.2, random_state=42)
p_te = model.predict(X_te)
R2 = 1 - ((y_te - p_te) ** 2).sum() / ((y_te - y_te.mean()) ** 2).sum()
MAE = np.abs(y_te - p_te).mean()
RMSE = np.sqrt(((y_te - p_te) ** 2).mean())
MAPE = (np.abs(y_te - p_te) / y_te).mean() * 100
BAND = np.percentile((y_te.values - p_te) / p_te, [10, 90]) * 100
meta = model.final_estimator_

LABELS = {"CO_43": "CO 43", "ponmani": "Ponmani", "delux ponni": "Delux Ponni"}
WEIGHTS = {"Maximum yield": (0.7, 0.2, 0.1, 0.0), "Maximum profit": (0.2, 0.7, 0.1, 0.0),
           "Short duration": (0.3, 0.2, 0.5, 0.0), "Fine grain": (0.2, 0.3, 0.0, 0.5)}


def recommend(ha, block, soil, nursery, pref):
    farms = pd.DataFrame([{"Hectares": ha, "Agriblock": block, "Variety": v, "Soil": soil, "Nursery": nursery}
                          for v in vinfo.index])
    pred = model.predict(enc(farms))
    t = pd.DataFrame({"Variety": vinfo.index, "Yield": pred, "Duration": vinfo["Duration_days"].values,
                      "Grain": vinfo["Grain_type"].values, "Price": vinfo["Price_Rs_per_kg"].values})
    t["Income"] = t["Yield"] * t["Price"]
    norm = lambda s: (s - s.min()) / (s.max() - s.min() + 1e-9)
    w = WEIGHTS[pref]
    t["s_y"], t["s_i"] = w[0] * norm(t["Yield"]), w[1] * norm(t["Income"])
    t["s_d"], t["s_g"] = w[2] * norm(-t["Duration"]), w[3] * (t["Grain"] == "Fine")
    t["Score"] = t[["s_y", "s_i", "s_d", "s_g"]].sum(axis=1)
    return t.sort_values("Score", ascending=False).reset_index(drop=True)


# climate profile per block
clim = []
for b, g in df.groupby("Agriblock"):
    r = g.iloc[0]
    clim.append({"Block": b,
                 "Rain": sum(r[c] for c in df.columns if "Rain" in c),
                 "Tmax": np.mean([r[c] for c in df.columns if c.startswith("Max temp")]),
                 "Tmin": np.mean([r[c] for c in df.columns if c.startswith("Min temp")]),
                 "RH": np.mean([r[c] for c in df.columns if c.startswith("Relative Humidity")]),
                 "YpH": g["Yield_per_ha"].mean(), "N": len(g)})
clim = pd.DataFrame(clim).sort_values("YpH", ascending=False)


# ------------------------------------------------------------------ figures from executed notebooks
def nb_images(path):
    nb = nbformat.read(os.path.join(ROOT, path), 4)
    imgs = {}
    for i, c in enumerate(nb.cells):
        for o in c.get("outputs", []):
            if "image/png" in o.get("data", {}):
                imgs[i] = o["data"]["image/png"].replace("\n", "")
                break
    return imgs


A = nb_images("Paddy_Yield_Prediction.ipynb")
B = nb_images("Hybrid_Model_From_Scratch.ipynb")


def fig(b64, caption, num, width="100%"):
    return (f'<figure><img src="data:image/png;base64,{b64}" style="width:{width}">'
            f'<figcaption><b>Figure {num}.</b> {caption}</figcaption></figure>')


# ------------------------------------------------------------------ SVG flowchart helpers
C_DARK, C_MID, C_LIGHT, C_GOLD = "#1f3a5f", "#2e5c8a", "#eaf1f8", "#111111"


def box(x, y, w, h, title, sub="", fill=C_LIGHT, stroke=C_DARK, tc="#111111", rx=10):
    s = f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="1.6"/>'
    if sub:
        s += (f'<text x="{x + w / 2}" y="{y + h / 2 - 5}" text-anchor="middle" font-size="13.5" font-weight="700" fill="{tc}">{title}</text>'
              f'<text x="{x + w / 2}" y="{y + h / 2 + 13}" text-anchor="middle" font-size="11" fill="{tc}" opacity=".85">{sub}</text>')
    else:
        s += f'<text x="{x + w / 2}" y="{y + h / 2 + 5}" text-anchor="middle" font-size="13.5" font-weight="700" fill="{tc}">{title}</text>'
    return s


def arrow(x1, y1, x2, y2, label=""):
    s = f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="#4a4f57" stroke-width="1.8" marker-end="url(#ah)"/>'
    if label:
        s += f'<text x="{(x1 + x2) / 2 + 6}" y="{(y1 + y2) / 2 + 4}" font-size="11" fill="#4a4f57">{label}</text>'
    return s


def svg(w, h, body):
    return (f'<svg viewBox="0 0 {w} {h}" width="100%" xmlns="http://www.w3.org/2000/svg" font-family="Segoe UI, Arial">'
            '<defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto">'
            '<path d="M0,0 L10,5 L0,10 z" fill="#4a4f57"/></marker></defs>' + body + '</svg>')


def vflow(steps, w=560, bw=420, bh=52, gap=24, highlight=()):
    x = (w - bw) / 2
    body, y = "", 6
    for i, (t, s) in enumerate(steps):
        hl = i in highlight
        body += box(x, y, bw, bh, t, s, fill=C_DARK if hl else C_LIGHT, tc="#fff" if hl else "#111111")
        if i < len(steps) - 1:
            body += arrow(w / 2, y + bh, w / 2, y + bh + gap - 2)
        y += bh + gap
    return svg(w, y - gap + 12, body)


# 1. overall pipeline
FLOW_MAIN = vflow([
    ("Raw dataset", f"UCI Paddy Dataset · {N_RAW:,} rows × {C_RAW} columns"),
    ("Data preprocessing", f"clean names & text · check nulls · remove {N_DUP} duplicates · outlier check"),
    ("Exploratory Data Analysis", "distributions · group averages · correlation heatmap"),
    ("Feature optimization", "remove leakage + redundant + dependent columns: 44 → 5 features"),
    ("Encoding & train/test split", "one-hot encoding · 80% train (1,870) / 20% test (468)"),
    ("Model training", "Linear Reg. · Decision Tree · Random Forest · Gradient Boosting · Hybrid Stacking"),
    ("Testing & comparison", "R², MAE, RMSE, MAPE on hidden test set · 5-fold cross-validation"),
    ("Key-factor identification", "permutation importance · per-hectare importance"),
    ("Decision support", "yield prediction · variety recommendation · day-wise input plan"),
    ("Deployment", "Streamlit web app (PaddyWise) using the saved model"),
], w=620, bw=520, highlight=(5, 8))

# 2. preprocessing
FLOW_PRE = vflow([
    ("Load CSV", f"{N_RAW:,} rows, {C_RAW} columns"),
    ("Clean column names", "strip spaces · remove units · short names (e.g. 'Paddy yield(in Kg)' → Yield)"),
    ("Clean text values", "strip spaces inside Agriblock, Variety, Soil, Nursery …"),
    ("Missing-value check", "0 missing values → nothing to fill"),
    ("Duplicate removal", f"{N_DUP} exact duplicate rows removed → {len(df):,} rows"),
    ("Outlier check", "boxplots of yield and yield/ha → realistic range, no removal"),
], w=560, bw=470)

# 3. feature funnel
FUNNEL = svg(640, 250, "".join([
    box(20, 10, 600, 44, "44 input columns in the raw data", "", fill=C_LIGHT),
    arrow(320, 54, 320, 72, ""), f'<text x="335" y="68" font-size="11" fill="#333333">− Trash (post-harvest → data leakage)</text>',
    box(70, 74, 500, 44, "43 columns", "", fill=C_LIGHT),
    arrow(320, 118, 320, 136, ""), f'<text x="335" y="132" font-size="11" fill="#333333">− 10 input doses (each = Hectares × constant, r = 1.0)</text>',
    box(130, 138, 380, 44, "33 columns", "", fill=C_LIGHT),
    arrow(320, 182, 320, 200, ""), f'<text x="335" y="196" font-size="11" fill="#333333">− 28 weather columns (fully fixed by Agriblock)</text>',
    box(190, 202, 260, 44, "5 features", "", fill=C_DARK, tc="#fff"),
]))

# 4. stacking architecture
STACK = svg(700, 300, "".join([
    box(10, 120, 120, 60, "Farm data", "5 features (14 cols)"),
    box(200, 20, 240, 60, "Gradient Boosting", "300 trees · depth 3 · η 0.05"),
    box(200, 120, 240, 60, "Gradient Boosting", "400 trees · depth 2 · η 0.03"),
    box(200, 220, 240, 60, "Random Forest", "300 trees · min leaf 3"),
    arrow(130, 150, 198, 50), arrow(130, 150, 198, 150), arrow(130, 150, 198, 250),
    box(500, 105, 190, 90, "Meta model", "Linear Regression (w ≥ 0)", fill=C_DARK, tc="#fff"),
    arrow(440, 50, 498, 135, "p1"), arrow(440, 150, 498, 150, "p2"), arrow(440, 250, 498, 165, "p3"),
    f'<text x="595" y="225" text-anchor="middle" font-size="12" fill="#111111" font-weight="700">final = b + w1·p1 + w2·p2 + w3·p3</text>',
    f'<text x="595" y="245" text-anchor="middle" font-size="10.5" fill="#4a4f57">weights learned from out-of-fold predictions</text>',
]))

# 5. recommendation engine
RECO = svg(700, 520, "".join([
    box(170, 6, 360, 54, "Farmer input", "area · agri-block · soil · nursery · priority · prices"),
    arrow(350, 60, 350, 84),
    box(170, 86, 360, 54, "Create 3 'what-if' farms", "same conditions, one per variety"),
    arrow(350, 140, 350, 164),
    box(170, 166, 360, 54, "Hybrid ML model predicts yield", "for CO 43, Ponmani, Delux Ponni", fill=C_DARK, tc="#fff"),
    arrow(350, 220, 350, 244),
    box(170, 246, 360, 54, "Compute criteria", "yield · income = yield × price · duration · grain type"),
    arrow(350, 300, 350, 324),
    box(170, 326, 360, 54, "Normalise each criterion to 0–1", "min–max across the three varieties"),
    arrow(350, 380, 350, 404),
    box(170, 406, 360, 54, "Weighted score (priority weights)", "Score = Σ weight × normalised criterion"),
    arrow(350, 460, 350, 476),
    box(120, 478, 460, 38, "Rank varieties → recommend #1 + input plan + climate profile", "", fill="#e6ecf3", stroke=C_GOLD),
]))

# 6. advice generation
ADVICE = vflow([
    ("Dataset input columns", "seed, land prep, DAP, weedicide, urea, potash, pesticide, micronutrients"),
    ("Divide by Hectares", "every farm used the same dose per hectare (median taken)"),
    ("Standard dose per hectare", "e.g. Seed 25 kg/ha · DAP 40 kg/ha · Urea 27.13 kg/ha"),
    ("Multiply by the farmer's area", "dose × hectares → exact amounts for this farm"),
    ("Attach the timing from the column names", "DAP_20days → Day 20 · Urea_40Days → Day 40 …"),
    ("Day-wise plan + calendar dates", "optional: add a start date → actual dates"),
], w=560, bw=470)

# 7. testing flow
TESTFLOW = vflow([
    ("Cleaned & encoded data", f"{len(df):,} rows × 14 columns"),
    ("80 / 20 random split (seed 42)", "1,870 training rows · 468 hidden test rows"),
    ("Fit each model on training rows only", "test rows are never seen during training"),
    ("Predict the 468 test farms", "compare prediction with actual yield"),
    ("Compute R², MAE, RMSE, MAPE", "rank the models"),
    ("5-fold cross-validation", "repeat train/test 5 times on different splits → mean ± std"),
    ("Extra checks", "feature-set comparison · residuals · from-scratch vs scikit-learn · live app check"),
], w=560, bw=470)

# day-wise timeline
PLAN = [("Seedrate", 0, "Seed for nursery", "kg"), ("Nursery_area", 0, "Nursery area to prepare", "cents"),
        ("LP_Nursery", 0, "Land preparation – nursery", "t"), ("LP_Mainfield", None, "Land preparation – main field (before transplanting)", "t"),
        ("DAP_20days", 20, "DAP fertilizer (basal phosphorus)", "kg"), ("Weed28D_thiobencarb", 28, "Thiobencarb weedicide", "units"),
        ("Urea_40Days", 40, "Urea top-dressing (nitrogen)", "kg"), ("Potassh_50Days", 50, "Muriate of potash (potassium)", "kg"),
        ("Pest_60Day", 60, "Pesticide spray", "ml"), ("Micronutrients_70Days", 70, "Micronutrient mixture", "kg")]
# (day, label, label row) — rows keep neighbouring labels from colliding
marks = [(0, "Sowing / nursery", 0), (20, "DAP", 1), (28, "Weedicide", 0), (40, "Urea", 1), (50, "Potash", 0),
         (60, "Pesticide", 1), (70, "Micronutrients", 0)]
harvests = [(135, "CO 43"), (140, "Delux Ponni"), (155, "Ponmani")]
tl = ""
X0, X1 = 60, 650
LINE_Y = 80
sx = lambda d: X0 + (X1 - X0) * d / 160
tl += f'<line x1="{X0}" y1="{LINE_Y}" x2="{X1}" y2="{LINE_Y}" stroke="{C_DARK}" stroke-width="4"/>'
for d, lab, row in marks:
    ty = 30 if row == 0 else 54
    tl += f'<line x1="{sx(d)}" y1="{ty + 4}" x2="{sx(d)}" y2="{LINE_Y}" stroke="#b8c2cd" stroke-dasharray="2,2"/>'
    tl += f'<circle cx="{sx(d)}" cy="{LINE_Y}" r="6" fill="{C_DARK}"/>'
    tl += f'<text x="{sx(d)}" y="{ty - 10}" text-anchor="middle" font-size="9.5" fill="#4a4f57">Day {d}</text>'
    tl += f'<text x="{sx(d)}" y="{ty + 2}" text-anchor="middle" font-size="11" font-weight="700" fill="{C_DARK}">{lab}</text>'
tl += f'<text x="{sx(145)}" y="20" text-anchor="middle" font-size="11" font-weight="700" fill="{C_GOLD}">Expected harvest</text>'
for i, (d, lab) in enumerate(harvests):
    ty = 36 + i * 13
    tl += f'<circle cx="{sx(d)}" cy="{LINE_Y}" r="6" fill="{C_GOLD}"/>'
    tl += f'<text x="{sx(145)}" y="{ty}" text-anchor="middle" font-size="10" fill="#333333">Day {d}: {lab}</text>'
for lo, hi, lab in [(0, 30, "Days 1–30"), (30, 60, "31–60"), (60, 90, "61–90"), (90, 120, "91–120")]:
    stage = {"Days 1–30": "establishment", "31–60": "tillering", "61–90": "panicle & flowering", "91–120": "grain filling"}[lab]
    tl += (f'<rect x="{sx(lo)}" y="{LINE_Y + 18}" width="{sx(hi) - sx(lo) - 3}" height="34" rx="5" fill="{C_LIGHT}" stroke="#b9cde3"/>'
           f'<text x="{(sx(lo) + sx(hi)) / 2}" y="{LINE_Y + 32}" text-anchor="middle" font-size="10" font-weight="700" fill="#111111">{lab}</text>'
           f'<text x="{(sx(lo) + sx(hi)) / 2}" y="{LINE_Y + 45}" text-anchor="middle" font-size="9.5" fill="#4a4f57">{stage}</text>')
tl += f'<text x="{X0 - 6}" y="{LINE_Y + 38}" text-anchor="end" font-size="9.5" fill="#4a4f57">Weather</text>'
tl += f'<text x="{X0 - 6}" y="{LINE_Y + 49}" text-anchor="end" font-size="9.5" fill="#4a4f57">periods</text>'
TIMELINE = svg(700, LINE_Y + 62, tl)


# ------------------------------------------------------------------ tables
def table(df_, cls=""):
    return df_.to_html(index=False, border=0, classes=f"t {cls}", escape=False)


metrics_tbl = pd.DataFrame({
    "Model": ["<b>Hybrid Stacking</b>", "Gradient Boosting", "Decision Tree", "Random Forest", "Linear Regression"],
    "R²": ["<b>0.9903</b>", "0.9902", "0.9893", "0.9891", "0.9875"],
    "MAE (kg)": ["<b>658</b>", "665", "687", "694", "770"], "RMSE (kg)": ["<b>913</b>", "917", "957", "966", "1,033"],
    "MAPE (%)": ["<b>2.89</b>", "2.93", "3.03", "3.07", "3.66"], "Accuracy (100−MAPE)": ["<b>97.11%</b>", "97.07%", "96.97%", "96.93%", "96.34%"]})
cv_tbl = pd.DataFrame({
    "Model": ["<b>Hybrid Stacking</b>", "Gradient Boosting", "Decision Tree", "Random Forest", "Linear Regression"],
    "CV mean R²": ["<b>0.9916</b>", "0.9915", "0.9904", "0.9901", "0.9887"],
    "CV std R²": ["0.0011", "0.0011", "0.0009", "0.0008", "0.0015"],
    "CV mean MAE (kg)": ["<b>611</b>", "617", "651", "664", "744"]})
scratch_tbl = pd.DataFrame({
    "Model": ["scikit-learn Hybrid (final model)", "<b>Our from-scratch Hybrid (NumPy)</b>", "Ours · Gradient Boosting (depth 2)",
              "Ours · Gradient Boosting (depth 3)", "Ours · Random Forest", "Ours · Linear Regression"],
    "R²": ["0.9903", "<b>0.9902</b>", "0.9903", "0.9902", "0.9894", "0.9875"],
    "MAE (kg)": ["658", "<b>661</b>", "657", "665", "686", "770"], "MAPE (%)": ["2.89", "<b>2.91</b>", "2.88", "2.93", "3.02", "3.66"]})
dose_tbl = pd.DataFrame([{"Input (dataset column)": c, "Dose per hectare": f"{dose[c]:g} {u}",
                          "For 4 ha": f"{dose[c] * 4:,.1f} {u}", "When": ("Day " + str(d)) if d is not None else "Before transplanting"}
                         for c, d, _, u in PLAN])
clim_tbl = pd.DataFrame({"Agri-block": clim["Block"], "Season rainfall (mm)": clim["Rain"].round(0).astype(int),
                         "Avg max temp (°C)": clim["Tmax"].round(1), "Avg min temp (°C)": clim["Tmin"].round(1),
                         "Avg humidity (%)": clim["RH"].round(0).astype(int), "Avg yield (kg/ha)": clim["YpH"].round(0).astype(int),
                         "Farms": clim["N"]})
w_tbl = pd.DataFrame([{"Farmer priority": k, "Yield": f"{v[0]:.0%}", "Income": f"{v[1]:.0%}",
                       "Shorter duration": f"{v[2]:.0%}", "Fine grain": f"{v[3]:.0%}"} for k, v in WEIGHTS.items()])
var_tbl = pd.DataFrame({"Variety": [LABELS[v] for v in vinfo.index], "Duration (days)": vinfo["Duration_days"].values,
                        "Grain type": vinfo["Grain_type"].values, "Indicative price (₹/kg)": vinfo["Price_Rs_per_kg"].values,
                        "Avg yield in data (kg/ha)": [round(df[df.Variety == v]["Yield_per_ha"].mean()) for v in vinfo.index]})

# recommendation worked example
EX = dict(ha=4, block="Cuddalore", soil="clay", nursery="wet")
ex = recommend(EX["ha"], EX["block"], EX["soil"], EX["nursery"], "Maximum profit")
ex_tbl = pd.DataFrame({"Rank": range(1, 4), "Variety": ex["Variety"].map(LABELS),
                       "Predicted yield (kg)": ex["Yield"].map("{:,.0f}".format),
                       "Income (₹)": ex["Income"].map("{:,.0f}".format), "Duration": ex["Duration"],
                       "Yield pts": (ex["s_y"] * 100).round(1), "Income pts": (ex["s_i"] * 100).round(1),
                       "Duration pts": (ex["s_d"] * 100).round(1), "Score /100": (ex["Score"] * 100).round(0).astype(int)})
pri_rows = []
for p in WEIGHTS:
    t = recommend(EX["ha"], EX["block"], EX["soil"], EX["nursery"], p)
    pri_rows.append({"Farmer priority": p, "Recommended": f"<b>{LABELS[t.Variety[0]]}</b>",
                     "Runner-up": LABELS[t.Variety[1]], "Winning score": f"{t.Score[0] * 100:.0f}/100"})
pri_tbl = pd.DataFrame(pri_rows)

# calendar example
START = dt.date(2026, 8, 1)
cal_rows = []
for c, d, what, u in PLAN:
    if d is None:
        continue
    cal_rows.append({"Day": d, "Date (if sown 1 Aug 2026)": (START + dt.timedelta(days=d)).strftime("%d %b %Y"),
                     "Activity": what, "Amount for 4 ha": f"{dose[c] * 4:,.1f} {u}"})
for v in vinfo.index:
    d = int(vinfo.loc[v, "Duration_days"])
    cal_rows.append({"Day": d, "Date (if sown 1 Aug 2026)": (START + dt.timedelta(days=d)).strftime("%d %b %Y"),
                     "Activity": f"Expected harvest – {LABELS[v]}", "Amount for 4 ha": "—"})
cal_tbl = pd.DataFrame(cal_rows).sort_values("Day")

# per-block yield range for the chosen farm
best_ex = ex.iloc[0]
lo_ex, hi_ex = best_ex.Yield * (1 + BAND[0] / 100), best_ex.Yield * (1 + BAND[1] / 100)

# ------------------------------------------------------------------ HTML
CSS = """
@page { size: A4; margin: 18mm 16mm 18mm 16mm; }
* { box-sizing: border-box; }
body { font-family: 'Calibri', 'Segoe UI', Arial, sans-serif; color: #111111; font-size: 11pt; line-height: 1.55; margin: 0;
  -webkit-print-color-adjust: exact; print-color-adjust: exact; }
h1 { font-size: 20pt; color: #0f1f33; margin: 0 0 6px; border-bottom: 3px solid #1f3a5f; padding-bottom: 6px; }
h2 { font-size: 14pt; color: #1f3a5f; margin: 22px 0 6px; }
h3 { font-size: 12pt; color: #111111; margin: 16px 0 4px; }
p { margin: 6px 0; text-align: justify; }
.chapter { page-break-before: always; }
.cover { height: 255mm; display: flex; flex-direction: column; justify-content: space-between; text-align: center; padding-top: 10mm; }
.cover .band { background: linear-gradient(120deg,#0f1f33,#2e5c8a); color: #fff; border-radius: 14px; padding: 26px 20px; }
.cover .band h1 { color: #fff; border: none; font-size: 24pt; margin: 6px 0; }
.cover .band .sub { color: #d6e4f2; font-size: 12pt; }
.cover table { margin: 0 auto; }
table.cover-team th, table.cover-team td { text-align: center; }
.small { font-size: 9.5pt; color: #4a4f57; }
table.t { border-collapse: collapse; width: 100%; margin: 8px 0 12px; font-size: 9.5pt; page-break-inside: avoid; }
table.t th { background: #1f3a5f; color: #fff; padding: 6px 7px; text-align: left; font-weight: 600; }
table.t td { padding: 5px 7px; border-bottom: 1px solid #d9e1ea; }
table.t tr:nth-child(even) td { background: #f4f7fb; }
figure { margin: 10px 0 14px; page-break-inside: avoid; text-align: center; }
figure img { border: 1px solid #d9e1ea; border-radius: 6px; }
figcaption { font-size: 9.5pt; color: #4a4f57; margin-top: 4px; }
.diagram { border: 1px solid #d9e1ea; border-radius: 10px; padding: 10px; margin: 10px 0 4px; page-break-inside: avoid; background: #fbfcfe; }
.note { background: #f4f6f9; border-left: 4px solid #111111; padding: 8px 12px; margin: 10px 0; font-size: 10pt; page-break-inside: avoid; }
.key { background: #eaf1f8; border-left: 4px solid #1f3a5f; padding: 8px 12px; margin: 10px 0; font-size: 10pt; page-break-inside: avoid; }
.formula { font-family: Cambria, 'Times New Roman', serif; font-size: 11.5pt; background: #f4f7fb; padding: 6px 12px; border-radius: 6px; display: inline-block; margin: 4px 0; }
ul, ol { margin: 4px 0 8px 0; padding-left: 22px; } li { margin: 2px 0; }
.toc li { margin: 4px 0; } .toc { columns: 1; }
.keep { break-inside: avoid; page-break-inside: avoid; }
h2, h3 { break-after: avoid; page-break-after: avoid; }
.grid2 { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
"""

f = iter(range(1, 100))  # figure counter
H = []
add = H.append

add(f"""<div class="cover">
<div>
  <div class="small">Machine Learning Project · B1_Sec_A_Sem_V</div>
  <div class="band" style="margin-top:14px">
    <div class="sub">PROJECT REPORT</div>
    <h1>A Hybrid, Feature-Optimized Decision-Support Framework for Paddy Yield Prediction and Variety Recommendation</h1>
    <div class="sub">PaddyWise — Yield &amp; Variety Advisor</div>
  </div>
</div>
<div>
  <h3>Submitted by — Group 2</h3>
  <table class="t cover-team" style="width:70%; margin:8px auto 12px"><tr><th>Roll No.</th><th>Name</th></tr>
  <tr><td>2427010043</td><td>Akshat Kumar</td></tr><tr><td>2427010049</td><td>Kushal Bajaj</td></tr>
  <tr><td>2427010336</td><td>Chirag Sahu</td></tr></table>
</div>
<div class="small">Dataset: M. Subramaniyan, "Paddy Dataset," UCI Machine Learning Repository, 2023. https://doi.org/10.24432/C55W3J<br>
Tools: Python · pandas · NumPy · scikit-learn · Matplotlib · Seaborn · Plotly · Streamlit</div>
</div>""")

add(f"""<div class="chapter"><h1>Abstract</h1>
<p>Rice is the staple crop of Tamil Nadu, and farmers must decide which paddy variety to grow and how to manage it before
the season begins. This project develops a <b>hybrid, feature-optimized, decision-support machine-learning framework</b> on the
UCI Paddy Dataset ({N_RAW:,} farm records, {C_RAW} columns, six agri-blocks of Cuddalore/Kallakurichi districts).
The framework (1) identifies the key factors influencing paddy yield, (2) predicts rice yield, and (3) recommends the most
suitable variety according to field conditions and the farmer's own priority.</p>
<p>After cleaning ({N_DUP} duplicate rows removed) and exploratory analysis, a <b>three-stage feature optimization</b> removed a
data-leakage column, ten dosage columns that are exact multiples of farm area, and 28 weather columns that are fully
determined by the agri-block, reducing <b>44 inputs to 5 features</b> with no loss of accuracy. Five regressors were compared
on a hidden 20 % test set and with 5-fold cross-validation. The proposed <b>Hybrid Stacking model</b> (two Gradient Boosting
models and a Random Forest combined by a non-negative linear meta-model) achieved the best results:
<b>R² = {R2:.4f}, MAE = {MAE:,.0f} kg, RMSE = {RMSE:,.0f} kg, MAPE = {MAPE:.2f} %</b> (≈ {100 - MAPE:.1f} % accuracy) and a
cross-validated R² of 0.9916 ± 0.0011. The hybrid model was also re-implemented from scratch in NumPy and matched scikit-learn
to within 0.08 % on average.</p>
<p>Farm area is the dominant factor for total yield; for productivity per hectare the agri-block (location and microclimate),
variety, nursery type and soil follow. The decision-support layer predicts the yield of every variety for the farmer's
conditions, ranks them with a transparent weighted multi-criteria score, and generates a day-wise input plan
(seed, fertilizer, weedicide, pesticide, micronutrients) scaled to the farm. Everything is delivered through an interactive
Streamlit web application.</p>
<p><b>Keywords:</b> paddy yield prediction, stacking ensemble, gradient boosting, random forest, feature selection,
decision support system, multi-criteria recommendation.</p>

</div><div class="chapter"><h1>Table of Contents</h1>
@@TOC@@</div>""")

add(f"""<div class="chapter"><h1>1. Introduction</h1>
<h2>1.1 Background</h2>
<p>Paddy (rice) cultivation supports millions of farmers in India. Yield depends on many interacting factors: farm size,
location and its weather, the variety grown, soil, nursery method and input management. Farmers usually make these
decisions from experience. A data-driven tool that estimates the expected yield and suggests the best variety for a
given field can reduce risk and improve income.</p>
<h2>1.2 Problem statement</h2>
<div class="key">Develop a hybrid, feature-optimized and decision-support machine learning framework that identifies the key
factors influencing paddy growth, predicts rice yield, and subsequently recommends the most suitable paddy variety based on
agricultural conditions and farmers' preferences.</div>
<h2>1.3 Objectives</h2>
<ol>
<li><b>Clean and understand</b> the UCI Paddy Dataset (preprocessing and EDA).</li>
<li><b>Optimize the feature set</b>: remove leakage, redundant and dependent columns while keeping accuracy.</li>
<li><b>Build and compare</b> several regression models and propose a <b>hybrid stacking ensemble</b>.</li>
<li><b>Test</b> every model on unseen data with standard metrics and cross-validation.</li>
<li><b>Identify the dominant factors</b> that drive yield.</li>
<li><b>Recommend a variety</b> using the model's predictions and the farmer's priorities, and generate a
<b>day-wise input plan</b>.</li>
<li><b>Deploy</b> the framework as an easy-to-use web application.</li>
</ol>
<h2>1.4 Meaning of the three key words</h2>
<table class="t"><tr><th>Term</th><th>What it means in this project</th></tr>
<tr><td><b>Hybrid</b></td><td>Several different ML models are combined by stacking; a meta-model learns how much to trust each.</td></tr>
<tr><td><b>Feature-optimized</b></td><td>Only the informative, non-redundant, leakage-free inputs are kept (44 → 5).</td></tr>
<tr><td><b>Decision-support</b></td><td>The output is not only a number: the system ranks varieties for the farmer's priority and produces an input plan.</td></tr>
</table>

<h1 style="margin-top:24px">2. Dataset description</h1>
<p>The dataset was published by M. Subramaniyan on the UCI Machine Learning Repository (2023). Each row is one paddy farm
for one season. It contains <b>{N_RAW:,} rows and {C_RAW} columns</b>: 44 inputs and the target <b>Paddy yield (kg)</b>.</p>
<table class="t"><tr><th>Group</th><th>Columns</th><th>Examples</th></tr>
<tr><td>Farm &amp; location</td><td>4</td><td>Hectares (1–6), Agriblock (6 blocks), Variety (3), Soil type (clay / alluvial)</td></tr>
<tr><td>Nursery &amp; land preparation</td><td>5</td><td>Nursery (wet / dry), nursery area, seed rate, land-preparation inputs</td></tr>
<tr><td>Fertilizer &amp; crop protection</td><td>6</td><td>DAP @ 20 days, thiobencarb @ 28 days, urea @ 40 days, potash @ 50 days, pesticide @ 60 days, micronutrients @ 70 days</td></tr>
<tr><td>Rainfall (4 periods)</td><td>8</td><td>rainfall and "AI" values for days 1–30, 31–50, 51–70, 71–105</td></tr>
<tr><td>Temperature (4 periods)</td><td>8</td><td>min / max temperature for days 1–30, 31–60, 61–90, 91–120</td></tr>
<tr><td>Wind (4 periods)</td><td>8</td><td>wind speed (knots) and wind direction</td></tr>
<tr><td>Humidity (4 periods)</td><td>4</td><td>relative humidity</td></tr>
<tr><td>Post-harvest</td><td>1</td><td>Trash (straw bundles)</td></tr>
<tr><td><b>Target</b></td><td>1</td><td><b>Paddy yield (kg)</b>: 5,410 to 38,814 kg</td></tr></table>
{table(var_tbl)}
<p class="small">Duration, grain type and prices are general published characteristics of these varieties (prices are
indicative and editable in the app). They are <b>not</b> part of the dataset and are used only in the recommendation step.</p></div>""")

add(f"""<div class="chapter"><h1>3. Methodology: the overall framework</h1>
<p>Figure {next(f)} shows the complete pipeline. Each block is explained in its own chapter. The two dark blocks are the
core of the project: the <b>hybrid model</b> and the <b>decision-support layer</b>.</p>
<div class="diagram">{FLOW_MAIN}</div>
<p class="small" style="text-align:center"><b>Figure 1.</b> End-to-end methodology of the proposed framework.</p>
<h2>3.1 Tools used</h2>
<table class="t"><tr><th>Stage</th><th>Library</th><th>Used for</th></tr>
<tr><td>Data handling</td><td>pandas, NumPy</td><td>loading, cleaning, duplicates, one-hot encoding, arithmetic</td></tr>
<tr><td>Visualisation</td><td>Matplotlib, Seaborn, Plotly</td><td>EDA plots, result charts, interactive app charts</td></tr>
<tr><td>Machine learning</td><td>scikit-learn</td><td>train_test_split, KFold, cross_val_score, 5 regressors, StackingRegressor, metrics, permutation importance</td></tr>
<tr><td>Persistence</td><td>joblib</td><td>saving the trained model (paddy_model.pkl)</td></tr>
<tr><td>Deployment</td><td>Streamlit</td><td>the PaddyWise web application</td></tr></table></div>""")

add(f"""<div class="chapter"><h1>4. Data preprocessing</h1>
<p>Raw data usually contains formatting problems, repeated records and errors. Preprocessing makes the data reliable before
any model sees it. Figure {next(f)} shows the steps.</p>
<div class="diagram" style="width:85%;margin:auto">{FLOW_PRE}</div>
<p class="small" style="text-align:center"><b>Figure 2.</b> Preprocessing pipeline.</p>
<h2>4.1 Cleaning column names and text</h2>
<p>Several headers contained trailing spaces (e.g. <code>'Hectares '</code>) and long units
(e.g. <code>'Paddy yield(in Kg)'</code>). We stripped the spaces, removed the bracketed units and renamed the columns to short
names (<code>Yield</code>, <code>Soil</code>, <code>Seedrate</code> …). Text values such as block and variety names were
stripped too, so that "Cuddalore " and "Cuddalore" are not treated as two different blocks.</p>
<h2>4.2 Missing values</h2>
<p><code>df.isnull().sum()</code> returned <b>0</b> for every column, so no filling was needed.</p>
<h2>4.3 Duplicate removal</h2>
<p>We found <b>{N_DUP} exact duplicate rows</b> ({N_DUP / N_RAW:.1%} of the data). If a row appears in both the training and
the test set, the model is tested on a row it has already seen, which inflates the score. Removing them gives
<b>{len(df):,} unique rows</b> and an honest evaluation.</p>
<h2>4.4 Outliers</h2>
<p>Boxplots of total yield and yield per hectare (Figure {next(f)}) show no extreme points. Yield per hectare lies between
{df.Yield_per_ha.min():,.0f} and {df.Yield_per_ha.max():,.0f} kg/ha, a realistic range, so no rows were removed.</p>
{fig(A[14], "Outlier check: total yield (left) and yield per hectare (right).", 3)}
<h2>4.5 Categorical values</h2>
<table class="t"><tr><th>Column</th><th>Categories (rows after cleaning)</th></tr>
{''.join(f"<tr><td>{c}</td><td>{', '.join(f'{k} ({v})' for k, v in df[c].value_counts().items())}</td></tr>" for c in ["Agriblock", "Variety", "Soil", "Nursery"])}
</table></div>""")

next(f)
add(f"""<div class="chapter"><h1>5. Exploratory data analysis (EDA)</h1>
<p>EDA means studying the data with statistics and graphs before modelling, to understand which factors matter and to
spot problems.</p>
<div class="keep"><h2>5.1 Distribution of yield</h2>
{fig(A[18], "Distribution of total paddy yield. The peaks correspond to the six farm sizes (1–6 ha).", 4, "80%")}</div>
<div class="keep"><h2>5.2 Yield versus farm area</h2>
{fig(A[20], "Yield grows almost linearly with farm area (correlation 0.994).", 5, "80%")}</div>
<p>The correlation between Hectares and Yield is <b>0.994</b>, the strongest relationship in the data.</p>
<div class="keep"><h2>5.3 Yield per hectare by category</h2>
{fig(A[23], "Average yield per hectare by variety, soil, nursery type and agri-block (y-axis starts at 5,700 to make small differences visible).", 6)}</div>
<p>Differences between categories are small (tens of kg/ha) compared with the effect of area. Delux Ponni and CO 43 average
slightly more than Ponmani, and Sankarapuram has the highest block average.</p>
<h2>5.4 Agri-block climate profiles</h2>
<p>Every farm in the same block shares exactly the same weather values, so each block has one climate profile:</p>
{table(clim_tbl)}
<div class="keep"><h2>5.5 Correlation heatmap</h2>
{fig(A[25], "Correlation heatmap of numeric columns. The solid block shows that all input-dose columns are perfectly correlated with Hectares.", 7)}</div>
<div class="key"><b>EDA findings.</b> (1) Farm area dominates total yield. (2) All fertilizer, seed and pesticide columns are
Hectares × a fixed dose, so they repeat the same information. (3) Weather is constant within each agri-block.
(4) Variety, soil and nursery have small but real effects on productivity.</div></div>""")
for _ in range(4):
    next(f)

add(f"""<div class="chapter"><h1>6. Feature optimization</h1>
<p>Feature optimization keeps only useful, non-repeated, legitimately available inputs. A smaller feature set is faster,
easier to explain, less likely to overfit and simpler for a farmer to fill in. We applied three filters (Figure 8).</p>
<div class="diagram">{FUNNEL}</div>
<p class="small" style="text-align:center"><b>Figure 8.</b> Feature-optimization funnel: 44 → 5 features.</p>
<h2>6.1 Filter 1: data leakage</h2>
<p><b>Trash</b> (straw bundles) is measured <i>after</i> harvest. A farmer cannot know it when asking for a prediction, so
using it would be "cheating" (target leakage). It was removed.</p>
<h2>6.2 Filter 2: redundancy (correlation &gt; 0.95)</h2>
<p>Seed rate, nursery area, both land-preparation inputs, DAP, thiobencarb, urea, potash, micronutrients and pesticide all had
correlation <b>1.000</b> with Hectares. Dividing each by Hectares gives a constant: the standard dose per hectare.
They carry no information beyond Hectares, so they were removed, and their per-hectare doses were saved for the advice module (Chapter 11).</p>
<h2>6.3 Filter 3: functional dependency on Agriblock</h2>
<p>For all 28 weather columns, the number of distinct values <i>inside</i> any agri-block is exactly <b>1</b>
(<code>df.groupby("Agriblock")[weather_cols].nunique().max() = 1</code>). Knowing the block therefore fixes all the weather
values, so the single categorical feature <b>Agriblock</b> represents location and microclimate together.</p>
<h2>6.4 Final features and encoding</h2>
<div class="key"><b>Final features:</b> Hectares · Agriblock · Variety · Soil · Nursery</div>
<p>Text features were <b>one-hot encoded</b> (each category becomes a 0/1 column), giving 14 model columns: Hectares,
6 block columns, 3 variety columns, 2 soil columns and 2 nursery columns.</p>
<h2>6.5 Did optimization cost accuracy?</h2>
<table class="t"><tr><th>Feature set</th><th>Model input columns</th><th>Test R² (hybrid model)</th></tr>
<tr><td>All original columns (except Trash)</td><td>70</td><td>0.9903</td></tr>
<tr><td><b>Optimized features</b></td><td><b>14</b></td><td><b>0.9903</b></td></tr></table>
<p>Accuracy is identical with <b>80 % fewer columns</b>, which confirms that the removed columns were redundant.</p></div>""")

add(f"""<div class="chapter"><h1>7. Model development</h1>
<h2>7.1 Train/test split</h2>
<p>The {len(df):,} rows were shuffled with a fixed seed (42) and split into <b>80 % training (1,870 rows)</b> and
<b>20 % testing (468 rows)</b>. The test rows are hidden during training and used only for the final evaluation.</p>
<h2>7.2 Models compared</h2>
<table class="t"><tr><th>Model</th><th>How it works</th><th>Settings</th></tr>
<tr><td>Linear Regression</td><td>Fits ŷ = b + Σ wᵢxᵢ by least squares (normal equation).</td><td>default</td></tr>
<tr><td>Decision Tree</td><td>Asks yes/no questions; each split minimises the squared error of the two groups; a leaf predicts its average.</td><td>max_depth = 8</td></tr>
<tr><td>Random Forest</td><td>Many trees on bootstrap samples; prediction = average of trees (lower variance).</td><td>300 trees</td></tr>
<tr><td>Gradient Boosting</td><td>Starts from the mean; each new small tree fits the remaining errors and is added with learning rate η.</td><td>300 trees, depth 3, η = 0.05</td></tr>
<tr><td><b>Hybrid Stacking</b></td><td>Three base models combined by a meta-model trained on their out-of-fold predictions.</td><td>see 7.3</td></tr></table>
<h2>7.3 The proposed hybrid stacking model</h2>
<div class="diagram">{STACK}</div>
<p class="small" style="text-align:center"><b>Figure 9.</b> Architecture of the hybrid stacking regressor.</p>
<p><b>Base models.</b> A deeper Gradient Boosting model (depth 3) captures interactions such as area × location; a shallower one
(depth 2) captures smooth trends; a Random Forest gives stable, low-variance predictions.</p>
<p><b>Meta-model.</b> A Linear Regression with <b>non-negative weights</b> learns final = b + w₁p₁ + w₂p₂ + w₃p₃.
Negative weights would mean "do the opposite of a good model", which is not meaningful, so they are not allowed.</p>
<p><b>Training procedure.</b></p>
<ol><li>Split the training data into 5 <b>shuffled</b> folds.</li>
<li>For each fold, train every base model on the other 4 folds and predict this fold. This gives
<b>out-of-fold predictions</b>: every training farm gets a prediction from models that never saw it.</li>
<li>Train the meta-model on these out-of-fold predictions so that it learns realistic trust weights.</li>
<li>Retrain all base models on the full training set for final use.</li></ol>
<div class="note"><b>Why shuffle the folds?</b> The CSV is sorted by farm size. Without shuffling, a fold could contain only
6-hectare farms and the base models trained without it would never see that size.</div>
<p><b>Learned weights.</b> The meta-model learned w = [{', '.join(f'{w:.3f}' for w in meta.coef_)}] (GB depth 3, GB depth 2,
Random Forest) with intercept {meta.intercept_:.1f}. The Random Forest received weight 0: it was slightly less accurate and
its errors overlapped with those of the boosting models, so it added no new information. Stacking selects useful models
automatically.</p></div>""")

add(f"""<div class="chapter"><h1>8. Testing and model comparison</h1>
<p>Testing answers the question <i>"how well will the model work on farms it has never seen?"</i> Figure 10 shows the procedure.</p>
<div class="diagram" style="width:85%;margin:auto">{TESTFLOW}</div>
<p class="small" style="text-align:center"><b>Figure 10.</b> Testing and comparison procedure.</p>
<div class="keep"><h2>8.1 Evaluation metrics</h2>
<table class="t"><tr><th>Metric</th><th>Formula</th><th>Meaning</th><th>Good value</th></tr>
<tr><td>R²</td><td class="formula">1 − Σ(y − ŷ)² / Σ(y − ȳ)²</td><td>share of yield variation explained (vs always guessing the mean)</td><td>close to 1</td></tr>
<tr><td>MAE</td><td class="formula">(1/n) Σ |y − ŷ|</td><td>average error in kg</td><td>low</td></tr>
<tr><td>RMSE</td><td class="formula">√[(1/n) Σ (y − ŷ)²]</td><td>error that penalises big mistakes more</td><td>low</td></tr>
<tr><td>MAPE</td><td class="formula">(100/n) Σ |y − ŷ| / y</td><td>average error in %</td><td>low</td></tr>
<tr><td>Accuracy</td><td class="formula">100 − MAPE</td><td>easy-to-read "percentage correct"</td><td>high</td></tr></table></div>
<h2>8.2 Hold-out test results (468 unseen farms)</h2>
{table(metrics_tbl)}
{fig(A[45], "Test-set comparison of the five models: R² (left) and MAE (right).", 11)}
<h2>8.3 Cross-validation (stability check)</h2>
<p>One split might be lucky. In 5-fold cross-validation the data is split five different ways; each model is trained and
tested five times and the scores are averaged.</p>
{table(cv_tbl)}
<p>The hybrid model's R² in the five folds was 0.9903, 0.9912, 0.9910, 0.9921 and 0.9935: consistently high, with a very
small standard deviation.</p>
<div class="keep"><h2>8.4 Prediction quality and residuals</h2>
{fig(A[47], "Hybrid model: actual vs predicted yield (left) and distribution of errors (right) on the test set.", 12)}</div>
<p>Points lie on the diagonal and the residuals are centred on zero, so the model is accurate and unbiased.
80 % of test errors fall between <b>{BAND[0]:+.1f} %</b> and <b>{BAND[1]:+.1f} %</b> of the prediction; this band is shown
in the app as the "likely range".</p>
<h2>8.5 Comparison summary</h2>
<ul><li>All tree-based models beat Linear Regression, because the effects of area and block are not perfectly linear.</li>
<li>The <b>Hybrid Stacking</b> model is best on the test set (R² 0.9903, MAE 658 kg) <b>and</b> in cross-validation
(R² 0.9916, MAE 611 kg).</li>
<li>Gains over the best single model are small because all good models are near the limit of the data: farms with
identical conditions still differ by natural variation that no model can predict. The hybrid is never worse than its best
member and chooses the mix automatically.</li>
<li>Feature optimization (70 → 14 columns) did not change accuracy (Section 6.5).</li>
<li>Our from-scratch NumPy implementation reproduces the hybrid's accuracy (Chapter 12).</li>
<li>The web app re-checks the model live on random test farms (Chapter 13).</li></ul></div>""")

add(f"""<div class="chapter"><h1>9. Key factors influencing paddy yield (dominant features)</h1>
<h2>9.1 Method: permutation importance</h2>
<p>For each feature we randomly shuffle its values in the test set and measure how much R² drops. If shuffling a feature
destroys accuracy, the model depends on it. One-hot columns were summed back into their original feature.</p>
{fig(A[56], "Permutation importance for total yield (hybrid model).", 13, "80%")}
<p><b>Farm area (Hectares) accounts for ≈ 99.9 % of the importance for total yield.</b> This matches EDA: bigger farms
produce proportionally more paddy, and all inputs scale with area.</p>
<div class="keep"><h2>9.2 What drives productivity (yield per hectare)?</h2>
<p>To see the factors behind <i>productivity</i>, we trained a Random Forest on yield per hectare:</p>
{fig(A[58], "Feature importance for yield per hectare.", 14, "80%")}</div>
<table class="t"><tr><th>Rank</th><th>Factor</th><th>Importance</th><th>Interpretation</th></tr>
<tr><td>1</td><td><b>Farm area</b></td><td>77.9 %</td><td>productivity per hectare also changes with farm size (scale and management effect)</td></tr>
<tr><td>2</td><td><b>Agri-block</b></td><td>8.0 %</td><td>location and its microclimate (rainfall, temperature, humidity, wind)</td></tr>
<tr><td>3</td><td><b>Variety</b></td><td>7.0 %</td><td>genetic potential of CO 43 / Ponmani / Delux Ponni</td></tr>
<tr><td>4</td><td><b>Nursery type</b></td><td>3.6 %</td><td>wet vs dry nursery establishment</td></tr>
<tr><td>5</td><td><b>Soil type</b></td><td>3.5 %</td><td>clay vs alluvial</td></tr></table>
<div class="key"><b>Dominant feature:</b> Farm area. <b>Most important controllable choice:</b> variety (with location fixed for a
given farmer). Weather affects yield through the agri-block profile.</div></div>""")

add(f"""<div class="chapter"><h1>10. Decision support: variety recommendation</h1>
<p>The decision-support layer turns predictions into a recommendation. Figure 15 shows the full logic. Only one step is
machine learning (dark box); the others are transparent formulas.</p>
<div class="diagram" style="width:90%;margin:auto">{RECO}</div>
<p class="small" style="text-align:center"><b>Figure 15.</b> Recommendation engine.</p>
<h2>10.1 Step-by-step</h2>
<ol>
<li><b>Input:</b> farm area, agri-block, soil, nursery type, the farmer's priority and (optionally) market prices.</li>
<li><b>What-if farms:</b> three copies of the farm are created, identical except for the variety.</li>
<li><b>ML prediction:</b> the hybrid model predicts the yield of each copy, giving the expected yield of each variety under
exactly the farmer's conditions.</li>
<li><b>Criteria:</b> income = predicted yield × price; duration and grain type come from the variety profile.</li>
<li><b>Normalisation:</b> each criterion is rescaled to 0–1 across the three varieties:
<span class="formula">norm(x) = (x − min) / (max − min)</span>. For duration, shorter is better, so we use −duration.</li>
<li><b>Weighted score:</b> <span class="formula">Score = w<sub>y</sub>·Yield + w<sub>i</sub>·Income + w<sub>d</sub>·Shortness + w<sub>g</sub>·FineGrain</span></li>
<li><b>Ranking:</b> the variety with the highest score is recommended; the others are shown as alternatives.</li></ol>
<h2>10.2 Priority weights</h2>
{table(w_tbl)}
<div class="note">The weights express the farmer's priority; they are design choices, not learned from data, because the
dataset contains no record of which variety a farmer "should" have chosen. This weighted-sum approach is a standard
multi-criteria decision-making (MCDM) method. A farmer or extension officer can change the weights, and future work could
derive them with AHP or from real farmer feedback.</div>
<h2>10.3 Worked example</h2>
<p>Farm: <b>{EX['ha']} ha, {EX['block']}, {EX['soil']} soil, {EX['nursery']} nursery</b>, priority = <b>Maximum profit</b>
(weights: yield 20 %, income 70 %, duration 10 %).</p>
{table(ex_tbl)}
<p>CO 43 has the highest predicted yield, but Delux Ponni's fine grain earns a higher price (₹27 vs ₹22/kg), so its income
is far higher. With profit as the priority, <b>Delux Ponni</b> wins. Expected yield {best_ex.Yield:,.0f} kg
(likely range {lo_ex:,.0f}–{hi_ex:,.0f} kg), estimated income ₹{best_ex.Income:,.0f}.</p>
<h3>Same farm, different priorities</h3>
{table(pri_tbl)}
<p>The recommendation changes with what the farmer values. This is the "farmers' preferences" part of the problem statement.</p></div>""")

add(f"""<div class="chapter"><h1>11. Advice generation and day-wise recommendation plan</h1>
<p>Besides the variety, the system gives practical advice: <b>how much of each input to apply and when</b>, plus the
expected yield, the likely range, the income and the climate profile of the block.</p>
<div class="diagram" style="width:85%;margin:auto">{ADVICE}</div>
<p class="small" style="text-align:center"><b>Figure 16.</b> How the input plan is generated from the data.</p>
<h2>11.1 Where the doses come from</h2>
<p>In Chapter 6 we found that every input column equals Hectares × a fixed amount. Dividing by Hectares therefore recovers the
<b>standard management practice per hectare</b> used on these farms. Multiplying it by the farmer's area gives the amounts
for that farm. The timing (day 20, 28, 40 …) is part of the column names in the dataset.</p>
{table(dose_tbl)}
<div class="keep"><h2>11.2 Day-wise timeline</h2>
<div class="diagram">{TIMELINE}</div>
<p class="small" style="text-align:center"><b>Figure 17.</b> Day-wise crop calendar with the weather periods recorded in the dataset
and the expected harvest of each variety.</p></div>
<p>The four weather periods in the dataset (days 1–30, 31–60, 61–90, 91–120) line up with the crop stages: nursery and
establishment, tillering, panicle initiation and flowering, and grain filling. The input schedule follows these stages:
phosphorus early for roots (DAP, day 20), weed control before the canopy closes (day 28), nitrogen for tillering (urea,
day 40), potassium for grain filling and stem strength (day 50), pest protection at peak vegetative growth (day 60), and
micronutrients before flowering (day 70).</p>
<div class="keep"><h2>11.3 Date-wise calendar (example)</h2>
<p>If the farmer gives a start date, each "Day N" becomes a calendar date. Example for a 4-hectare farm sown on
<b>1 August 2026</b>:</p>
{table(cal_tbl)}</div>
<p class="small">"Day" is counted from the start of the crop season as recorded in the dataset. Units for the weedicide are
reported exactly as in the dataset. Doses reflect the practice recorded on the farms in the dataset; local agricultural
department guidance should take priority where it differs.</p>
<h2>11.4 Summary of the advice given to a farmer</h2>
<ol><li><b>Which variety to grow</b>, ranked for the chosen priority, with the reason (score breakdown).</li>
<li><b>Expected yield</b> (kg and tonnes), yield per hectare, and a likely range.</li>
<li><b>Estimated income</b> at the entered market prices.</li>
<li><b>Crop duration and expected harvest</b> for each variety.</li>
<li><b>Day-wise input plan</b>: seed, nursery area, land preparation, DAP, weedicide, urea, potash, pesticide and
micronutrients, in exact amounts for the farm.</li>
<li><b>Season climate</b> of the block (rainfall, temperature, humidity) for planning.</li></ol></div>""")

add(f"""<div class="chapter"><h1>12. From-scratch implementation and verification</h1>
<p>To show a full understanding of the algorithms, the complete hybrid model was re-implemented using <b>only NumPy</b>
(notebook <code>Hybrid_Model_From_Scratch.ipynb</code>): train/test split, metrics, Linear Regression (normal equation with
pseudo-inverse), non-negative Linear Regression (coordinate descent), Decision Tree (exhaustive SSE split search with running
sums), Random Forest (bootstrap + averaging), Gradient Boosting (sequential residual fitting), K-Fold and Stacking.</p>
{fig(B[23], "Our from-scratch Gradient Boosting: the test error falls as trees are added (log scale).", 18, "80%")}
<h2>12.1 Results on the same 468 test farms</h2>
{table(scratch_tbl)}
{fig(B[35], "Left: our hybrid vs scikit-learn's hybrid. Right: our hybrid vs actual yield.", 19)}
<p>Our implementation reproduces the test split exactly and its predictions differ from scikit-learn's by only
<b>0.08 % on average</b> (max 0.44 %). Both meta-models gave the Random Forest weight 0. The small differences come from
random sampling, 100 vs 300 forest trees, and scikit-learn's slightly different split criterion (Friedman MSE).</p></div>""")

add(f"""<div class="chapter"><h1>13. Web application (PaddyWise)</h1>
<p>The framework is deployed as an interactive Streamlit application (<code>app.py</code>) that loads the saved hybrid model.</p>
<table class="t"><tr><th>Tab</th><th>What it does</th></tr>
<tr><td><b>Yield Predictor</b></td><td>Live prediction as inputs change: yield, yield/ha, likely range, number of real farms with identical
conditions and their actual average; histogram placing the prediction among real farms; <b>live check on 6 random unseen test farms</b>
(actual vs predicted vs error %).</td></tr>
<tr><td><b>Variety Advisor</b></td><td>Recommended variety for the chosen priority, ranked cards for all varieties, score-breakdown chart,
yield-vs-farm-size chart, explanation of how the recommendation is made with the current weights, climate profile, day-wise input plan.</td></tr>
<tr><td><b>Data Insights</b></td><td>Interactive EDA charts, key factors, feature-optimization summary.</td></tr>
<tr><td><b>Model Performance</b></td><td>Live test-set metrics, actual-vs-predicted plot, model comparison, stacking architecture.</td></tr>
<tr><td><b>About</b></td><td>Problem statement, pipeline, team, tech stack, dataset citation.</td></tr></table>
<p><b>Running:</b> <code>pip install -r requirements.txt</code> then <code>streamlit run app.py</code>. The app can be deployed free
on Streamlit Community Cloud from a GitHub repository.</p>

<h1 style="margin-top:26px">14. Discussion and limitations</h1>
<ul>
<li><b>Data structure.</b> Inputs are exact multiples of area and weather is fixed per block, so the effective information is
five features. This limits how much management practices can be studied separately from farm size.</li>
<li><b>Whole-hectare farms only.</b> The data covers 1–6 ha in whole numbers; predictions outside this range are extrapolations,
so the app restricts the inputs.</li>
<li><b>Small variety differences.</b> Predicted yields of the three varieties differ by tens of kg/ha, so the recommendation
is driven mainly by price, duration and grain type through the priority weights.</li>
<li><b>Hand-chosen weights and external variety data.</b> Priority weights and variety profiles (duration, grain, price) are
not learned from the dataset; they are transparent and editable.</li>
<li><b>Irreducible noise.</b> Identical farms show different yields, so R² ≈ 0.99 is close to the practical maximum.</li></ul>

<h1 style="margin-top:26px">15. Conclusion and future scope</h1>
<p>We built a complete hybrid, feature-optimized, decision-support framework for paddy. Preprocessing removed {N_DUP}
duplicates; feature optimization reduced 44 inputs to 5 without loss of accuracy; the hybrid stacking model achieved
<b>R² = {R2:.4f}, MAE = {MAE:,.0f} kg, MAPE = {MAPE:.2f} %</b> on unseen farms, with stable cross-validation and an
independent from-scratch verification. Farm area is the dominant factor, followed by agri-block (microclimate), variety,
nursery and soil. The decision-support layer recommends the best variety for each farmer's priority and produces a day-wise,
area-scaled input plan, all available in an interactive web app.</p>
<p><b>Future scope:</b> live weather and market-price APIs; more districts, seasons and varieties; learning priority weights
from farmer feedback or AHP; soil-test (NPK) based fertilizer advice; a regional-language mobile interface.</p>

<h1 style="margin-top:26px">16. References</h1>
<ol class="small">
<li>M. Subramaniyan, "Paddy Dataset," UCI Machine Learning Repository, 2023. https://doi.org/10.24432/C55W3J</li>
<li>L. Breiman, "Random Forests," <i>Machine Learning</i>, 45(1), 5–32, 2001.</li>
<li>J. H. Friedman, "Greedy Function Approximation: A Gradient Boosting Machine," <i>Annals of Statistics</i>, 29(5), 2001.</li>
<li>D. H. Wolpert, "Stacked Generalization," <i>Neural Networks</i>, 5(2), 241–259, 1992.</li>
<li>F. Pedregosa et al., "Scikit-learn: Machine Learning in Python," <i>JMLR</i>, 12, 2825–2830, 2011.</li>
<li>A. Fisher, C. Rudin, F. Dominici, "All Models are Wrong, but Many are Useful" (permutation importance), <i>JMLR</i>, 20, 2019.</li>
<li>Tamil Nadu Agricultural University, TNAU Agritech Portal: rice varieties and crop production guide.</li></ol></div>""")

add("""<div class="chapter"><h1>Appendix A: Frequently asked viva questions</h1>
<table class="t"><tr><th style="width:32%">Question</th><th>Answer</th></tr>
<tr><td>Is this classification or regression?</td><td>Regression: the target (yield in kg) is a continuous number. The recommendation is a scoring rule on top of regression predictions.</td></tr>
<tr><td>Why remove duplicates?</td><td>A duplicate in both train and test lets the model be tested on data it has seen, inflating scores.</td></tr>
<tr><td>Why drop Trash?</td><td>It is measured after harvest: target leakage.</td></tr>
<tr><td>Why drop fertilizer and weather columns?</td><td>Fertilizer columns are Hectares × constant (r = 1.0); weather is fully determined by Agriblock. They add no new information; accuracy is unchanged.</td></tr>
<tr><td>What is one-hot encoding?</td><td>Turning a text category into 0/1 columns, one per category, so models can use it.</td></tr>
<tr><td>What is stacking?</td><td>An ensemble where base models' out-of-fold predictions train a meta-model that learns the best way to combine them.</td></tr>
<tr><td>What is the meta-model?</td><td>A non-negative Linear Regression whose inputs are the base models' predictions; its weights say how much to trust each model.</td></tr>
<tr><td>Why is the Random Forest weight 0?</td><td>It was slightly less accurate and its errors overlapped with the boosting models', so it added no new information; scikit-learn and our NumPy version both chose 0.</td></tr>
<tr><td>Why out-of-fold predictions?</td><td>Predictions on seen rows look too good; out-of-fold ones behave like predictions on new farms, so the learned weights are realistic.</td></tr>
<tr><td>What does R² = 0.99 mean?</td><td>The model explains 99 % of the variation in yield compared with always predicting the average.</td></tr>
<tr><td>Why not R² = 1?</td><td>Farms with identical conditions have different yields (natural variation that is not recorded).</td></tr>
<tr><td>Why cross-validation?</td><td>To check the score is not due to one lucky split; 5 different splits give a mean and standard deviation.</td></tr>
<tr><td>What is the dominant feature?</td><td>Farm area; for yield per hectare: area, then agri-block, variety, nursery, soil.</td></tr>
<tr><td>How is the variety recommended?</td><td>The ML model predicts each variety's yield for the farm; income = yield × price; criteria are normalised to 0–1 and combined with priority weights; the highest score wins.</td></tr>
<tr><td>Are the weights learned?</td><td>No. They encode the farmer's priority (a standard MCDM approach) and are editable; the dataset has no "correct variety" label to learn from.</td></tr>
<tr><td>Where does the day-wise plan come from?</td><td>Per-hectare doses recovered from the dataset (input ÷ hectares), multiplied by the farm area; timings come from the column names (day 20, 28, 40, 50, 60, 70).</td></tr>
<tr><td>How do you know your code is right?</td><td>Hidden test set + 5-fold CV + a from-scratch NumPy implementation that matches scikit-learn within 0.08 % + live checks on unseen farms in the app.</td></tr>
</table></div>""")

# ------------------------------------------------------------------ table of contents
# Pass 1 writes "–" for every page; build_pdfs.sh then finds each heading in the PDF, stores the
# page numbers in toc_pages.json and runs this script again (pass 2) to fill them in.
import json
import re

body = "".join(H)
heads = []
for level, text in re.findall(r"<h([12])[^>]*>(.*?)</h\1>", body, flags=re.S):
    text = re.sub(r"<[^>]+>", "", text).strip()
    if text in ("Abstract", "Table of Contents") or text.startswith("A Hybrid"):   # skip cover title
        continue
    heads.append((int(level), text))
json.dump(heads, open(os.path.join(os.path.dirname(OUT), "toc_heads.json"), "w", encoding="utf-8"), ensure_ascii=False)

pages_file = os.path.join(os.path.dirname(OUT), "toc_pages.json")
pages = json.load(open(pages_file, encoding="utf-8")) if os.path.exists(pages_file) else {}
rows = ['<tr class="lvl1"><td class="no"></td><td>Abstract</td><td class="pg">' + str(pages.get("Abstract", "–")) + "</td></tr>"]
for level, text in heads:
    m = re.match(r"^((?:\d+\.)+\d*|Appendix [A-Z]:?)\s*(.*)$", text)
    no, title = (m.group(1).rstrip(":"), m.group(2)) if m else ("", text)
    rows.append(f'<tr class="lvl{level}"><td class="no">{no}</td><td>{title}</td><td class="pg">{pages.get(text, "–")}</td></tr>')
TOC = ('<table class="toc"><thead><tr><th class="no">No.</th><th>Title</th><th class="pg">Page</th></tr></thead>'
       f'<tbody>{"".join(rows)}</tbody></table>')
H = [h.replace("@@TOC@@", TOC) for h in H]
CSS += """
table.toc { width: 100%; border-collapse: collapse; font-size: 10pt; margin-top: 6px; }
table.toc th { background: #1f3a5f; color: #fff; text-align: left; padding: 6px 8px; font-weight: 600; }
table.toc td { padding: 3.2px 8px; border-bottom: 1px solid #e3e8ef; }
table.toc .no { width: 92px; white-space: nowrap; } table.toc .pg { width: 60px; text-align: right; }
table.toc tr.lvl1 td { font-weight: 700; color: #0f1f33; background: #eaf1f8; border-top: 1px solid #b9cde3; padding-top: 5px; padding-bottom: 5px; }
table.toc tr.lvl2 td:nth-child(2) { padding-left: 22px; color: #333; }
table.toc tr.lvl2 td.no { color: #4a4f57; padding-left: 16px; }
"""

html = f"""<!doctype html><html><head><meta charset="utf-8"><title>Paddy ML Project Report</title><style>{CSS}</style></head>
<body>{''.join(H)}</body></html>"""
open(OUT, "w", encoding="utf-8").write(html)
print("wrote", OUT, f"{len(html) / 1e6:.1f} MB")
