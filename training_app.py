"""
╔══════════════════════════════════════════════════════════════╗
║  NucleicML — Training & Analysis App                        ║
║  Inputs: Field_MHz, T1_s, T2_s → Output: Nucleic_acid_class ║
╚══════════════════════════════════════════════════════════════╝
Run: streamlit run training_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import joblib
import os
import io
import json
import time
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, f1_score, roc_auc_score,
                             balanced_accuracy_score)
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                               ExtraTreesClassifier, AdaBoostClassifier)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA

# ─── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NucleicML · Training",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── THEME / STYLES ───────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
h1, h2, h3 { font-family: 'DM Serif Display', serif; }

.main { background: #0d1117; }
.block-container { padding-top: 1.5rem; }

.metric-card {
    background: linear-gradient(135deg, #161b22 0%, #1c2128 100%);
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 0.6rem;
}
.metric-value { font-size: 2rem; font-weight: 700; font-family: 'DM Mono'; color: #58a6ff; }
.metric-label { font-size: 0.78rem; color: #8b949e; letter-spacing: 0.08em; text-transform: uppercase; }

.section-header {
    background: linear-gradient(90deg, #1f2937, transparent);
    border-left: 3px solid #58a6ff;
    padding: 0.6rem 1rem;
    border-radius: 0 8px 8px 0;
    margin: 1.5rem 0 1rem 0;
    font-family: 'DM Serif Display', serif;
    font-size: 1.15rem;
    color: #e6edf3;
}
.winner-badge {
    background: linear-gradient(135deg, #1a3a1a, #1f4a1f);
    border: 1px solid #3fb950;
    border-radius: 8px;
    padding: 0.8rem 1.2rem;
    color: #3fb950;
    font-family: 'DM Mono';
}
.stDataFrame { font-family: 'DM Mono', monospace; }

div[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1117 0%, #161b22 100%);
    border-right: 1px solid #21262d;
}
.stButton > button {
    background: linear-gradient(135deg, #238636, #2ea043);
    color: white;
    border: none;
    border-radius: 8px;
    font-family: 'DM Sans';
    font-weight: 500;
    transition: all 0.2s;
}
.stButton > button:hover { transform: translateY(-1px); box-shadow: 0 4px 12px rgba(35,134,54,0.4); }
</style>
""", unsafe_allow_html=True)

# ─── HELPERS ──────────────────────────────────────────────────────────────────
PALETTE = ["#58a6ff", "#3fb950", "#f78166", "#d2a8ff", "#79c0ff", "#56d364"]
CLASS_COLORS = {
    "DNA_rare_intermediate":    "#f78166",
    "DNA_common_intermediate":  "#58a6ff",
    "RNA_intermediates":        "#3fb950",
    "RNA_oligonucleotides":     "#d2a8ff",
}

def set_dark_style():
    plt.rcParams.update({
        "figure.facecolor":  "#0d1117",
        "axes.facecolor":    "#161b22",
        "axes.edgecolor":    "#30363d",
        "axes.labelcolor":   "#8b949e",
        "xtick.color":       "#8b949e",
        "ytick.color":       "#8b949e",
        "text.color":        "#e6edf3",
        "grid.color":        "#21262d",
        "grid.linewidth":    0.8,
        "font.family":       "monospace",
        "font.size":         13,
        "axes.titlesize":    15,
        "axes.labelsize":    13,
        "xtick.labelsize":   12,
        "ytick.labelsize":   12,
        "legend.fontsize":   12,
        "figure.dpi":        350,
        "savefig.dpi":       350,
    })
set_dark_style()

@st.cache_data
def load_data(uploaded):
    try:
        df = pd.read_csv(uploaded, encoding="latin1")
    except Exception:
        df = pd.read_csv(uploaded, encoding="utf-8", errors="replace")
    return df

def prepare_data(df):
    """
    Strategy for columns 4 & 5:
      - Col 4 (Structure): full descriptive name → kept in metadata lookup dict
        to map predicted class → likely structure names (for the prediction app).
      - Col 5 (Class): sample-level abbreviation identifier → dropped from
        features (not useful for generalised classification; just an ID).
    Both are excluded from model training. Only cols 1-3 are features.
    """
    feature_cols = ["Field_MHz", "T1_s", "T2_s"]
    target_col   = "Nucleic_acid_intermediate"
    
    # Build class→structures lookup (used by prediction app)
    cls_map = (df.groupby(target_col)["Structure"]
                 .apply(lambda s: sorted(s.unique().tolist()))
                 .to_dict())

    X = df[feature_cols].copy()
    y = df[target_col].copy()
    return X, y, cls_map, feature_cols

def train_models(X_train, X_test, y_train, y_test, le):
    models = {
        "Random Forest":        RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
        "Extra Trees":          ExtraTreesClassifier(n_estimators=200, random_state=42, n_jobs=-1),
        "Gradient Boosting":    GradientBoostingClassifier(n_estimators=150, learning_rate=0.1, random_state=42),
        "AdaBoost":             AdaBoostClassifier(n_estimators=100, random_state=42, algorithm="SAMME"),
        "Logistic Regression":  LogisticRegression(max_iter=2000, C=1.0, random_state=42),
        "SVM (RBF)":            SVC(kernel="rbf", probability=True, C=10, gamma="scale", random_state=42),
        "K-Nearest Neighbors":  KNeighborsClassifier(n_neighbors=7),
        "Decision Tree":        DecisionTreeClassifier(max_depth=12, random_state=42),
        "Naive Bayes":          GaussianNB(),
        # early_stopping=False avoids np.isnan() crash on string label arrays in
        # older sklearn builds; we rely on max_iter=500 for convergence instead.
        "MLP Neural Net":       MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500,
                                              random_state=42, early_stopping=False),
    }

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_train)
    X_te_s = scaler.transform(X_test)

    # Models that need scaling
    scale_needed = {"Logistic Regression", "SVM (RBF)", "K-Nearest Neighbors", "MLP Neural Net"}

    # Encode string labels → integers so every model (especially MLP) gets a
    # numeric target array.  We decode predictions back to strings afterwards.
    y_tr_enc = le.transform(y_train)
    y_te_enc = le.transform(y_test)

    results = {}
    trained  = {}
    progress = st.progress(0)
    status   = st.empty()

    for i, (name, model) in enumerate(models.items()):
        status.markdown(f"⚙️ Training **{name}**…")
        t0 = time.time()
        Xtr = X_tr_s if name in scale_needed else X_train.values
        Xte = X_te_s if name in scale_needed else X_test.values

        model.fit(Xtr, y_tr_enc)

        # Decode integer predictions → original string class names
        y_pred_enc = model.predict(Xte)
        y_pred     = le.inverse_transform(y_pred_enc)
        y_prob     = model.predict_proba(Xte) if hasattr(model, "predict_proba") else None

        acc  = accuracy_score(y_test, y_pred)
        bacc = balanced_accuracy_score(y_test, y_pred)
        f1   = f1_score(y_test, y_pred, average="weighted")
        try:
            auc = roc_auc_score(
                y_te_enc,
                y_prob,
                multi_class="ovr", average="weighted"
            ) if y_prob is not None else np.nan
        except Exception:
            auc = np.nan

        elapsed = time.time() - t0
        results[name] = {
            "Accuracy": acc, "Balanced Acc": bacc,
            "F1 (weighted)": f1, "ROC-AUC": auc,
            "Train time (s)": round(elapsed, 2),
            "y_pred": y_pred, "y_prob": y_prob,
        }
        trained[name] = {"model": model, "scaled": name in scale_needed}
        progress.progress((i + 1) / len(models))

    status.empty()
    progress.empty()
    return results, trained, scaler

# ─── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧬 NucleicML")
    st.markdown("**Training & Analysis Suite**")
    st.markdown("---")
    st.markdown("#### Upload Dataset")
    uploaded = st.file_uploader("Choose CSV file", type=["csv"])
    st.markdown("---")
    st.markdown("#### Train/Test Split")
    test_size = st.slider("Test size", 0.10, 0.40, 0.20, 0.05)
    rand_seed = st.number_input("Random seed", 0, 9999, 42)
    st.markdown("---")
    st.markdown("---")
    # On Streamlit Cloud the filesystem is ephemeral — always use the download button.
    model_dir = "./models"  # writable locally; ignored on cloud (download button used)
    st.caption("Columns 1–3 → Features  \nColumn 4 → Structure (metadata)  \nColumn 5 → Class ID (dropped)  \nColumn 6 → Target label")

# ─── MAIN ─────────────────────────────────────────────────────────────────────
st.markdown("# 🧬 NucleicML — Training & Analysis")
st.markdown("*Multi-model classification of nucleic acid intermediates from NMR relaxation parameters*")

if uploaded is None:
    st.info("👈 Upload your dataset CSV using the sidebar to begin.")
    st.markdown("""
    **Expected columns:**
    | # | Column | Role |
    |---|--------|------|
    | 1 | `Field_MHz` | Feature (magnetic field strength) |
    | 2 | `T1_s` | Feature (longitudinal relaxation time) |
    | 3 | `T2_s` | Feature (transverse relaxation time) |
    | 4 | `Structure` | Metadata – full structure name (kept as lookup) |
    | 5 | `Class` | Metadata – sample abbreviation (dropped from training) |
    | 6 | `Nucleic_acid_intermediate` | **Target label** |
    """)
    st.stop()

# ── Load ──────────────────────────────────────────────────────────────────────
df = load_data(uploaded)
X, y, cls_map, feat_cols = prepare_data(df)

# ═══════════════════════════════════════════════════════════════════════
#  SECTION 1 – EXPLORATORY DATA ANALYSIS
# ═══════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-header">① Exploratory Data Analysis</div>', unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(f'<div class="metric-card"><div class="metric-value">{len(df):,}</div><div class="metric-label">Total Samples</div></div>', unsafe_allow_html=True)
with c2:
    st.markdown(f'<div class="metric-card"><div class="metric-value">{df["Nucleic_acid_intermediate"].nunique()}</div><div class="metric-label">Target Classes</div></div>', unsafe_allow_html=True)
with c3:
    st.markdown(f'<div class="metric-card"><div class="metric-value">{df["Structure"].nunique()}</div><div class="metric-label">Unique Structures</div></div>', unsafe_allow_html=True)
with c4:
    missing = df[feat_cols].isnull().sum().sum()
    st.markdown(f'<div class="metric-card"><div class="metric-value">{missing}</div><div class="metric-label">Missing Values</div></div>', unsafe_allow_html=True)

st.markdown("#### Dataset Preview")
st.dataframe(df.head(10), width='stretch')

# ── Stats ──────────────────────────────────────────────────────────────────
st.markdown("#### Descriptive Statistics (Features)")
st.dataframe(X.describe().T.style.format("{:.4f}"), width='stretch')

# ── Class distribution ─────────────────────────────────────────────────────
col_a, col_b = st.columns(2)

with col_a:
    st.markdown("#### Class Distribution")
    counts = y.value_counts()
    fig, ax = plt.subplots(figsize=(6, 4), dpi=350)
    colors = [CLASS_COLORS.get(c, "#58a6ff") for c in counts.index]
    bars = ax.barh(counts.index, counts.values, color=colors, edgecolor="#30363d", linewidth=0.6)
    for bar, val in zip(bars, counts.values):
        ax.text(val + 10, bar.get_y() + bar.get_height()/2,
                f"{val:,} ({val/len(y)*100:.1f}%)",
                va="center", fontsize=14, color="#8b949e")
    ax.set_xlabel("Count")
    ax.set_title("Target Class Frequency", color="#e6edf3", fontsize=15)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig, width='stretch')
    plt.close()

with col_b:
    st.markdown("#### Class Balance (Pie)")
    fig, ax = plt.subplots(figsize=(5, 4), dpi=350)
    wedge_colors = [CLASS_COLORS.get(c, "#58a6ff") for c in counts.index]
    wedges, texts, autotexts = ax.pie(
        counts.values, labels=None, autopct="%1.1f%%",
        colors=wedge_colors, startangle=140,
        wedgeprops=dict(edgecolor="#0d1117", linewidth=1.5),
        textprops=dict(color="#e6edf3", fontsize=14)
    )
    ax.legend(counts.index, loc="lower left", fontsize=13,
              labelcolor="#8b949e", framealpha=0.2)
    ax.set_title("Target Distribution", color="#e6edf3", fontsize=15)
    plt.tight_layout()
    st.pyplot(fig, width='stretch')
    plt.close()

# ── Feature distributions by class ────────────────────────────────────────
st.markdown("#### Feature Distributions by Class")
fig, axes = plt.subplots(1, 3, figsize=(14, 4), dpi=350)
for ax, col in zip(axes, feat_cols):
    for cls in y.unique():
        vals = X.loc[y == cls, col]
        ax.hist(vals, bins=40, alpha=0.65, label=cls,
                color=CLASS_COLORS.get(cls, "#58a6ff"),
                edgecolor="none", density=True)
    ax.set_title(col, color="#e6edf3", fontsize=15)
    ax.set_ylabel("Density")
    ax.grid(alpha=0.3)
axes[0].legend(fontsize=13, labelcolor="#e6edf3", framealpha=0.2)
plt.suptitle("Feature Distributions per Class", color="#e6edf3", fontsize=16, y=1.02)
plt.tight_layout()
st.pyplot(fig, width='stretch')
plt.close()

# ── Box plots ─────────────────────────────────────────────────────────────
st.markdown("#### Box Plots: Feature vs Class")
fig, axes = plt.subplots(1, 3, figsize=(14, 4), dpi=350)
tmp = pd.concat([X, y.rename("Class")], axis=1)
for ax, col in zip(axes, feat_cols):
    classes = sorted(tmp["Class"].unique())
    data_by_class = [tmp.loc[tmp["Class"] == c, col].values for c in classes]
    bp = ax.boxplot(data_by_class, patch_artist=True, notch=False,
                    medianprops=dict(color="#e6edf3", linewidth=2),
                    flierprops=dict(marker="o", markersize=2, alpha=0.3,
                                   markerfacecolor="#8b949e", markeredgecolor="none"))
    for patch, cls in zip(bp["boxes"], classes):
        patch.set_facecolor(CLASS_COLORS.get(cls, "#58a6ff"))
        patch.set_alpha(0.7)
    ax.set_xticklabels([c.replace("_", "\n") for c in classes], fontsize=11)
    ax.set_title(col, color="#e6edf3", fontsize=15)
    ax.grid(axis="y", alpha=0.3)
plt.suptitle("Feature Spread by Class", color="#e6edf3", fontsize=16, y=1.02)
plt.tight_layout()
st.pyplot(fig, width='stretch')
plt.close()

# ── Violin plots ──────────────────────────────────────────────────────────
st.markdown("#### Violin Plots: Feature Density per Class")
fig, axes = plt.subplots(1, 3, figsize=(14, 4), dpi=350)
for ax, col in zip(axes, feat_cols):
    classes = sorted(tmp["Class"].unique())
    data_by_class = [tmp.loc[tmp["Class"] == c, col].values for c in classes]
    parts = ax.violinplot(data_by_class, showmedians=True, showextrema=False)
    for i, (pc, cls) in enumerate(zip(parts["bodies"], classes)):
        pc.set_facecolor(CLASS_COLORS.get(cls, "#58a6ff"))
        pc.set_alpha(0.6)
    parts["cmedians"].set_edgecolor("#e6edf3")
    ax.set_xticks(range(1, len(classes)+1))
    ax.set_xticklabels([c.replace("_", "\n") for c in classes], fontsize=11)
    ax.set_title(col, color="#e6edf3", fontsize=15)
    ax.grid(axis="y", alpha=0.3)
plt.suptitle("Feature Density (Violin)", color="#e6edf3", fontsize=16, y=1.02)
plt.tight_layout()
st.pyplot(fig, width='stretch')
plt.close()

# ── Correlation & Scatter ──────────────────────────────────────────────────
col_c, col_d = st.columns(2)
with col_c:
    st.markdown("#### Feature Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(5, 4), dpi=350)
    corr = X.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, annot=True, fmt=".3f", cmap="RdYlBu_r",
                mask=mask, ax=ax, linewidths=0.5,
                linecolor="#0d1117", annot_kws={"size": 13},
                cbar_kws={"shrink": 0.8})
    ax.set_title("Pearson Correlation", color="#e6edf3")
    plt.tight_layout()
    st.pyplot(fig, width='stretch')
    plt.close()

with col_d:
    st.markdown("#### T1 vs T2 Scatter by Class")
    fig, ax = plt.subplots(figsize=(5, 4), dpi=350)
    for cls in y.unique():
        mask = y == cls
        ax.scatter(X.loc[mask, "T1_s"], X.loc[mask, "T2_s"],
                   c=CLASS_COLORS.get(cls, "#58a6ff"),
                   alpha=0.35, s=8, label=cls, edgecolors="none")
    ax.set_xlabel("T1_s (Longitudinal)")
    ax.set_ylabel("T2_s (Transverse)")
    ax.set_title("T1 vs T2 Relaxation Space", color="#e6edf3")
    ax.legend(fontsize=13, labelcolor="#e6edf3", framealpha=0.2, markerscale=3)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig, width='stretch')
    plt.close()

# ── PCA ───────────────────────────────────────────────────────────────────
st.markdown("#### PCA Projection (2D)")
pca = PCA(n_components=2)
Xs = StandardScaler().fit_transform(X)
Xpca = pca.fit_transform(Xs)
fig, ax = plt.subplots(figsize=(7, 4), dpi=350)
for cls in y.unique():
    mask = y == cls
    ax.scatter(Xpca[mask, 0], Xpca[mask, 1],
               c=CLASS_COLORS.get(cls, "#58a6ff"),
               alpha=0.4, s=9, label=cls, edgecolors="none")
ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)")
ax.set_title("PCA — 2D Feature Space", color="#e6edf3")
ax.legend(fontsize=13, labelcolor="#e6edf3", framealpha=0.2, markerscale=3)
ax.grid(alpha=0.3)
plt.tight_layout()
st.pyplot(fig, width='stretch')
plt.close()

# ── Structure breakdown ────────────────────────────────────────────────────
st.markdown("#### Structures per Class")
struct_count = df.groupby(["Nucleic_acid_intermediate", "Structure"]).size().reset_index(name="Count")
st.dataframe(struct_count.sort_values(["Nucleic_acid_intermediate","Count"], ascending=[True,False]),
             width='stretch', height=200)

# ═══════════════════════════════════════════════════════════════════════
#  SECTION 2 – MODEL TRAINING
# ═══════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-header">② Model Training</div>', unsafe_allow_html=True)

le = LabelEncoder()
y_enc = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=rand_seed, stratify=y
)
st.markdown(f"**Split:** {len(X_train):,} train / {len(X_test):,} test samples  "
            f"(stratified, seed={rand_seed})")

if st.button("🚀 Train All Models", width='stretch'):
    with st.spinner("Training in progress…"):
        results, trained, scaler = train_models(X_train, X_test, y_train, y_test, le)
    st.session_state["results"]  = results
    st.session_state["trained"]  = trained
    st.session_state["scaler"]   = scaler
    st.session_state["le"]       = le
    st.session_state["X_test"]   = X_test
    st.session_state["y_test"]   = y_test
    st.session_state["cls_map"]  = cls_map
    st.session_state["feat_cols"] = feat_cols
    st.success("✅ All models trained successfully!")

if "results" not in st.session_state:
    st.info("Click **Train All Models** to continue.")
    st.stop()

results  = st.session_state["results"]
trained  = st.session_state["trained"]
scaler   = st.session_state["scaler"]
le_obj   = st.session_state["le"]
X_test   = st.session_state["X_test"]
y_test   = st.session_state["y_test"]

# ═══════════════════════════════════════════════════════════════════════
#  SECTION 3 – EVALUATION
# ═══════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-header">③ Model Evaluation</div>', unsafe_allow_html=True)

# ── Leaderboard ───────────────────────────────────────────────────────────
metrics_df = pd.DataFrame({
    name: {k: v for k, v in m.items() if k not in ("y_pred", "y_prob")}
    for name, m in results.items()
}).T.reset_index(names="Model")
metrics_df = metrics_df.sort_values("F1 (weighted)", ascending=False)

best_name = metrics_df.iloc[0]["Model"]
st.markdown(f'<div class="winner-badge">🏆 Best Model: <strong>{best_name}</strong> — F1={metrics_df.iloc[0]["F1 (weighted)"]:.4f} | Acc={metrics_df.iloc[0]["Accuracy"]:.4f}</div>', unsafe_allow_html=True)

st.markdown("#### Leaderboard")
st.dataframe(
    metrics_df.drop(columns=["y_pred","y_prob"] if "y_pred" in metrics_df.columns else [])
    .style.format({c: "{:.4f}" for c in ["Accuracy","Balanced Acc","F1 (weighted)","ROC-AUC","Train time (s)"]})
    .highlight_max(subset=["Accuracy","F1 (weighted)","ROC-AUC"], color="#1f3a1f")
    .highlight_min(subset=["Train time (s)"], color="#1f2a3a"),
    width='stretch'
)

# ── Bar chart comparison ───────────────────────────────────────────────────
st.markdown("#### Metric Comparison")
met_cols = ["Accuracy", "Balanced Acc", "F1 (weighted)", "ROC-AUC"]
fig, axes = plt.subplots(1, 4, figsize=(16, 4), dpi=350)
for ax, met in zip(axes, met_cols):
    vals = [results[n].get(met, np.nan) for n in metrics_df["Model"]]
    colors_bar = ["#3fb950" if n == best_name else "#30363d" for n in metrics_df["Model"]]
    bars = ax.barh(metrics_df["Model"], vals, color=colors_bar, edgecolor="#21262d")
    ax.set_xlim(0, 1.05)
    ax.set_title(met, color="#e6edf3", fontsize=14)
    ax.grid(axis="x", alpha=0.3)
    for bar, v in zip(bars, vals):
        if not np.isnan(v):
            ax.text(v + 0.005, bar.get_y() + bar.get_height()/2,
                    f"{v:.3f}", va="center", fontsize=13, color="#8b949e")
plt.suptitle("Model Performance Comparison", color="#e6edf3", fontsize=16, y=1.02)
plt.tight_layout()
st.pyplot(fig, width='stretch')
plt.close()

# ── Confusion matrices ────────────────────────────────────────────────────
st.markdown("#### Confusion Matrices")
top_models = metrics_df["Model"].head(6).tolist()
fig, axes = plt.subplots(2, 3, figsize=(16, 9), dpi=350)
axes = axes.flatten()
for ax, name in zip(axes, top_models):
    y_pred = results[name]["y_pred"]
    cm = confusion_matrix(y_test, y_pred, labels=le_obj.classes_)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(le_obj.classes_)))
    ax.set_yticks(range(len(le_obj.classes_)))
    ax.set_xticklabels([c.replace("_", "\n") for c in le_obj.classes_], fontsize=11)
    ax.set_yticklabels([c.replace("_", "\n") for c in le_obj.classes_], fontsize=11)
    for i in range(len(le_obj.classes_)):
        for j in range(len(le_obj.classes_)):
            ax.text(j, i, f"{cm[i,j]}\n({cm_norm[i,j]:.2f})",
                    ha="center", va="center",
                    fontsize=13,
                    color="white" if cm_norm[i,j] > 0.5 else "#8b949e")
    star = " ★" if name == best_name else ""
    ax.set_title(f"{name}{star}", color="#e6edf3", fontsize=14)
    ax.set_xlabel("Predicted", fontsize=13)
    ax.set_ylabel("True", fontsize=13)
plt.suptitle("Confusion Matrices (normalised)", color="#e6edf3", fontsize=16, y=1.01)
plt.tight_layout()
st.pyplot(fig, width='stretch')
plt.close()

# ── Per-class report for best ─────────────────────────────────────────────
st.markdown(f"#### Classification Report — {best_name}")
y_pred_best = results[best_name]["y_pred"]
report = classification_report(y_test, y_pred_best, target_names=le_obj.classes_, output_dict=True)
report_df = pd.DataFrame(report).T
st.dataframe(report_df.style.format("{:.4f}"), width='stretch')

# ── Feature importance (if applicable) ────────────────────────────────────
tree_models = [n for n in top_models if hasattr(trained[n]["model"], "feature_importances_")]
if tree_models:
    st.markdown("#### Feature Importances (Tree-based Models)")
    fig, axes = plt.subplots(1, min(3, len(tree_models)), figsize=(6*min(3, len(tree_models)), 4), dpi=350)
    if len(tree_models) == 1: axes = [axes]
    for ax, name in zip(axes, tree_models[:3]):
        imp = trained[name]["model"].feature_importances_
        ax.barh(feat_cols, imp, color=PALETTE[:3])
        ax.set_title(name, color="#e6edf3", fontsize=14)
        ax.grid(axis="x", alpha=0.3)
    plt.suptitle("Feature Importances", color="#e6edf3", fontsize=15, y=1.02)
    plt.tight_layout()
    st.pyplot(fig, width='stretch')
    plt.close()

# ═══════════════════════════════════════════════════════════════════════
#  SECTION 4 – SAVE MODEL
# ═══════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-header">④ Save Best Model</div>', unsafe_allow_html=True)

col_save1, col_save2 = st.columns(2)
with col_save1:
    selected_model = st.selectbox("Choose model to save", options=metrics_df["Model"].tolist(),
                                   index=0, help="Default is the top-ranked model")
with col_save2:
    save_name = st.text_input("File name (no extension)", value="best_nucleic_model")

model_obj   = trained[selected_model]["model"]
needs_scale = trained[selected_model]["scaled"]

bundle = {
    "model":        model_obj,
    "scaler":       scaler if needs_scale else None,
    "needs_scaler": needs_scale,
    "label_encoder": le_obj,
    "feature_cols": feat_cols,
    "class_to_structures": cls_map,
    "model_name":   selected_model,
    "metrics": {k: v for k, v in results[selected_model].items()
                if k not in ("y_pred", "y_prob")},
}

# Build download bytes (works on local AND cloud)
buf = io.BytesIO()
joblib.dump(bundle, buf)
buf.seek(0)

# Also save to disk locally if possible (silently ignored on cloud)
try:
    os.makedirs(model_dir, exist_ok=True)
    pkl_path  = os.path.join(model_dir, f"{save_name}.pkl")
    meta_path = os.path.join(model_dir, f"{save_name}_meta.json")
    joblib.dump(bundle, pkl_path)
    meta = {
        "model_name":   selected_model,
        "feature_cols": feat_cols,
        "classes":      list(le_obj.classes_),
        "needs_scaler": needs_scale,
        "class_to_structures": cls_map,
        "metrics": {k: (float(v) if isinstance(v, (np.floating, float)) else v)
                    for k, v in bundle["metrics"].items()},
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    st.success(f"✅ Also saved locally to `{pkl_path}`")
except Exception:
    pass  # Cloud: filesystem is ephemeral — download button is the primary save method

st.info("⬇️ Click below to download your trained model. Upload this .pkl into the Prediction App.")
st.download_button(
    label=f"⬇️ Download  {save_name}.pkl",
    data=buf,
    file_name=f"{save_name}.pkl",
    mime="application/octet-stream",
    width='stretch'
)
st.markdown(f"**Model:** {selected_model}  \n"
            f"**Accuracy:** {results[selected_model]['Accuracy']:.4f}  \n"
            f"**F1 (weighted):** {results[selected_model]['F1 (weighted)']:.4f}")

st.markdown("---")
st.caption("NucleicML · Nucleic Acid Intermediate Classifier · Built with Streamlit & scikit-learn")
