import streamlit as st
import numpy as np
import pandas as pd
import torch

import pathlib

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy.signal import welch, find_peaks, hilbert




# ─────────────────────────────────────────────
# Path to load data
# ─────────────────────────────────────────────
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"


# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="SSA · Store Sales",
    page_icon="📈",
    layout="wide",
)

# ─────────────────────────────────────────────
# Minimal CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background:  #f8fafc;
        border-radius: 8px;
        padding: 16px 20px;
        border-left: 4px solid #2563eb;
    }
    .metric-label { font-size: 12px; color: #6b7280; text-transform: uppercase; letter-spacing: .05em; }
    .metric-value { font-size: 26px; font-weight: 700; color: #111827; margin-top: 2px; }
    .section-title { font-size: 20px; font-weight: 600; color: #1e3a5f; margin-bottom: 8px; }
    .info-box {
    background-color:  #f8fafc;
    border-left: 5px solid #FB7185;
    color: #1F2937;
    padding: 16px;
    border-radius: 8px;
    line-height: 1.6;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# SSA functions (self-contained, no src import)
# ─────────────────────────────────────────────
def run_svd(ts, L, q=None):
    ts = torch.as_tensor(np.array(ts), dtype=torch.float64)
    N = len(ts)
    K = N - L + 1
    X = ts.unfold(0, K, 1)
    if q is None:
        U, S, VT = torch.linalg.svd(X, full_matrices=False)
    else:
        torch.manual_seed(42)
        U, S, V = torch.svd_lowrank(X, q=min(q, min(X.shape)))
        VT = V.T
    del X
    return {"U": U, "S": S, "VT": VT}


def diagonal_average(Y):
    L, K = Y.shape
    N = L + K - 1
    counts = torch.zeros(N, dtype=Y.dtype)
    y = torch.zeros(N, dtype=Y.dtype)
    for i in range(L):
        idx = torch.arange(i, i + K)
        y.index_add_(0, idx, Y[i])
        counts.index_add_(0, idx, torch.ones(K, dtype=Y.dtype))
    return y / counts


def reconstruct_group(triple, group):
    U, S, VT = triple["U"], triple["S"], triple["VT"]
    L, K = U.shape[0], VT.shape[1]
    X_group = torch.zeros((L, K), dtype=U.dtype)
    for i in group:
        X_group += S[i] * torch.outer(U[:, i], VT[i, :])
    ts_group = diagonal_average(X_group)
    del X_group
    return ts_group.cpu().numpy()


def reconstruct_all(triple, groups):
    return {name: reconstruct_group(triple, grp) for name, grp in groups.items()}


def w_correlation_matrix(ts_list, L):
    r = len(ts_list)
    N = ts_list[0].shape[0]
    K = N - L + 1
    n = torch.arange(N, dtype=torch.float64)
    w = torch.minimum(
        torch.minimum(n + 1, torch.tensor(L, dtype=torch.float64)),
        torch.minimum(torch.tensor(K, dtype=torch.float64), N - n),
    )
    F = torch.stack([torch.as_tensor(x, dtype=torch.float64) for x in ts_list])
    F_w = F * w.unsqueeze(0)
    W_ip = F_w @ F.T
    norms = torch.sqrt(torch.diag(W_ip))
    return (W_ip / torch.outer(norms, norms)).abs().numpy()


# ─────────────────────────────────────────────
# Hard-coded analysis results
# ─────────────────────────────────────────────
GROUPS_OSC = {
    "T≈3.5d (A)":  [0, 1],
    "T≈7d":        [2, 3],
    "T≈2.3d (A)":  [4, 5],
    "T≈2.3d (B)":  [6, 7],
    "T≈3.5d (B)":  [8, 9],
    "T≈20d":       [10, 11],
    "T≈15.3d":     [16, 17],
    "T≈11.7d":     [18, 19],
}

GROUP_COLORS = {
    "T≈3.5d (A)": "#2563eb",
    "T≈7d":       "#16a34a",
    "T≈2.3d (A)": "#dc2626",
    "T≈2.3d (B)": "#9333ea",
    "T≈3.5d (B)": "#ea580c",
    "T≈20d":      "#0F766E",
    "T≈15.3d":    "#CA8A04",
    "T≈11.7d":    "#C026D3",  
    "Noise":      "#9CA3AF", 
}

VARIANCE_TABLE = pd.DataFrame([
    {"Group": "T≈3.5d (A)",  "Components": "[1,2]",    "Period (days)": 3.5,  "Contribution (%)": 14.87, "Cumulative (%)": 14.87},
    {"Group": "T≈7d",        "Components": "[3,4]",    "Period (days)": 7.0,  "Contribution (%)": 14.29, "Cumulative (%)": 29.16},
    {"Group": "T≈2.3d (A)",  "Components": "[5,6]",    "Period (days)": 2.3,  "Contribution (%)": 13.87, "Cumulative (%)": 43.03},
    {"Group": "T≈2.3d (B)",  "Components": "[7,8]",    "Period (days)": 2.3,  "Contribution (%)":  6.36, "Cumulative (%)": 49.39},
    {"Group": "T≈3.5d (B)",  "Components": "[9,10]",    "Period (days)": 3.5,  "Contribution (%)":  2.73, "Cumulative (%)": 52.12},
    {"Group": "T≈20d",       "Components": "[11,12]",  "Period (days)": 20.0, "Contribution (%)":  1.20, "Cumulative (%)": 53.32},
    {"Group": "T≈15.3d",     "Components": "[17,18]",  "Period (days)": 15.3, "Contribution (%)":  0.97, "Cumulative (%)": 54.22},
    {"Group": "T≈11.7d",     "Components": "[19, 20]",  "Period (days)": 11.7, "Contribution (%)": 0.97, "Cumulative (%)": 55.05},
])


# ─────────────────────────────────────────────
# Sidebar 
# ─────────────────────────────────────────────

# Fixed SSA parameters
L1 = 14
L2 = 840


with st.sidebar:

    st.title("Project Information")

    st.markdown("""
**Sequential Singular Spectrum Analysis (SSA)** applied to store sales.

The workflow uses two sequential SSA decompositions to separate the series into **trend, amplitude-modulated oscillatory modes, and noise**.
""")

    st.markdown("---")

    st.markdown("### Dataset")

    st.write("**Source:** Kaggle – Store Sales Time Series Forecasting")
    st.write("**Store:** #1")
    st.write("**Product Family:** Beverages")
    st.write("**Period:** 2013–2017")
    st.write("**Frequency:** Daily")

    st.markdown("---")

    st.markdown("### SSA Parameters")

    st.write(f"**Stage 1:** Window length L = {L1}")
    st.write(f"**Stage 2:** Window length L = {L2}")
    

    st.markdown("---")

    st.caption("Sequential SSA workflow implemented in Python.")


# ─────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────
st.title("📈 Store Sales — Singular Spectrum Analysis Decomposition")
st.markdown(
    "Singular Spectrum Analysis (SSA) applied to **Beverages sales, Store #1** "
    "(2013–2017). The analysis separates trend, amplitude-modulated oscillatory "
    "components, and noise via two sequential SSA stages."
)

# ─────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_DIR / "sales.csv", parse_dates=["date"]).set_index("date")
    return df["sales"]

sales = load_data()
ts_np = sales.values.astype(float)
dates = sales.index


# ---------- Sequential SSA ----------
triple1 = run_svd(ts_np, L1)
trend = reconstruct_group(triple1, [0])     
residual = ts_np - trend

triple2 = run_svd(residual, L2)

# ---- Create Noise automatically ----
used_idx = sorted(i for g in GROUPS_OSC.values() for i in g)
ncomp = len(triple2["S"])

GROUPS_OSC["Noise"] = [
    i for i in range(ncomp)
    if i not in used_idx
]

# ---- Reconstruct groups ----
grp_rec = reconstruct_all(triple2, GROUPS_OSC)

oscillations = np.zeros_like(residual)

for k, v in grp_rec.items():
    if k != "Noise":
        oscillations += v
        
noise_sig = grp_rec["Noise"]
reconstructed = trend + oscillations + noise_sig

W = w_correlation_matrix(
    [grp_rec[m] for m in GROUPS_OSC if m != "Noise"],
    L2
)




# ─────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "🔍 Original Series",
    "📈 First Stage",
    "🔄 Second Stage: Visual",
    "📋 Second Stage: Grouping",
    "📅 Weekly Oscillatory Modes",
    "🌊 Longer Period and Noise",
    "🧩 Reconstruction",
   
])



# ══════════════════════════════════════════════
# TAB 1 — Original series
# ══════════════════════════════════════════════
with tab1:

    st.markdown("""
        ## Sequential SSA Workflow
        """)

    col_a, col_b = st.columns([1.0, 1.5])

        
    with col_a:
        
        graph = """
        digraph SSA {
        
            rankdir=TB;
            bgcolor="transparent";
            nodesep=0.45;
            ranksep=0.60;
        
            node [
                shape=box,
                style="rounded,filled",
                fillcolor="#FFF4EE",
                color="#2563EB",
                fontname="Helvetica",
                fontsize=11
            ];
        
            A [label="Original\\nTime Series"];
            B [label="Stage 1\\nL = 14"];
            C [label="Trend"];
            D [label="Residual"];
            E [label="Stage 2\\nL = 840"];
            F [label="Oscillatory\\nModes"];
            G [label="Noise"];
            H [label="Reconstruction"];
        
            // Stage 2 a la derecha del residual
            { rank=same; D; E; }
        
            // Oscillatory Modes y Noise debajo de Stage 2
            { rank=same; F; G; }
        
            A -> B;
            B -> C;
            B -> D;
            D -> E;
        
            E -> F;
            E -> G;
        
            C -> H;
            F -> H;
            G -> H;
        }
        """
        st.graphviz_chart(graph)
    

    with col_b:
        st.markdown("""             
            #### Stage 1 — Trend Extraction (Small Window Length):
            First, SSA is applied to the original time series by:

           -  Embedding the series into a Hankel matrix.
           -  Applying Singular Value Decomposition (SVD) to the Hankel matrix.
           -  Inspecting the first elementary components visually.
           - Extracting the trend.
             """)
        
       
        st.markdown("""
           
            #### Stage 2 — Oscillatory Mode Analysis (Large Window Length)
         A second SSA is applied to the residual of Stage 1 by the following steps:
         
        - Embedding the residual into a Hankel matrix.
        - Applying Singular Value Decomposition (SVD) to the Hankel matrix.
        - Visually inspecting the elementary components.
        - Grouping elementary components based on their oscillatory behavior.
        - Reconstructing the oscillatory modes and noise.
        - Reconstructing the time series
        
        """) 



    
    st.markdown("---")        
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    col_a, col_b = st.columns([2.0, 0.6])
    
    with col_a:
    
        st.markdown("""  ### Original Sales Time Series
         """)
        fig_original = go.Figure()
        fig_original.add_trace(go.Scatter(x=dates, y=ts_np, mode="lines",
                                 line=dict(color="#2563eb", width=0.9), name="Sales"))
        fig_original.update_layout(
            xaxis_title="Date", yaxis_title="Beverage Sales (units)",
            height=360, template="plotly_white",
            margin=dict(l=50, r=20, t=20, b=40),
        )
        st.plotly_chart(fig_original, key = "original series", use_container_width=True)

      
    
    with col_b:
        st.markdown(
            '<div class="section-title">Descriptive Statistics</div>',
            unsafe_allow_html=True
        )
    
       
        # Descriptive statistics
        stats = sales.describe().rename("Value").to_frame()
        stats["Value"] = stats["Value"].round(2)
    
        st.dataframe(
            stats,
            use_container_width=True,
            height=290
        )

      
        
         # Data quality card
        missing = sales.isna().sum()
    
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Missing Values</div>
            <div class="metric-value">{missing} ✓</div>
        </div>
        """, unsafe_allow_html=True)
    
      
    
    

# ══════════════════════════════════════════════
# TAB 2 — First stage: Visual Inspection -Trend Extraction
# ══════════════════════════════════════════════
with tab2:
    st.markdown("""
        ## Stage 1: Initial Decomposition and Trend Extraction
        """)

    col_a, col_b = st.columns(2)
     
    with st.spinner("Running SSA Stage 1 (trend)…"):
       ev1 = (triple1["S"].cpu().numpy() ** 2)
       U = triple1["U"].cpu().numpy()

    with col_a:
         # scree plot
        st.markdown('<div class="section-title">SSA Scree Plot — Eigenvalue Spectrum</div>',
                unsafe_allow_html=True,
                   )   
        x_scree = np.arange(1, min(51, len(ev1) + 1))
        y_scree = ev1[:50]
    
        fig_sc = go.Figure()
        fig_sc.add_trace(go.Scatter(
            x=x_scree, y=y_scree, mode="lines+markers",
            line=dict(color="#2563eb", width=2),
            marker=dict(size=5), name=f"L={L1}",
        ))
        fig_sc.update_layout(
            xaxis_title="Component Index", yaxis_title="Eigenvalue",
            yaxis_type="log", height=400, template="plotly_white",
            margin=dict(l=60, r=20, t=20, b=40),
        )
        fig_sc.update_yaxes(
        exponentformat="power"
        )
        st.plotly_chart(fig_sc, key = "eigenvalues", use_container_width=True)


        st.markdown("""
        <div class="info-box">
        <ul>
        <li><b>Scree plot</b>
            <ul>
             <li>The first component accounts for most of the variance, representing the dominant trend.</li>
             <li>A clear elbow after the first component marks the transition from the trend to other structures like oscillatory modes.</li>
            </ul>
        </li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
    with col_b:
    # Eigenvectors
        st.markdown(
            '<div class="section-title">Leading Eigenvectors</div>',
            unsafe_allow_html=True
        )
        
        fig_ev = go.Figure()
        
        for i in range(6):
            fig_ev.add_trace(
                go.Scatter(
                    x=np.arange(len(U[:, i])),
                    y=U[:, i],
                    mode="lines",
                    name=f"EV {i+1}"
                )
            )
        
        fig_ev.update_layout(
            xaxis_title="Index",
            yaxis_title="Amplitude",
            height=450,
            template="plotly_white",
            margin=dict(l=60, r=20, t=20, b=40),
        )
        st.plotly_chart(fig_ev, key = "eignvectors", use_container_width=True)

        st.markdown("""
        <div class="info-box">
            </ul>
            <li><b>Eigenvectors</b>
                <ul>
                    <li>The first eigenvector is nearly constant, which is characteristic of a trend component.</li>
                    <li>The remaining eigenvectors display oscillatory patterns associated with higher-frequency variations.</li>
                </ul>
            </li>
            </ul>
            </div>
    """, unsafe_allow_html=True)
        
    
    
    
    st.markdown("<br>", unsafe_allow_html=True)   
    
    st.markdown("""
        #### Reconstructed Trend (First Component) and Original Series
        """)
        
    fig_trend = go.Figure()
    
    # Original series
    fig_trend.add_trace(
        go.Scatter(
            x=dates,
            y=ts_np,
            mode="lines",
            line=dict(color="#60A5FA", width=1),
            name="Original Series"
        )
    )

    # Trend
    fig_trend.add_trace(
        go.Scatter(
            x=dates,
            y=trend,
            mode="lines",
            line=dict(color="#111827", width=2),
            name="Trend"
        )
    )
    
    fig_trend.update_layout(
        xaxis_title="Date",
        yaxis_title="Beverage Sales (units)",
        height=360,
        template="plotly_white",
        margin=dict(l=50, r=20, t=20, b=40),
    )

    st.plotly_chart(fig_trend, key="trend", use_container_width=True)

    # Relative contributions
    ev1 = triple1["S"].cpu().numpy()**2

    contrib1 = 100 * ev1 / ev1.sum()
    
    trend_pct = contrib1[0]
    residual_pct = contrib1[1:].sum()
    
        # Relative contributions
    ev1 = triple1["S"].cpu().numpy()**2
    
    contrib1 = 100 * ev1 / ev1.sum()
    
    trend_pct = contrib1[0]
    residual_pct = contrib1[1:].sum()
    
    col1, col2 = st.columns([1.5, 1.5])

    with col1:
        st.metric(
            label="Trend Relative Contribution",
            value=f"{trend_pct:.2f}%"
        )
    
    with col2:
        st.metric(
            label="Residual Relative Contribution",
            value=f"{residual_pct:.2f}%"
        )

# ══════════════════════════════════════════════
# TAB 3 — Visual Inspection of Residual
# ══════════════════════════════════════════════
with tab3:
    st.markdown("""
        ## Stage 2: Visual inspection of the Residual Decomposition 
        """)

    col_a, col_b = st.columns(2)
     
    with st.spinner("Running SSA Stage 2"):
       ev2 = (triple2["S"].cpu().numpy() ** 2)
       U2 = triple2["U"].cpu().numpy()

    with col_a:
         # scree plot
        st.markdown('<div class="section-title">SSA Scree Plot — Eigenvalue Spectrum</div>',
                unsafe_allow_html=True,
                   )   
        x_scree = np.arange(1, min(51, len(ev2) + 1))
        y_scree = ev2[:50]
    
        fig_sc2 = go.Figure()
        fig_sc2.add_trace(go.Scatter(
            x=x_scree, y=y_scree, mode="lines+markers",
            line=dict(color="#2563eb", width=2),
            marker=dict(size=5), name=f"L={L2}",
        ))
        fig_sc2.update_layout(
            xaxis_title="Component Index", yaxis_title="Eigenvalue",
            yaxis_type="log", height=400, template="plotly_white",
            margin=dict(l=60, r=20, t=20, b=40),
        )
        st.plotly_chart(fig_sc2, key = "eigenvalues2", use_container_width=True)
        
    with col_b:
        # W-correlation heatmap 
        
        st.markdown('<div class="section-title">W-Correlation Matrix </div>',
                    unsafe_allow_html=True)
       
        n_components = 30

        ts_rec = [
            reconstruct_group(triple2, [i])
            for i in range(n_components)
        ]

        W = w_correlation_matrix(ts_rec, L2)

        labels = [str(i+1) for i in range(n_components)]
        fig_w = go.Figure(data=go.Heatmap(
            z=W, x= labels, y=labels,
            colorscale="Viridis", zmin=0, zmax=1,
            colorbar=dict(title="|W|"),
            hovertemplate="%{x} vs %{y}<br>|W| = %{z:.3f}<extra></extra>",
        ))
        fig_w.update_layout(
            height=420, template="plotly_white",
            yaxis=dict(autorange="reversed"),
            margin=dict(l=120, r=20, t=20, b=80),
        )
        st.plotly_chart(fig_w, key = "w_matrix", use_container_width=True)
        
    st.markdown(
    '<div class="info-box">'
    '<b>Scree plot:</b> Initial components (1–12) form pairs of nearly equal eigenvalues.<br><br>'
    '<b>W-correlation matrix:</b> Off-diagonal values close to 0 indicate that components '
    'are well separated. Values close to 1 suggest that the corresponding components '
    'should be merged into the same group.<br><br>'
    '<b>Eigenvectors:</b> Each eigenvector has length L, the window length used to construct the Hankel matrix. '
         ' Paired eigenvectors with similar frequencies represent the same oscillatory mode.'
        'Below, only the first 100 entries of each eigenvector are shown for clarity.'
    '</div>',
    unsafe_allow_html=True,
)

    # Eigenvectors: one-dimensional plots
   
    n_eig = 8
    
    fig_u2 = make_subplots(
        rows=2,
        cols=4,
        subplot_titles=[f"EV{i+1}" for i in range(n_eig)],
        vertical_spacing=0.15,
        horizontal_spacing=0.08,
    )
    
    for i in range(n_eig):
  
        row = i // 4 + 1
        col = i % 4 + 1
    
        fig_u2.add_trace(
            go.Scatter(
                x=np.arange(U2.shape[0]),
                y=U2[:100, i],
                mode="lines",
                line=dict(color="#2563EB", width=1.5),
                showlegend=False,
            ),
            row=row,
            col=col,
        )
    
    fig_u2.add_annotation(
    text="Embedding Index",
    x=0.5,
    y=-0.09,
    xref="paper",
    yref="paper",
    showarrow=False,
    font=dict(size=14)
)

    fig_u2.add_annotation(
    text="Amplitude",
    x=-0.06,
    y=0.5,
    xref="paper",
    yref="paper",
    textangle=-90,
    showarrow=False,
    font=dict(size=14)
)    
    
    fig_u2.update_layout(
        title="First Eight Eigenvectors (EV)",
        height=650,
        template="plotly_white",
        margin=dict(l=70, r=20, t=60, b=70),
    )
    
    st.plotly_chart(fig_u2, use_container_width=True)






# ══════════════════════════════════════════════
# TAB 4 — Pairs and Variance table 
# ══════════════════════════════════════════════
with tab4:
       
    # Pair plots
    U2 = triple2["U"].cpu().numpy()
    
    pairs = [(0,1), (2,3), (4,5), (6,7), (8,9), (10,11), (16,17),(18,19) ]
    
    fig_pairs = make_subplots(
        rows=2,
        cols=4,
        subplot_titles=[f"Pair ({i+1}, {j+1})" for i, j in pairs],
        horizontal_spacing=0.14,
        vertical_spacing=0.32,
    )
    
    for k, (i, j) in enumerate(pairs):
    
        row = k // 4 + 1
        col = k % 4 + 1
    
        fig_pairs.add_trace(
            go.Scatter(
                x=U2[:150, i],
                y=U2[:150, j],
                mode="lines",
                marker=dict(
                    size=4,
                    opacity=0.3,
                ),
                showlegend=False,
            ),
            row=row,
            col=col,
        )
    
        fig_pairs.update_xaxes(title_text=f"EV{i+1}", row=row, col=col)
        fig_pairs.update_yaxes(title_text=f"EV{j+1}", row=row, col=col)
    
    fig_pairs.update_layout(
        title="Eigenvector Pair Plots",
        height=550,
        template="plotly_white",
        margin=dict(l=40, r=20, t=60, b=40),
    )
    
    st.plotly_chart(fig_pairs, use_container_width=True)



    # Variance
    st.markdown('<div class="section-title">Variance Contribution by Group</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="info-box">'
        'The table below summarizes the selected groups based on the visual inspection of the scree plot, W-correlation matrix, elementary eigenvectors, and eigenvector pair plots. The reported contributions are relative to the total variance of the <b>detrended residual</b> (Stage 2 SSA). The trend alone accounts for ~93 % of the original series variance.</div>',
        unsafe_allow_html=True,
    )

    # Styled table
    def color_row(row):
        c = GROUP_COLORS.get(row["Group"], "#ffffff")
        return ["background-color: white; color: #111" for _ in row]

    st.dataframe(
        VARIANCE_TABLE.style.apply(color_row, axis=1).format({
            "Contribution (%)": "{:.2f}",
            "Cumulative (%)":   "{:.2f}",
            "Period (days)":    lambda x: f"{x:.1f}" if x else "—",
        }),
        use_container_width=True, height=310,
    )


                   
# ══════════════════════════════════════════════
# TAB 5 —  Weekly Oscillatory modes
# ══════════════════════════════════════════════
with tab5:
    st.markdown('<div class="section-title">Reconstructed Weekly and Sub-Weekly Sine-Waves</div>',
                unsafe_allow_html=True) 

    st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Weekly Modes</div>
            <div style="font-size:15px; line-height:1.5;">
            All reconstructed modes exhibit  modulated amplitudes.
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    osc_names = [k for k in GROUPS_OSC if k != "Noise"][:5]
    n_modes   = len(osc_names)
    cols_per_row = 2
    rows_needed  = (n_modes + 1) // cols_per_row

    for row_i in range(rows_needed):
        cols = st.columns(cols_per_row)
        for col_j, mode in enumerate(
            osc_names[row_i * cols_per_row: (row_i + 1) * cols_per_row]
        ):
            sig = grp_rec[mode]
            color = GROUP_COLORS[mode]

            fig_m = go.Figure()
            fig_m.add_trace(go.Scatter(
                x=dates, y=sig, mode="lines",
                line=dict(color=color, width=0.7), name=mode,
            ))
            
            fig_m.update_layout(
                title=dict(text=mode, font=dict(size=13)),
                xaxis_title="Date", yaxis_title="Amplitude",
                height=260, template="plotly_white",
                margin=dict(l=40, r=10, t=35, b=35),
                legend=dict(font=dict(size=10)),
            )
            cols[col_j].plotly_chart(fig_m, use_container_width=True)

    
     

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Weekly Mode</div>
            <div style="font-size:15px; line-height:1.5;">
            The <b>7-day</b> mode reflects the expected weekly sales cycle.
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">3.5-Day Modes</div>
            <div style="font-size:15px; line-height:1.5;">
            Two independent <b>3.5-day</b> modes share the same period but evolve
            differently over time. 
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">2.3-Day Modes</div>
            <div style="font-size:15px; line-height:1.5;">
            Two independent <b>2.3-day</b> modes exhibit distinct temporal behavior
            despite having the same period.
            </div>
        </div>
        """, unsafe_allow_html=True)    



        
# ══════════════════════════════════════════════
# TAB 6 — Longer Period modes and noise
# ══════════════════════════════════════════════
with tab6:
    st.markdown('<div class="section-title">Longer Period Modes and Noise</div>',
                unsafe_allow_html=True)

    st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Weekly Modes</div>
            <div style="font-size:15px; line-height:1.5;">
            All reconstructed series exhibit  modulated amplitudes.
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    osc_names = [k for k in GROUPS_OSC][5:]
    n_modes   = len(osc_names)
    cols_per_row = 2
    rows_needed  = (n_modes + 1) // cols_per_row

    for row_i in range(rows_needed):
        cols = st.columns(cols_per_row)
        for col_j, mode in enumerate(
            osc_names[row_i * cols_per_row: (row_i + 1) * cols_per_row]
        ):
            sig = grp_rec[mode]
            color = GROUP_COLORS[mode]

            fig_m = go.Figure()
            fig_m.add_trace(go.Scatter(
                x=dates, y=sig, mode="lines",
                line=dict(color=color, width=0.7), name=mode,
            ))
            
            
            fig_m.update_layout(
                title=dict(text=mode, font=dict(size=13)),
                xaxis_title="Date", yaxis_title="Amplitude",
                height=260, template="plotly_white",
                margin=dict(l=40, r=10, t=35, b=35),
                legend=dict(font=dict(size=10)),
            )
            cols[col_j].plotly_chart(fig_m, use_container_width=True)

    
    
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Longer Periods</div>
            <div style="font-size:15px; line-height:1.5;">
            Longer-period oscillations (T≈20, 15.3, and 11.7 days) contribute less variance than the weekly and sub-weekly modes but remain identifiable after SSA reconstruction.
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Noise</div>
            <div style="font-size:15px; line-height:1.5;">
            This group contains ungrouped components and residual variability, including possible weak non-isolated oscillatory contributions
            </div>
        </div>
        """, unsafe_allow_html=True)




# ══════════════════════════════════════════════
# TAB 7 — Reconstruction
# ══════════════════════════════════════════════

with tab7:
    st.markdown('<div class="section-title"> Reconstructed Series</div>',
                unsafe_allow_html=True)

    fig_rec = go.Figure()

    # Original
    fig_rec.add_trace(
        go.Scatter(
            x=dates,
            y=ts_np,
            mode="lines",
            name="Original",
            line=dict(color="#60A5FA", width=1),
        )
    )
    
    # Reconstruction
    fig_rec.add_trace(
        go.Scatter(
            x=dates,
            y=reconstructed,
            mode="lines",
            name="Reconstruction",
            line=dict(color="red", 
                      width=1,
                      dash="dash" ),
        )
    )
    
    fig_rec.update_layout(
        title="Final Reconstruction: : Trend + Osc. Modes + Noise",
        xaxis_title="Date",
        yaxis_title="Beverage Sales (units)",
        height=420,
        template="plotly_white",
    )
    
    st.plotly_chart(fig_rec, use_container_width=True)

   
    

    max_error = np.max(np.abs(ts_np - reconstructed))
    rmse = np.sqrt(np.mean((ts_np - reconstructed)**2))

    col1, col2 = st.columns([1, 1])
    with col1:
        st.metric(
            label="Max. error",
            value=f"{max_error:.6e}%"
        )
    
    with col2:
        st.metric(
            label="RMSE",
            value=f"{rmse:.6e}%"
        )
   


    st.markdown("""
           <div class="info-box">
    
           <h4 style="margin-top:0; margin-bottom:0.6rem;">Conclusions</h4>

           <ul>
                <li>The extracted trend reveals sustained growth in beverage sales throughout the study period, indicating consistent business growth.</li>
                <li>A clear 7-day oscillation was identified, naturally corresponding to the weekly sales cycle.</li>
                <li>Sub-weekly oscillations with approximate periods of 3.5 and 2.3 days contribute even more than the weekly cycle.</li>
                <li>Additional oscillatory modes with approximate periods of 20.5, 15.3, and 11.7 days were also identified. These longer-period patterns may reflect other recurring activities not explicitly represented in the available sales data.</li>
                <li>The dominant frequencies are represented by multiple eigenvector pairs rather than a single pair, reflecting amplitude modulation in the residual series. Moreover, groups sharing the same dominant period can exhibit markedly different temporal evolutions, indicating that similar frequencies do not necessarily correspond to the same oscillatory process.</li>
               <li>Overall, the two-stage SSA successfully separates long-term growth from multiple seasonal patterns, providing interpretable insights that can support business decision-making while revealing hidden temporal structures in the data.</li>
            <ul>
            </div>
            """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────
st.markdown("---")
st.caption(
    "Store Sales · SSA Analysis · Beverages · Store #1 · 2013–2017 · "
    "Kaggle Time Series Forecasting Dataset"
)
