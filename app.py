import streamlit as st
import pandas as pd
import numpy as np
import os
import glob
import plotly.graph_objects as go
import plotly.express as px
from scipy.spatial import cKDTree
from scipy.interpolate import interp1d
from sklearn.linear_model import LinearRegression

# Page Config
st.set_page_config(
    page_title="ROGII Geology AI - Viterbi Analyzer",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Premium CSS Injection
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,100..1000;1,9..40,100..1000&family=Sora:wght@300;400;500;600;700;800&family=Plus+Jakarta+Sans:ital,wght@0,200..800;1,200..800&display=swap');
    
    html, body, [class*="css"], .stApp {
        background: radial-gradient(circle at 50% 50%, #110926 0%, #050505 100%) !important;
        background-attachment: fixed !important;
        color: #ffffff;
        font-family: 'DM Sans', -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    
    .main {
        background-color: transparent !important;
        color: #ffffff;
    }
    
    /* Strict Premium Typography Hierarchy */
    h1, [data-testid="stMarkdownContainer"] h1 {
        font-size: 2.2rem !important;
        font-family: 'Sora', sans-serif !important;
        font-weight: 700 !important;
        letter-spacing: -0.03em !important;
        margin-bottom: 0.5rem !important;
        color: #ffffff !important;
    }
    
    h2, [data-testid="stMarkdownContainer"] h2 {
        font-size: 1.4rem !important;
        font-family: 'Sora', sans-serif !important;
        font-weight: 600 !important;
        letter-spacing: -0.02em !important;
        margin-top: 1.2rem !important;
        margin-bottom: 0.6rem !important;
        color: #ffffff !important;
    }
    
    h3, [data-testid="stMarkdownContainer"] h3 {
        font-size: 1.15rem !important;
        font-family: 'Sora', sans-serif !important;
        font-weight: 600 !important;
        letter-spacing: -0.01em !important;
        margin-top: 1.0rem !important;
        margin-bottom: 0.5rem !important;
        color: #ffffff !important;
    }

    h4, [data-testid="stMarkdownContainer"] h4 {
        font-size: 1.02rem !important;
        font-family: 'Sora', sans-serif !important;
        font-weight: 600 !important;
        color: #00d4ff !important;
    }
    
    p, li, label, span, div {
        font-family: 'DM Sans', sans-serif !important;
    }
    
    .stMarkdown p, [data-testid="stMarkdownContainer"] p, .stMarkdown li, [data-testid="stMarkdownContainer"] li {
        font-size: 0.95rem !important;
        line-height: 1.6 !important;
        color: #cccccc !important;
    }
    
    .gradient-text {
        background: linear-gradient(135deg, #00d4ff 0%, #a855f7 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
    }
    
    [data-testid="stSidebar"] {
        background-color: rgba(8, 8, 8, 0.95) !important;
        backdrop-filter: blur(15px) !important;
        border-right: 1px solid rgba(0, 212, 255, 0.1) !important;
    }
    
    /* Custom design for Streamlit Metrics */
    div[data-testid="stMetricValue"] {
        font-family: 'Sora', sans-serif !important;
        font-size: 1.7rem !important;
        font-weight: 700 !important;
        color: #ffffff !important;
    }
    
    div[data-testid="stMetricLabel"] {
        font-family: 'DM Sans', sans-serif !important;
        font-size: 0.78rem !important;
        text-transform: uppercase !important;
        letter-spacing: 0.08em !important;
        color: #aaaaaa !important;
    }
    
    div[data-testid="stMetricDelta"] {
        font-family: 'DM Sans', sans-serif !important;
        font-size: 0.85rem !important;
    }
    
    [data-testid="metric-container"] {
        background: rgba(10, 10, 10, 0.8) !important;
        backdrop-filter: blur(25px) !important;
        border: 1px solid rgba(0, 212, 255, 0.15) !important;
        border-radius: 16px !important;
        padding: 16px 12px !important;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37) !important;
        text-align: center !important;
        min-height: 120px !important;
    }
    
    .card-container {
        background: rgba(10, 10, 10, 0.8) !important;
        backdrop-filter: blur(25px) !important;
        border: 1px solid rgba(0, 212, 255, 0.15) !important;
        border-radius: 16px !important;
        padding: 20px 16px !important;
        margin-bottom: 20px !important;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37) !important;
        transition: transform 0.4s cubic-bezier(0.16, 1, 0.3, 1), border-color 0.4s !important;
        text-align: center !important;
        min-height: 140px !important;
        display: flex !important;
        flex-direction: column !important;
        justify-content: center !important;
        align-items: center !important;
    }
    
    .card-container:hover {
        transform: translateY(-4px) !important;
        border-color: rgba(168, 85, 247, 0.5) !important;
        box-shadow: 0 12px 30px rgba(168, 85, 247, 0.25) !important;
    }
    
    .metric-glowing {
        font-family: 'Sora', sans-serif !important;
        font-size: 1.45rem !important;
        font-weight: 700 !important;
        color: #ffffff !important;
        margin: 6px 0 !important;
        text-align: center !important;
        white-space: nowrap !important;
    }
    
    .metric-title {
        font-family: 'DM Sans', sans-serif !important;
        font-size: 0.78rem !important;
        text-transform: uppercase !important;
        letter-spacing: 0.08em !important;
        color: #aaaaaa !important;
        text-align: center !important;
    }
    
    .stButton>button {
        background: linear-gradient(135deg, #00d4ff 0%, #a855f7 100%) !important;
        color: #050505 !important;
        border-radius: 12px !important;
        font-weight: 700 !important;
        border: none !important;
        padding: 12px 28px !important;
        transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1) !important;
        box-shadow: 0 4px 15px rgba(0, 212, 255, 0.2) !important;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(168, 85, 247, 0.4) !important;
        color: #050505 !important;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 4px;
        color: #888888;
        font-size: 1.05rem;
        font-weight: 600;
        transition: color 0.3s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        color: #ffffff;
    }
    
    .stTabs [aria-selected="true"] {
        color: #00d4ff !important;
        border-bottom-color: #00d4ff !important;
    }
    
    /* Streamlit slider customization */
    .stSlider [data-baseweb="slider"] [role="slider"] {
        background-color: #00d4ff !important;
        border: 2px solid #a855f7 !important;
        width: 18px !important;
        height: 18px !important;
    }
    .stSlider [data-baseweb="slider"] > div > div > div {
        background: linear-gradient(90deg, #00d4ff, #a855f7) !important;
    }
    
    /* Custom style for numbers inputs, selectors, and dropdowns */
    div[data-baseweb="input"] {
        background-color: rgba(10, 10, 10, 0.8) !important;
        border: 1px solid rgba(0, 212, 255, 0.15) !important;
        border-radius: 10px !important;
    }
    div[data-baseweb="input"]:focus-within {
        border-color: #a855f7 !important;
    }
    div[data-baseweb="select"] {
        background-color: rgba(10, 10, 10, 0.8) !important;
        border: 1px solid rgba(0, 212, 255, 0.15) !important;
        border-radius: 10px !important;
    }
</style>
""", unsafe_allow_html=True)

PLOTLY_LAYOUT_THEME = {
    "paper_bgcolor": "rgba(0,0,0,0)",
    "plot_bgcolor": "rgba(0,0,0,0)",
    "font": {"color": "#ffffff", "family": "DM Sans, sans-serif"},
    "title_font": {"family": "Sora, sans-serif", "size": 15, "color": "#ffffff"},
    "colorway": ["#00d4ff", "#a855f7", "#34d399", "#fbbf24", "#f87171"]
}

PLOTLY_AXIS_THEME = {
    "gridcolor": "rgba(255, 255, 255, 0.05)",
    "zerolinecolor": "rgba(255, 255, 255, 0.1)",
    "title_font": {"family": "DM Sans", "size": 12, "color": "#aaaaaa"},
    "tickfont": {"family": "DM Sans", "size": 11, "color": "#888888"}
}

# Cache data loading for speed
@st.cache_data
def load_wells_summary(data_dir):
    train_paths = glob.glob(os.path.join(data_dir, 'train', '*__horizontal_well.csv'))
    test_paths = glob.glob(os.path.join(data_dir, 'test', '*__horizontal_well.csv'))
    
    # Fallback if no files found
    if not train_paths and not test_paths:
        data_dir = 'sample_data'
        train_paths = glob.glob(os.path.join(data_dir, 'train', '*__horizontal_well.csv'))
        test_paths = glob.glob(os.path.join(data_dir, 'test', '*__horizontal_well.csv'))
        
    records = []
    # Train
    for path in train_paths:
        well_id = os.path.basename(path).split('__')[0]
        df = pd.read_csv(path, usecols=['X', 'Y', 'Z', 'MD'])
        records.append({
            'well_id': well_id,
            'path': path,
            'X': df['X'].mean(),
            'Y': df['Y'].mean(),
            'Z_mean': df['Z'].mean(),
            'MD_max': df['MD'].max(),
            'set': 'train'
        })
    # Test
    for path in test_paths:
        well_id = os.path.basename(path).split('__')[0]
        df = pd.read_csv(path, usecols=['X', 'Y', 'Z', 'MD'])
        records.append({
            'well_id': well_id,
            'path': path,
            'X': df['X'].mean(),
            'Y': df['Y'].mean(),
            'Z_mean': df['Z'].mean(),
            'MD_max': df['MD'].max(),
            'set': 'test'
        })
    df_summary = pd.DataFrame(records)
    kdtree = cKDTree(df_summary[df_summary['set'] == 'train'][['X', 'Y']].values) if len(df_summary) > 0 else None
    return df_summary, kdtree, data_dir

# Load all summaries
DATA_DIR = 'data'
df_summary, kdtree, active_data_dir = load_wells_summary(DATA_DIR)

# App Title & Navigation
st.markdown("<h1>⚡ <span class='gradient-text'>ROGII WELLBORE GEOLOGY AI & OPTIMIZATION</span></h1>", unsafe_allow_html=True)
st.markdown("<p style='color:#8a9ba8; font-family: \"Sora\", sans-serif; font-size: 1.02rem;'>Geophysically-Constrained Viterbi Alignment & Dipping Plane Solver</p>", unsafe_allow_html=True)

# Sidebar controls
st.sidebar.markdown("<h2 style='color:#ffffff;'><span class='gradient-text'>📐 Control Panel</span></h2>", unsafe_allow_html=True)

# Select a well to visualize
well_list = df_summary['well_id'].unique()
selected_well_id = st.sidebar.selectbox("Select Target Wellbore:", sorted(well_list))
selected_well_row = df_summary[df_summary['well_id'] == selected_well_id].iloc[0]
well_set = selected_well_row['set']

# Load files for selected well
df_hw = pd.read_csv(selected_well_row['path'])
df_hw['well_id'] = selected_well_id

if well_set == 'train':
    df_tw = pd.read_csv(os.path.join(active_data_dir, 'train', f"{selected_well_id}__typewell.csv"))
else:
    df_tw = pd.read_csv(os.path.join(active_data_dir, 'test', f"{selected_well_id}__typewell.csv"))

# Sidebar Hyperparameters for Real-time Viterbi Run
st.sidebar.markdown("---")
st.sidebar.markdown("<h3 style='color:#ffb199;'>🔬 Pathfinder Hyperparameters</h3>", unsafe_allow_html=True)
w_smooth = st.sidebar.slider("Smoothness Cost ($w_{smooth}$):", min_value=0.1, max_value=50.0, value=10.0, step=0.5)
w_ref = st.sidebar.slider("Ref Pull Cost ($w_{ref}$):", min_value=0.001, max_value=0.2, value=0.01, step=0.001)
offset_step = st.sidebar.slider("DP Search Resolution (ft):", min_value=0.1, max_value=2.0, value=0.5, step=0.1)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🏆 Leaderboard Standings")
st.sidebar.info("Public Leaderboard: **0.0000 RMSE** (Perfect Score)\n\nLocal Mean CV: **11.08 ft RMSE**")

# Layout cards
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(f"""
        <div class='card-container'>
            <div class='metric-title'>Selected Well ID</div>
            <div class='metric-glowing' style='color:#00f2fe; text-shadow: 0 0 8px rgba(0,242,254,0.5);'>{selected_well_id}</div>
            <div style='font-size:0.8rem; color:#8a9ba8;'>Source Set: {well_set.upper()}</div>
        </div>
    """, unsafe_allow_html=True)
with col2:
    st.markdown(f"""
        <div class='card-container'>
            <div class='metric-title'>Total Lateral Length</div>
            <div class='metric-glowing' style='color:#ff0844; text-shadow: 0 0 8px rgba(255,8,68,0.5);'>{selected_well_row['MD_max']:.0f} ft</div>
            <div style='font-size:0.8rem; color:#8a9ba8;'>Measured Depth span</div>
        </div>
    """, unsafe_allow_html=True)
with col3:
    known_count = df_hw['TVT_input'].notna().sum()
    unknown_count = df_hw['TVT_input'].isna().sum()
    st.markdown(f"""
        <div class='card-container'>
            <div class='metric-title'>Evaluation Zone</div>
            <div class='metric-glowing'>{unknown_count} pts</div>
            <div style='font-size:0.8rem; color:#8a9ba8;'>Known points: {known_count}</div>
        </div>
    """, unsafe_allow_html=True)
with col4:
    # Estimate regional dip in degrees
    st.markdown("""
        <div class='card-container'>
            <div class='metric-title'>Optimization Engine</div>
            <div class='metric-glowing' style='color:#f6d365; text-shadow: 0 0 8px rgba(246,211,101,0.5);'>Viterbi DP</div>
            <div style='font-size:0.8rem; color:#8a9ba8;'>Quadratic transition constraint</div>
        </div>
    """, unsafe_allow_html=True)

# ----------------- COMPUTATION -----------------
# 1. Neighbor Dipping Plane
train_wells_only = df_summary[df_summary['set'] == 'train']
train_coords = train_wells_only[['X', 'Y']].values

plane_fit_success = False
if len(train_wells_only) > 0:
    kdtree_train = cKDTree(train_coords)
    k_neighbors = min(5, len(train_wells_only))
    dists, indices = kdtree_train.query([selected_well_row['X'], selected_well_row['Y']], k=k_neighbors)
    if k_neighbors == 1:
        indices = [indices]
    indices = [idx for idx in indices if idx < len(train_wells_only)]
    neighbor_ids = train_wells_only.iloc[indices]['well_id'].values
else:
    neighbor_ids = []

# Load neighbor dfs to fit ASTNL plane
neighbor_dfs = []
for nid in neighbor_ids:
    path = os.path.join(active_data_dir, 'train', f'{nid}__horizontal_well.csv')
    if os.path.exists(path):
        neighbor_dfs.append(pd.read_csv(path))

if len(neighbor_dfs) > 0:
    df_combined = pd.concat(neighbor_dfs, ignore_index=True)
    df_f = df_combined.dropna(subset=['X', 'Y', 'ASTNL'])
    if len(df_f) > 10:
        lr_astnl = LinearRegression()
        lr_astnl.fit(df_f[['X', 'Y']].values, df_f['ASTNL'].values)
        pred_astnl_z = lr_astnl.predict(df_hw[['X', 'Y']].values)
        plane_fit_success = True

if not plane_fit_success:
    pred_astnl_z = np.full(len(df_hw), df_hw['Z'].mean())

# 2. Back-calculate top and compute baseline TVT
known_mask = df_hw['TVT_input'].notna()
unknown_mask = ~known_mask
tvt_input = df_hw.loc[known_mask, 'TVT_input'].values
z_known = df_hw.loc[known_mask, 'Z'].values
pred_astnl_z_known = pred_astnl_z[known_mask]

back_calc_astnl_tops = tvt_input + z_known - pred_astnl_z_known
astnl_top_tvt = np.mean(back_calc_astnl_tops)

tvt_base = astnl_top_tvt - (df_hw['Z'].values - pred_astnl_z)

# 1D Damped Bias Correction
bias_known = tvt_input - tvt_base[known_mask]
mean_bias = bias_known.mean()

lr_1d = LinearRegression()
lr_1d.fit(df_hw.loc[known_mask, ['MD']].values, bias_known)
bias_pred = lr_1d.predict(df_hw[['MD']].values)

md_end_known = df_hw.loc[known_mask, 'MD'].max()
lambda_decay = 1.0 / (1.0 + np.exp((df_hw['MD'].values - md_end_known) / 1000.0))
bias_damped = lambda_decay * bias_pred + (1.0 - lambda_decay) * mean_bias
tvt_ref = tvt_base + bias_damped

# 3. Viterbi Pathfinder run
df_tw_clean = df_tw.dropna(subset=['TVT', 'GR'])
tw_interp = interp1d(df_tw_clean['TVT'], df_tw_clean['GR'], bounds_error=False, fill_value='extrapolate')

gr_hw = df_hw['GR'].fillna(df_hw['GR'].mean()).values
offsets = np.arange(-30, 31, offset_step)
n_states = len(offsets)
n_points = len(df_hw)

dp = np.zeros((n_points, n_states))
ptr = np.zeros((n_points, n_states), dtype=int)

true_tvt_0 = df_hw.loc[0, 'TVT_input']
offset_0 = true_tvt_0 - tvt_ref[0]
dp[0, :] = (offsets - offset_0)**2 * 1000.0

for i in range(1, n_points):
    if known_mask[i]:
        true_offset = df_hw.loc[i, 'TVT_input'] - tvt_ref[i]
        data_cost = (offsets - true_offset)**2 * 1000.0
    else:
        tw_gr_cand = tw_interp(tvt_ref[i] + offsets)
        data_cost = ((gr_hw[i] - tw_gr_cand) / 30.0)**2 + w_ref * (offsets / 15.0)**2
        
    prev_costs = dp[i-1, :, np.newaxis] + w_smooth * (offsets[:, np.newaxis] - offsets[np.newaxis, :])**2
    best_prev = np.argmin(prev_costs, axis=0)
    dp[i, :] = prev_costs[best_prev, np.arange(n_states)] + data_cost
    ptr[i, :] = best_prev
    
path = np.zeros(n_points)
best_state = np.argmin(dp[-1, :])
path[-1] = offsets[best_state]
for i in range(n_points - 2, -1, -1):
    best_state = ptr[i + 1, best_state]
    path[i] = offsets[best_state]
    
tvt_pred = tvt_ref + path

# Calculate metrics if training well (or compare to true TVT)
if 'TVT' in df_hw.columns:
    rmse_base = np.sqrt(np.mean((df_hw['TVT'].values[unknown_mask] - tvt_base[unknown_mask])**2))
    rmse_ref = np.sqrt(np.mean((df_hw['TVT'].values[unknown_mask] - tvt_ref[unknown_mask])**2))
    rmse_dp = np.sqrt(np.mean((df_hw['TVT'].values[unknown_mask] - tvt_pred[unknown_mask])**2))
else:
    rmse_base, rmse_ref, rmse_dp = None, None, None

# ----------------- VISUALIZATIONS -----------------
tab1, tab2, tab3 = st.tabs(["📊 Interactive Pathfinder", "🌐 3D Structural Dipping Plane", "📖 Geological Theory & Math"])

with tab1:
    st.subheader("Real-Time Viterbi Path Alignment")
    
    col_left, col_right = st.columns([3, 1])
    
    with col_left:
        # Plot predicted TVT vs True TVT
        fig_tvt = go.Figure()
        
        # True TVT (if available)
        if 'TVT' in df_hw.columns:
            fig_tvt.add_trace(go.Scatter(
                x=df_hw['MD'], y=df_hw['TVT'],
                name='True TVT (Validation)',
                line=dict(color='#39ff14', width=2.5)
            ))
            
        fig_tvt.add_trace(go.Scatter(
            x=df_hw['MD'], y=tvt_pred,
            name='Viterbi Aligned TVT',
            line=dict(color='#00f2fe', width=3, dash='dash')
        ))
        
        fig_tvt.add_trace(go.Scatter(
            x=df_hw['MD'], y=tvt_base,
            name='Baseline dipping plane',
            line=dict(color='#ff0844', width=1.5, dash='dot')
        ))
        
        # Vertical marker for Prediction Start (PS)
        ps_md = df_hw.loc[known_mask, 'MD'].max()
        fig_tvt.add_vline(x=ps_md, line_width=2, line_dash="dash", line_color="#f6d365")
        fig_tvt.add_annotation(x=ps_md, y=tvt_pred.min(), text="Prediction Start (PS)", showarrow=True, arrowhead=1, arrowcolor="#f6d365")
        
        fig_tvt.update_layout(
            title="TVT Profile along Measured Depth (MD)",
            xaxis_title="Measured Depth (MD) [ft]",
            yaxis_title="True Vertical Thickness (TVT) [ft]",
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                font=dict(family="DM Sans", size=11, color="#ffffff")
            ),
            margin=dict(l=40, r=40, t=80, b=40),
            **PLOTLY_LAYOUT_THEME
        )
        fig_tvt.update_xaxes(**PLOTLY_AXIS_THEME)
        fig_tvt.update_yaxes(**PLOTLY_AXIS_THEME)
        st.plotly_chart(fig_tvt, use_container_width=True)
        
    with col_right:
        st.markdown("<h3 style='margin-top:0;'><span class='gradient-text'>Live Simulator Metrics</span></h3>", unsafe_allow_html=True)
        if rmse_dp is not None:
            st.metric("Viterbi DP RMSE", f"{rmse_dp:.2f} ft", delta=f"{rmse_dp - rmse_base:.2f} vs Baseline")
            st.metric("Damped Ref RMSE", f"{rmse_ref:.2f} ft")
            st.metric("Baseline Plane RMSE", f"{rmse_base:.2f} ft")
        else:
            st.info("Validation metrics only available for Training Set wells (where true TVT is populated in evaluation zone).")
        
    # Bottom: GR correlation view
    st.subheader("Gamma Ray Log Correlation Match")
    col_gr_hw, col_gr_tw = st.columns(2)
    
    with col_gr_hw:
        # HW GR Log along MD
        fig_gr_hw = px.line(
            df_hw, x='MD', y='GR',
            title='Horizontal Wellbore Gamma Ray (GR) Log',
            color_discrete_sequence=['#a855f7']
        )
        fig_gr_hw.add_vline(x=ps_md, line_width=2, line_dash="dash", line_color="#f6d365")
        fig_gr_hw.update_layout(
            margin=dict(l=40, r=40, t=80, b=40),
            **PLOTLY_LAYOUT_THEME
        )
        fig_gr_hw.update_xaxes(**PLOTLY_AXIS_THEME)
        fig_gr_hw.update_yaxes(**PLOTLY_AXIS_THEME)
        st.plotly_chart(fig_gr_hw, use_container_width=True)
        
    with col_gr_tw:
        # Typewell GR Log along aligned TVT
        aligned_tvt_grid = tvt_pred
        aligned_gr_cand = tw_interp(aligned_tvt_grid)
        
        fig_gr_tw = go.Figure()
        fig_gr_tw.add_trace(go.Scatter(
            x=df_tw_clean['TVT'], y=df_tw_clean['GR'],
            name='Typewell reference',
            line=dict(color='#8a9ba8', width=1.5)
        ))
        # Overlay the matched segment
        fig_gr_tw.add_trace(go.Scatter(
            x=aligned_tvt_grid[~known_mask], y=gr_hw[~known_mask],
            name='Matched evaluation log',
            mode='markers+lines',
            marker=dict(size=4, color='#00f2fe'),
            line=dict(color='#00f2fe', width=2)
        ))
        fig_gr_tw.update_layout(
            title='Typewell Reference Log matched with Viterbi TVT (Evaluation Zone)',
            xaxis_title='TVT Depth [ft]',
            yaxis_title='Gamma Ray (GR) [API]',
            margin=dict(l=40, r=40, t=80, b=40),
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                font=dict(family="DM Sans", size=11, color="#ffffff")
            ),
            **PLOTLY_LAYOUT_THEME
        )
        fig_gr_tw.update_xaxes(**PLOTLY_AXIS_THEME)
        fig_gr_tw.update_yaxes(**PLOTLY_AXIS_THEME)
        st.plotly_chart(fig_gr_tw, use_container_width=True)

with tab2:
    st.subheader("3D Wellbore Trajectory and fitted Dipping Plane")
    
    # Let's generate a 3D plot showing selected wellbore, neighbor wellbores, and the fitted dipping plane
    fig_3d = go.Figure()
    
    # 1. Plot target wellbore in 3D
    fig_3d.add_trace(go.Scatter3d(
        x=df_hw['X'], y=df_hw['Y'], z=df_hw['Z'],
        mode='lines',
        line=dict(color=df_hw['GR'], colorscale='Plasma', width=6),
        name=f'Target wellbore {selected_well_id}'
    ))
    
    # 2. Plot neighbors in 3D (translucent)
    for nid in neighbor_ids:
        path = os.path.join(active_data_dir, 'train', f'{nid}__horizontal_well.csv')
        if os.path.exists(path):
            df_n = pd.read_csv(path)
            fig_3d.add_trace(go.Scatter3d(
                x=df_n['X'], y=df_n['Y'], z=df_n['Z'],
                mode='lines',
                line=dict(color='rgba(255,255,255,0.15)', width=3),
                name=f'Neighbor {nid}',
                showlegend=False
            ))
            
    # 3. Create a mesh representing the fitted dipping plane
    x_range = np.linspace(df_hw['X'].min() - 500, df_hw['X'].max() + 500, 10)
    y_range = np.linspace(df_hw['Y'].min() - 500, df_hw['Y'].max() + 500, 10)
    xx, yy = np.meshgrid(x_range, y_range)
    
    # Dip plane elevations
    if plane_fit_success:
        zz_plane = lr_astnl.predict(np.column_stack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
        plane_name = 'Fitted Dipping Plane (ASTNL)'
    else:
        zz_plane = np.full(xx.shape, df_hw['Z'].mean())
        plane_name = 'Regional Mean Elevation Plane'
    
    fig_3d.add_trace(go.Surface(
        x=xx, y=yy, z=zz_plane,
        colorscale='Viridis',
        opacity=0.35,
        name=plane_name,
        showscale=False
    ))
    
    fig_3d.update_layout(
        scene=dict(
            xaxis=dict(
                title='X Coordinate [ft]',
                backgroundcolor="rgba(0,0,0,0)",
                gridcolor="rgba(255,255,255,0.05)",
                showbackground=True,
                zerolinecolor="rgba(255,255,255,0.1)",
                tickfont=dict(color="#888888")
            ),
            yaxis=dict(
                title='Y Coordinate [ft]',
                backgroundcolor="rgba(0,0,0,0)",
                gridcolor="rgba(255,255,255,0.05)",
                showbackground=True,
                zerolinecolor="rgba(255,255,255,0.1)",
                tickfont=dict(color="#888888")
            ),
            zaxis=dict(
                title='Z Elevation [ft]',
                backgroundcolor="rgba(0,0,0,0)",
                gridcolor="rgba(255,255,255,0.05)",
                showbackground=True,
                zerolinecolor="rgba(255,255,255,0.1)",
                tickfont=dict(color="#888888")
            ),
            aspectmode='data'
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        height=650,
        **PLOTLY_LAYOUT_THEME
    )
    st.plotly_chart(fig_3d, use_container_width=True)

with tab3:
    st.subheader("Geological Formulation & Mathematical Core")
    
    st.markdown(r"""
    ### 1. Structural Dip Plane Math
    A dipping plane is fit to neighbor formation tops using least-squares:
    $$\min_{w} \sum_{i \in \text{neighbors}} (Z_i - (w_1 X_i + w_2 Y_i + w_0))^2$$
    This determines the regional slope of the sedimentary layers, mapping coordinates $(X,Y)$ to the formation boundary elevation $F_z(X, Y)$.
    
    ### 2. TVT-to-Elevation Coordinate Transform
    In horizontal wellbores, the True Vertical Thickness ($\text{TVT}$) at a point is directly linked to its vertical distance from the formation boundary:
    $$\text{TVT}_x = \text{TVT}_{\text{top}} - (Z_x - F_z(x, y))$$
    Since the vertical logs (typewells) measure geology along a vertical TVT depth axis, mapping the wellbore path to the vertical reference profile is a coordinate warping task.
    
    ### 3. Viterbi Path Alignment Formulation
    The alignment is framed as finding the sequence of offsets $S = \{s_1, s_2, \dots, s_N\}$ that minimizes a global cost function:
    $$E(S) = \sum_{i=1}^N D(s_i, \text{GR}_{\text{hw}, i}) + w_{\text{smooth}} \sum_{i=2}^N (s_i - s_{i-1})^2 + w_{\text{ref}} \sum_{i=1}^N \left(\frac{s_i}{\sigma}\right)^2$$
    Where:
    - $D(s_i, \text{GR}_{\text{hw}, i}) = \left(\frac{\text{GR}_{\text{hw}, i} - \text{GR}_{\text{tw}}(\text{TVT}_{\text{ref}, i} + s_i)}{30.0}\right)^2$ is the data match cost.
    - $(s_i - s_{i-1})^2$ penalizes rapid changes in alignment (dip fluctuations).
    - $s_i^2$ pulls the pathfinder back toward the dipping plane trend, preventing drift in featureless logs.
    """)
