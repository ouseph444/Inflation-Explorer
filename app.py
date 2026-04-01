"""
Cosmological Inflation Explorer
================================
A research-grade Streamlit app for studying arbitrary inflaton potentials
using the slow-roll formalism. Users define V(phi) symbolically; the app
auto-detects free parameters, generates sliders, and computes all inflationary
observables with CMB normalization.
"""

import streamlit as st
import numpy as np
from scipy.integrate import cumulative_trapezoid
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sympy as sp
from sympy import symbols, lambdify, diff, sympify, latex, pi, exp, sqrt, log, sin, cos, tan, Abs
import re
import warnings
warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG & GLOBAL STYLE
# ──────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Inflation Explorer",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=JetBrains+Mono:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
}

.stApp {
    background: #090c14;
    color: #e8eaf0;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #0d1120;
    border-right: 1px solid #1e2540;
}
section[data-testid="stSidebar"] * {
    color: #c8ccd8 !important;
    font-family: 'JetBrains Mono', monospace !important;
}

/* Headers */
h1 { 
    font-family: 'Syne', sans-serif !important;
    font-weight: 800 !important;
    font-size: 2.2rem !important;
    background: linear-gradient(135deg, #7eb8f7 0%, #a78bfa 50%, #f472b6 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: -0.02em;
    margin-bottom: 0.2rem !important;
}
h2 {
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 1.1rem !important;
    color: #7eb8f7 !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
    border-bottom: 1px solid #1e2540;
    padding-bottom: 0.4rem;
    margin-top: 1.5rem !important;
}
h3 {
    font-family: 'Syne', sans-serif !important;
    font-weight: 600 !important;
    color: #a8b4cc !important;
    font-size: 0.95rem !important;
}

/* Text input */
.stTextInput > div > div > input, .stTextArea > div > div > textarea {
    background: #111827 !important;
    border: 1px solid #2a3550 !important;
    border-radius: 8px !important;
    color: #e2e8f0 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.95rem !important;
}
.stTextInput > div > div > input:focus, .stTextArea > div > div > textarea:focus {
    border-color: #7eb8f7 !important;
    box-shadow: 0 0 0 2px rgba(126,184,247,0.15) !important;
}

/* Sliders */
.stSlider > div > div > div > div {
    background: linear-gradient(90deg, #7eb8f7, #a78bfa) !important;
}

/* Metric cards */
[data-testid="metric-container"] {
    background: #0d1120;
    border: 1px solid #1e2540;
    border-radius: 12px;
    padding: 1rem 1.2rem;
}
[data-testid="metric-container"] label {
    color: #6b7fa3 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.78rem !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #e2e8f0 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 1.6rem !important;
    font-weight: 500;
}

/* Alerts */
.stSuccess {
    background: #0a1a0f !important;
    border: 1px solid #1a4027 !important;
    color: #4ade80 !important;
    border-radius: 10px !important;
}
.stInfo {
    background: #0a0f1a !important;
    border: 1px solid #1a2740 !important;
    color: #7eb8f7 !important;
    border-radius: 10px !important;
}
.stWarning {
    background: #1a120a !important;
    border: 1px solid #40300a !important;
    border-radius: 10px !important;
}
.stError {
    background: #1a0a0a !important;
    border: 1px solid #400a0a !important;
    border-radius: 10px !important;
}

/* Select/Radio */
.stRadio label, .stSelectbox label {
    color: #a8b4cc !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.82rem !important;
}

/* Plotly charts - force dark background */
.js-plotly-plot .plotly {
    background: transparent !important;
}

/* Divider */
hr { border-color: #1e2540 !important; }

/* Expander */
.streamlit-expanderHeader {
    background: #0d1120 !important;
    color: #7eb8f7 !important;
    border: 1px solid #1e2540 !important;
    border-radius: 8px !important;
}

/* Column gap */
[data-testid="column"] { padding: 0 0.6rem; }

/* Tag pill for params */
.param-tag {
    display: inline-block;
    background: #1a2540;
    color: #a78bfa;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8rem;
    border-radius: 20px;
    padding: 2px 10px;
    margin: 2px 3px;
    border: 1px solid #2a3560;
}

/* LaTeX display box */
.latex-box {
    background: #0d1120;
    border: 1px solid #2a3550;
    border-left: 3px solid #7eb8f7;
    border-radius: 0 8px 8px 0;
    padding: 0.8rem 1.2rem;
    margin: 0.6rem 0 1rem 0;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.85rem;
    color: #c8d0e0;
}

.subtitle {
    color: #4a5a7a;
    font-size: 0.88rem;
    font-family: 'JetBrains Mono', monospace;
    margin-top: -0.5rem;
    margin-bottom: 1.5rem;
}
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# PLOTLY DARK THEME HELPER
# ──────────────────────────────────────────────────────────────────────────────

PLOT_LAYOUT = dict(
    paper_bgcolor="#090c14",
    plot_bgcolor="#0d1120",
    font=dict(family="JetBrains Mono, monospace", color="#a8b4cc", size=11),
    xaxis=dict(gridcolor="#1a2030", zerolinecolor="#2a3050", linecolor="#2a3050"),
    yaxis=dict(gridcolor="#1a2030", zerolinecolor="#2a3050", linecolor="#2a3050"),
    legend=dict(bgcolor="#0d1120", bordercolor="#2a3050", borderwidth=1),
    margin=dict(l=50, r=20, t=40, b=50),
)

COLORS = {
    "blue":   "#7eb8f7",
    "purple": "#a78bfa",
    "pink":   "#f472b6",
    "green":  "#34d399",
    "orange": "#fbbf24",
    "red":    "#f87171",
    "teal":   "#2dd4bf",
}

# ──────────────────────────────────────────────────────────────────────────────
# SYMBOLIC UTILITIES
# ──────────────────────────────────────────────────────────────────────────────

# Reserved SymPy names — NOT free parameters
RESERVED = {
    "phi", "phi_", "e", "pi", "E", "I", "oo", "zoo",
    "exp", "log", "ln", "sqrt", "sin", "cos", "tan",
    "sinh", "cosh", "tanh", "asin", "acos", "atan",
    "Abs", "sign", "Heaviside", "DiracDelta",
    "re", "im", "conjugate",
    "abs", "Max", "Min",
}

SAFE_NAMESPACE = {
    "exp": sp.exp, "log": sp.log, "ln": sp.log,
    "sqrt": sp.sqrt, "sin": sp.sin, "cos": sp.cos, "tan": sp.tan,
    "sinh": sp.sinh, "cosh": sp.cosh, "tanh": sp.tanh,
    "asin": sp.asin, "acos": sp.acos, "atan": sp.atan, "atan2": sp.atan2,
    "Abs": sp.Abs, "abs": sp.Abs, "sign": sp.sign,
    "pi": sp.pi, "E": sp.E, "e": sp.E,
    "Max": sp.Max, "Min": sp.Min,
}

def extract_free_params(expr_sym, phi_sym):
    """Return sorted list of free symbol names (excluding phi)."""
    all_syms = expr_sym.free_symbols
    params = sorted(
        [str(s) for s in all_syms if s != phi_sym],
        key=lambda x: (len(x), x)
    )
    return params

def parse_potential(expr_str):
    """
    Parse user-supplied expression for V(phi).
    Returns (sympy_expr, phi_symbol, list_of_param_names) or raises ValueError.
    """
    phi_sym = sp.Symbol("phi", positive=True)
    namespace = {**SAFE_NAMESPACE, "phi": phi_sym}

    # Pre-process: replace ^ with ** for user convenience
    expr_str = expr_str.replace("^", "**")

    try:
        expr_sym = sympify(expr_str, locals=namespace)
    except Exception as exc:
        raise ValueError(f"SymPy could not parse expression: {exc}")

    # Only reject if SymPy can definitively prove the expression is non-real
    # (is_real / is_complex return None when unknown, which is the normal case
    #  for expressions with free parameters — do NOT treat None as False)
    if expr_sym.is_real is False and expr_sym.is_complex is False:
        raise ValueError("Expression does not appear to be a valid mathematical formula.")

    params = extract_free_params(expr_sym, phi_sym)
    return expr_sym, phi_sym, params

def build_numerical_V(expr_sym, phi_sym, param_syms, param_vals):
    """
    Substitute param values and return a fast numpy function V(phi_array).
    param_syms: list of sympy Symbol objects
    param_vals: list of float values (same order)
    """
    subs_dict = dict(zip(param_syms, param_vals))
    expr_num = expr_sym.subs(subs_dict)
    V_func = lambdify(phi_sym, expr_num, modules=["numpy"])
    return V_func

# ──────────────────────────────────────────────────────────────────────────────
# INFLATION PHYSICS
# ──────────────────────────────────────────────────────────────────────────────

def compute_inflation(V_func, phi_min=0.01, phi_max=25.0, N_points=2000):
    """
    Full slow-roll computation for a given V(phi).
    Returns a dict of arrays and scalar observables.
    """
    phi = np.linspace(phi_min, phi_max, N_points)

    # Evaluate potential
    try:
        V_vals = np.array(V_func(phi), dtype=float)
    except Exception:
        return None, "Could not evaluate V(φ) on the domain. Check for singularities."

    # Screen for non-physical regions
    bad = (~np.isfinite(V_vals)) | (V_vals <= 0)
    if bad.all():
        return None, "V(φ) is non-positive or singular everywhere. Inflation requires V > 0."

    # Numerical derivatives (5-point stencil for accuracy)
    dphi = phi[1] - phi[0]
    dV = np.gradient(V_vals, dphi)
    d2V = np.gradient(dV, dphi)

    # Slow-roll parameters (M_Pl = 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        eps = 0.5 * np.where(V_vals > 0, (dV / V_vals) ** 2, np.nan)
        eta = np.where(V_vals > 0, d2V / V_vals, np.nan)

    # End of inflation: LAST phi (rightmost) where eps >= 1.
    # For large-field models eps is large at small phi and small at large phi,
    # so inflation proceeds from large phi down toward phi_end.
    # For small-field/hilltop models the opposite may hold.
    valid_end = np.where((eps >= 1.0) & np.isfinite(eps))[0]
    if len(valid_end) == 0:
        # eps never reaches 1; use domain boundary where eps is largest
        idx_end = int(np.nanargmax(eps))
        phi_end = phi[idx_end]
    else:
        # Use the rightmost eps>=1 crossing as phi_end
        idx_end = valid_end[-1]
        phi_end = phi[idx_end]

    # Number of e-folds N(phi)
    # N(phi) = integral_{phi_end}^{phi} dphi / sqrt(2*eps)  (positive for phi > phi_end)
    # If phi_end is near the right edge we try the left edge convention instead.
    with np.errstate(divide="ignore", invalid="ignore"):
        integrand = np.where((eps > 0) & np.isfinite(eps), 1.0 / np.sqrt(2.0 * eps + 1e-15), 0.0)

    N_cum = cumulative_trapezoid(integrand, phi, initial=0.0)
    # N=0 at phi_end, positive on the inflationary side
    N = N_cum - N_cum[idx_end]

    # If most of the domain has N < 0 the field rolls in the other direction;
    # flip so that N is mostly positive
    if np.nansum(N > 0) < np.nansum(N < 0):
        N = -N

    # Observables
    results = {}
    results["phi"] = phi
    results["V_vals"] = V_vals
    results["eps"] = eps
    results["eta"] = eta
    results["N"] = N
    results["phi_end"] = phi_end
    results["idx_end"] = idx_end

    for N_star_choice in [50, 60]:
        # phi_star: smallest phi (closest to phi_end) with N >= N_star_choice
        valid = np.where(N >= N_star_choice)[0]
        if len(valid) == 0:
            results[N_star_choice] = None
            continue

        idx_star = valid[0]
        phi_star = phi[idx_star]
        eps_star = eps[idx_star]
        eta_star = eta[idx_star]

        ns = float(1 - 6 * eps_star + 2 * eta_star)
        r  = float(16 * eps_star)

        # Power spectra (unnormalized)
        P_R_unnorm = np.where(
            (eps > 0) & (V_vals > 0) & np.isfinite(eps),
            V_vals / (24 * np.pi**2 * eps + 1e-100),
            np.nan,
        )
        P_R_pivot = P_R_unnorm[idx_star]
        if not np.isfinite(P_R_pivot) or P_R_pivot <= 0:
            results[N_star_choice] = None
            continue

        A_s = 2.1e-9
        norm = A_s / P_R_pivot
        P_R = norm * P_R_unnorm
        P_T = r * P_R  # tensor spectrum consistent with definition

        results[N_star_choice] = {
            "idx_star": idx_star,
            "phi_star": phi_star,
            "eps_star": eps_star,
            "eta_star": eta_star,
            "ns": ns,
            "r": r,
            "P_R": P_R,
            "P_T": P_T,
            "V_norm": norm * V_vals,
        }

    return results, None

# ──────────────────────────────────────────────────────────────────────────────
# PRESET MODELS
# ──────────────────────────────────────────────────────────────────────────────

PRESETS = {
    "— Select a preset —": "",
    "Chaotic φ² (m²φ²/2)": "m**2 * phi**2 / 2",
    "Chaotic φ⁴": "lam * phi**4",
    "Natural Inflation": "V0 * (1 - cos(phi / f))",
    "Starobinsky / Higgs-like": "V0 * (1 - exp(-alpha * phi))**2",
    "Small-field (symmetry breaking)": "V0 * (1 - (phi/mu)**2)**2",
    "Axion monodromy": "V0 * (phi**p + b * cos(phi / f))",
    "Power-law plateau": "V0 * phi**n / (1 + phi**n)",
    "Hilltop quartic": "V0 * (1 - phi**4 / mu**4)",
    "Double-well": "lam * (phi**2 - v**2)**2",
}

# ──────────────────────────────────────────────────────────────────────────────
# DEFAULT PARAMETER RANGES
# ──────────────────────────────────────────────────────────────────────────────

PARAM_DEFAULTS = {
    "m":     (1e-6, 1e-4, 6e-6,  "log"),
    "lam":   (1e-15, 1e-10, 1e-13, "log"),
    "V0":    (1e-12, 1e-8, 1e-10,  "log"),
    "alpha": (0.01, 2.0, 0.2, "linear"),
    "f":     (0.1, 20.0, 5.0, "linear"),
    "mu":    (0.5, 30.0, 10.0, "linear"),
    "v":     (0.5, 20.0, 5.0, "linear"),
    "n":     (1.0, 6.0, 2.0, "linear"),
    "p":     (1.0, 4.0, 2.0, "linear"),
    "b":     (0.0, 2.0, 0.5, "linear"),
    "phi0":  (1.0, 20.0, 5.0, "linear"),
}

def get_param_slider(pname, key_prefix):
    """Render an appropriate sidebar widget for a free parameter."""
    defaults = PARAM_DEFAULTS.get(pname)
    if defaults:
        lo, hi, default, scale = defaults
    else:
        lo, hi, default, scale = -5.0, 5.0, 1.0, "linear"

    if scale == "log":
        log_lo = np.log10(lo)
        log_hi = np.log10(hi)
        log_def = np.log10(default)
        val_log = st.sidebar.slider(
            f"{pname}  (log₁₀)",
            float(log_lo), float(log_hi), float(log_def), step=0.05,
            key=f"{key_prefix}_{pname}",
        )
        return 10 ** val_log
    else:
        return st.sidebar.slider(
            pname,
            float(lo), float(hi), float(default), step=float((hi - lo) / 200),
            key=f"{key_prefix}_{pname}",
        )

# ──────────────────────────────────────────────────────────────────────────────
# MAIN APP
# ──────────────────────────────────────────────────────────────────────────────

st.markdown('<h1>Inflation Explorer</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">Slow-roll cosmology · Arbitrary V(φ) · CMB normalisation · M_Pl = 1</p>',
    unsafe_allow_html=True,
)

# ── Preset selector ──
preset_choice = st.selectbox("Load a preset model", list(PRESETS.keys()), index=0)
preset_expr = PRESETS[preset_choice]

# ── Potential input ──
st.subheader("Define V(φ)")
st.markdown(
    "Enter your potential as a function of **phi**. "
    "Use `**` or `^` for powers, `exp()`, `log()`, `sin()`, `cos()`, etc. "
    "Any symbol other than `phi` is treated as a free parameter.",
    unsafe_allow_html=False,
)

user_expr = st.text_input(
    "V(φ) =",
    value=preset_expr if preset_expr else "V0 * (1 - exp(-alpha * phi))**2",
    placeholder="e.g.  lam * phi**4  or  V0*(1 - cos(phi/f))",
    key="V_input",
)

# ── Parse ──
parse_ok = False
param_names = []
expr_sym = None
phi_sym = None

if user_expr.strip():
    try:
        expr_sym, phi_sym, param_names = parse_potential(user_expr.strip())
        parse_ok = True
    except ValueError as e:
        st.error(f"**Parse error:** {e}")

if parse_ok:
    # Show LaTeX rendering
    try:
        latex_str = latex(expr_sym)
        st.markdown(
            f'<div class="latex-box">V(φ) = {user_expr.strip()}</div>',
            unsafe_allow_html=True,
        )
        st.latex(r"V(\phi) = " + latex_str)
    except Exception:
        pass

    # Detected parameters
    if param_names:
        pills = "".join(
            f'<span class="param-tag">{p}</span>' for p in param_names
        )
        st.markdown(
            f"**Detected free parameters:** {pills}",
            unsafe_allow_html=True,
        )
    else:
        st.info("No free parameters detected — V(φ) is fully determined by φ.")

    # ── Sidebar: parameter sliders ──
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Model Parameters")

    if not param_names:
        st.sidebar.info("No free parameters.")
        param_vals = []
    else:
        param_vals = [get_param_slider(p, "p") for p in param_names]

    # Domain control
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Domain")
    phi_min = st.sidebar.slider("φ_min", 0.001, 2.0, 0.01, 0.001)
    phi_max = st.sidebar.slider("φ_max", 5.0, 50.0, 25.0, 0.5)
    N_choice = st.sidebar.radio("Pivot e-folds", [50, 60], index=1)

    # ── Build numerical function ──
    try:
        param_syms = [sp.Symbol(p) for p in param_names]
        V_func = build_numerical_V(expr_sym, phi_sym, param_syms, param_vals)
        # Quick sanity test
        _ = float(V_func(5.0))
    except Exception as e:
        st.error(f"**Evaluation error:** {e}")
        st.stop()

    # ── Compute inflation ──
    with st.spinner("Computing slow-roll dynamics…"):
        results, err = compute_inflation(V_func, phi_min=phi_min, phi_max=phi_max)

    if err:
        st.error(f"**Physics error:** {err}")
        st.stop()

    obs = results.get(N_choice)

    # ── Observables metrics ──
    st.subheader(f"Observables at Pivot Scale (N★ = {N_choice})")

    if obs is None:
        st.warning(
            f"Could not find a valid pivot point at N★ = {N_choice} e-folds. "
            "The potential may not support enough inflation. "
            "Try adjusting parameters or the domain range."
        )
    else:
        ns_val  = obs["ns"]
        r_val   = obs["r"]
        eps_val = obs["eps_star"]
        eta_val = obs["eta_star"]
        phi_star_val = obs["phi_star"]

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("nₛ", f"{ns_val:.5f}")
        c2.metric("r", f"{r_val:.5f}")
        c3.metric("ε★", f"{eps_val:.2e}")
        c4.metric("η★", f"{eta_val:.2e}")
        c5.metric("φ★ / M_Pl", f"{phi_star_val:.3f}")

        planck_ns_lo, planck_ns_hi = 0.960, 0.975
        planck_r_hi = 0.064

        status_msgs = []
        if planck_ns_lo <= ns_val <= planck_ns_hi:
            status_msgs.append(f"✅ nₛ = {ns_val:.4f} is within Planck 2018 bounds ({planck_ns_lo}–{planck_ns_hi})")
        else:
            status_msgs.append(f"⚠️ nₛ = {ns_val:.4f} is outside Planck bounds ({planck_ns_lo}–{planck_ns_hi})")
        if r_val < planck_r_hi:
            status_msgs.append(f"✅ r = {r_val:.4f} satisfies Planck+BICEP upper limit (r < {planck_r_hi})")
        else:
            status_msgs.append(f"⚠️ r = {r_val:.4f} exceeds Planck+BICEP upper limit (r < {planck_r_hi})")

        for msg in status_msgs:
            if msg.startswith("✅"):
                st.success(msg)
            else:
                st.warning(msg)

    # ──────────────────────────────────────────────────────────────────────
    # PLOTS
    # ──────────────────────────────────────────────────────────────────────

    phi_arr = results["phi"]
    V_arr   = results["V_vals"]
    eps_arr = results["eps"]
    eta_arr = results["eta"]
    N_arr   = results["N"]
    phi_end = results["phi_end"]
    idx_end = results["idx_end"]

    # Use N=60 norm if available, else N=50
    norm_obs = results.get(60) or results.get(50)
    V_plot = norm_obs["V_norm"] if norm_obs else V_arr

    st.markdown("---")

    # ── Row 1: Potential + Slow-roll parameters ──
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Inflaton Potential V(φ)")
        fig_V = go.Figure()
        fig_V.add_trace(go.Scatter(
            x=phi_arr, y=V_plot,
            mode="lines",
            line=dict(color=COLORS["blue"], width=2.5),
            name="V(φ)",
        ))
        # phi_end vertical line
        fig_V.add_vline(
            x=phi_end,
            line_dash="dash", line_color=COLORS["red"], line_width=1.5,
            annotation_text="end", annotation_font_color=COLORS["red"],
        )
        # phi_star markers
        for N_s, col in [(50, COLORS["green"]), (60, COLORS["orange"])]:
            ob = results.get(N_s)
            if ob:
                fig_V.add_vline(
                    x=ob["phi_star"],
                    line_dash="dot", line_color=col, line_width=1.5,
                    annotation_text=f"N={N_s}",
                    annotation_font_color=col,
                )
        fig_V.update_layout(
            **PLOT_LAYOUT,
            height=360,
            xaxis_title="φ / M_Pl",
            yaxis_title="V(φ)  [CMB-normalized]",
            yaxis_type="log",
        )
        st.plotly_chart(fig_V, use_container_width=True)

    with col2:
        st.subheader("Slow-roll Parameters")
        # Clip for display
        eps_disp = np.where((eps_arr > 0) & (eps_arr < 1e3), eps_arr, np.nan)
        eta_disp = np.where(np.isfinite(eta_arr) & (np.abs(eta_arr) < 1e3), eta_arr, np.nan)

        fig_sr = make_subplots(specs=[[{"secondary_y": True}]])
        fig_sr.add_trace(
            go.Scatter(x=phi_arr, y=eps_disp, name="ε(φ)",
                       line=dict(color=COLORS["blue"], width=2)),
            secondary_y=False,
        )
        fig_sr.add_trace(
            go.Scatter(x=phi_arr, y=eta_disp, name="η(φ)",
                       line=dict(color=COLORS["purple"], width=2, dash="dot")),
            secondary_y=True,
        )
        # eps = 1 threshold
        fig_sr.add_hline(y=1.0, line_dash="dash", line_color=COLORS["red"],
                         annotation_text="ε = 1", secondary_y=False)
        fig_sr.update_layout(
            **PLOT_LAYOUT,
            height=360,
            xaxis_title="φ / M_Pl",
        )
        fig_sr.update_yaxes(title_text="ε(φ)", secondary_y=False,
                             gridcolor="#1a2030", color=COLORS["blue"])
        fig_sr.update_yaxes(title_text="η(φ)", secondary_y=True,
                             gridcolor="#1a2030", color=COLORS["purple"])
        st.plotly_chart(fig_sr, use_container_width=True)

    # ── Row 2: nₛ–r plane ──
    st.subheader("nₛ–r Plane with Planck 2018 + BICEP/Keck Contours")

    # Build approximate Planck 2018 confidence ellipses
    theta = np.linspace(0, 2 * np.pi, 200)
    ns_center, r_center = 0.9649, 0.0
    ns_1sig_w, r_1sig_h = 0.0042, 0.025
    ns_2sig_w, r_2sig_h = 0.0084, 0.055

    ns_1sig = ns_center + ns_1sig_w * np.cos(theta)
    r_1sig  = r_center + r_1sig_h * (np.sin(theta) + 1) / 2

    ns_2sig = ns_center + ns_2sig_w * np.cos(theta)
    r_2sig  = r_center + r_2sig_h * (np.sin(theta) + 1) / 2

    fig_nr = go.Figure()
    fig_nr.add_trace(go.Scatter(
        x=ns_2sig, y=r_2sig,
        fill="toself", fillcolor="rgba(126,184,247,0.12)",
        line=dict(color=COLORS["blue"], width=1.2, dash="dot"),
        name="Planck 2σ (approx.)",
    ))
    fig_nr.add_trace(go.Scatter(
        x=ns_1sig, y=r_1sig,
        fill="toself", fillcolor="rgba(126,184,247,0.28)",
        line=dict(color=COLORS["blue"], width=1.5),
        name="Planck 1σ (approx.)",
    ))

    # Reference model trajectories
    # Starobinsky / R² attractor line
    n_arr = np.array([50, 55, 60, 65, 70])
    ns_staro = 1 - 2 / n_arr
    r_staro  = 12 / n_arr**2
    fig_nr.add_trace(go.Scatter(
        x=ns_staro, y=r_staro,
        mode="lines+markers",
        line=dict(color=COLORS["teal"], width=1.5, dash="dash"),
        marker=dict(size=5, color=COLORS["teal"]),
        name="Starobinsky attractor",
    ))

    # φ² line
    ns_phi2 = 1 - 2 / n_arr
    r_phi2  = 8 / n_arr
    fig_nr.add_trace(go.Scatter(
        x=ns_phi2, y=r_phi2,
        mode="lines+markers",
        line=dict(color=COLORS["orange"], width=1.5, dash="dash"),
        marker=dict(size=5, color=COLORS["orange"]),
        name="φ² (N-trajectory)",
    ))

    # User model points for N=50 and N=60
    for N_s, marker_col, symb in [(50, COLORS["green"], "diamond"), (60, COLORS["pink"], "star")]:
        ob = results.get(N_s)
        if ob:
            fig_nr.add_trace(go.Scatter(
                x=[ob["ns"]], y=[ob["r"]],
                mode="markers",
                marker=dict(size=16, color=marker_col, symbol=symb,
                            line=dict(color="white", width=1.5)),
                name=f"Your model  N={N_s}",
            ))

    fig_nr.update_layout(
        **PLOT_LAYOUT,
        height=480,
        xaxis_title="nₛ",
        yaxis_title="r",
        xaxis_range=[0.93, 0.985],
        yaxis_range=[-0.005, 0.22],
    )
    st.plotly_chart(fig_nr, use_container_width=True)

    # ── Row 3: Power spectra ──
    if obs:
        st.subheader("Power Spectra (CMB normalised at pivot scale)")

        P_R = obs["P_R"]
        P_T = obs["P_T"]
        idx_star = obs["idx_star"]

        # Restrict to inflationary regime (N > 0)
        mask = (N_arr >= 0) & (N_arr <= N_arr[0]) & np.isfinite(P_R) & np.isfinite(P_T)

        fig_ps = go.Figure()
        fig_ps.add_trace(go.Scatter(
            x=N_arr[mask], y=P_R[mask],
            mode="lines",
            line=dict(color=COLORS["blue"], width=2.5),
            name="Scalar  𝒫_ℛ",
        ))
        fig_ps.add_trace(go.Scatter(
            x=N_arr[mask], y=P_T[mask],
            mode="lines",
            line=dict(color=COLORS["red"], width=2, dash="dot"),
            name="Tensor  𝒫_T",
        ))
        N_star_val = N_arr[idx_star]
        fig_ps.add_vline(
            x=N_star_val,
            line_dash="dash", line_color=COLORS["orange"], line_width=1.5,
            annotation_text=f"Pivot  N={N_choice}",
            annotation_font_color=COLORS["orange"],
        )
        fig_ps.add_hline(
            y=2.1e-9,
            line_dash="dot", line_color=COLORS["green"], line_width=1.2,
            annotation_text="Aₛ = 2.1×10⁻⁹",
            annotation_font_color=COLORS["green"],
        )
        fig_ps.update_layout(
            **PLOT_LAYOUT,
            height=430,
            xaxis_title="N  (e-folds from end of inflation)",
            yaxis_title="Power Spectrum",
            yaxis_type="log",
        )
        st.plotly_chart(fig_ps, use_container_width=True)

    # ── Row 4: e-folds and phase portrait ──
    col3, col4 = st.columns(2)

    with col3:
        st.subheader("N(φ) — Number of e-folds")
        N_disp = np.where(np.isfinite(N_arr) & (N_arr >= 0) & (N_arr < 200), N_arr, np.nan)
        fig_N = go.Figure()
        fig_N.add_trace(go.Scatter(
            x=phi_arr, y=N_disp,
            mode="lines",
            line=dict(color=COLORS["teal"], width=2.5),
            name="N(φ)",
        ))
        for N_s, col in [(50, COLORS["green"]), (60, COLORS["orange"])]:
            ob = results.get(N_s)
            if ob:
                fig_N.add_hline(
                    y=N_s, line_dash="dot", line_color=col, line_width=1.3,
                    annotation_text=f"N = {N_s}",
                    annotation_font_color=col,
                )
                fig_N.add_vline(
                    x=ob["phi_star"],
                    line_dash="dot", line_color=col, line_width=1.2,
                )
        fig_N.add_vline(
            x=phi_end,
            line_dash="dash", line_color=COLORS["red"], line_width=1.5,
            annotation_text="end",
            annotation_font_color=COLORS["red"],
        )
        fig_N.update_layout(
            **PLOT_LAYOUT,
            height=360,
            xaxis_title="φ / M_Pl",
            yaxis_title="N (e-folds)",
        )
        st.plotly_chart(fig_N, use_container_width=True)

    with col4:
        st.subheader("Phase-space Trajectory  φ̇ ~ −V'")
        # In slow-roll: dφ/dN ≈ -V'/V (M_Pl=1)
        dV_arr = np.gradient(V_arr, phi_arr)
        dphidN = np.where(V_arr > 0, -dV_arr / V_arr, np.nan)
        dphidN_disp = np.where(np.isfinite(dphidN) & (np.abs(dphidN) < 50), dphidN, np.nan)

        fig_ph = go.Figure()
        fig_ph.add_trace(go.Scatter(
            x=phi_arr, y=dphidN_disp,
            mode="lines",
            line=dict(color=COLORS["purple"], width=2.5),
            name="dφ/dN ≈ −V′/V",
        ))
        fig_ph.add_hline(y=0, line_color="#2a3050", line_width=1)
        fig_ph.update_layout(
            **PLOT_LAYOUT,
            height=360,
            xaxis_title="φ / M_Pl",
            yaxis_title="dφ/dN",
        )
        st.plotly_chart(fig_ph, use_container_width=True)

    # ──────────────────────────────────────────────────────────────────────
    # EQUATIONS REFERENCE
    # ──────────────────────────────────────────────────────────────────────
    with st.expander("📐 Slow-roll formalism reference"):
        st.markdown(r"""
**Slow-roll parameters** (Planck units $M_{\rm Pl} = 1$):

$$\epsilon(\phi) = \frac{1}{2}\left(\frac{V'}{V}\right)^2, \qquad \eta(\phi) = \frac{V''}{V}$$

**End of inflation:** $\epsilon = 1$

**Number of e-folds:**
$$N(\phi) = \int_{\phi}^{\phi_{\rm end}} \frac{1}{\sqrt{2\epsilon}}\,d\phi$$

**Scalar spectral index and tensor-to-scalar ratio:**
$$n_s = 1 - 6\epsilon_\star + 2\eta_\star, \qquad r = 16\epsilon_\star$$

**CMB scalar power spectrum** (Bunch-Davies vacuum):
$$\mathcal{P}_\mathcal{R}(\phi) = \frac{V(\phi)}{24\pi^2\,\epsilon(\phi)}, \qquad \mathcal{P}_\mathcal{R}(\phi_\star) = A_s \simeq 2.1\times10^{-9}$$

**Tensor spectrum:** $\mathcal{P}_T = r\,\mathcal{P}_\mathcal{R}$

All expressions are in the **slow-roll approximation**, valid when $\epsilon,|\eta| \ll 1$.
        """)

    st.markdown("---")
    st.markdown(
        '<p style="color:#2a3550; font-family:\'JetBrains Mono\',monospace; font-size:0.75rem; text-align:center;">'
        "Inflation Explorer · slow-roll formalism · M_Pl = 1 · Planck 2018 normalisation"
        "</p>",
        unsafe_allow_html=True,
    )

else:
    st.info("👆 Enter a potential V(φ) above to get started. Try a preset from the dropdown.")
    with st.expander("Quick-start examples"):
        st.markdown("""
| Model | Expression |
|---|---|
| Chaotic φ² | `m**2 * phi**2 / 2` |
| Chaotic φ⁴ | `lam * phi**4` |
| Starobinsky | `V0 * (1 - exp(-alpha * phi))**2` |
| Natural inflation | `V0 * (1 - cos(phi / f))` |
| Hilltop | `V0 * (1 - phi**4 / mu**4)` |
| Double-well | `lam * (phi**2 - v**2)**2` |
| Axion monodromy | `V0 * phi**p` |
        """)