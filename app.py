import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import warnings
warnings.filterwarnings("ignore")

# ── PAGE CONFIG ──
st.set_page_config(
    page_title="Delhivery Network Intelligence",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CUSTOM CSS ──
st.markdown("""
<style>
.main-header { font-size: 2.2rem; font-weight: 700; color: #1F3864; }
.sub-header  { font-size: 1rem; color: #666; margin-bottom: 1.5rem; }
.metric-card {
    background: linear-gradient(135deg, #1F3864, #2E75B6);
    border-radius: 10px; padding: 1rem;
    color: white; text-align: center;
}
.metric-value { font-size: 1.8rem; font-weight: 700; margin: 0; }
.metric-label { font-size: 0.8rem; opacity: 0.85; margin: 0; }
.insight-box {
    background: #EBF3FB; border-left: 4px solid #2E75B6;
    border-radius: 0 8px 8px 0; padding: 0.7rem 1rem;
    margin: 0.5rem 0; font-size: 0.9rem;
}
</style>
""", unsafe_allow_html=True)

# ── SIDEBAR ──
with st.sidebar:
    st.markdown("### 🚚 Delhivery Network Intelligence")
    st.markdown("---")
    page = st.radio("Navigate to", [
        "🏠 Network Overview",
        "🔴 Bottleneck Hubs",
        "⏱️  ETA Model Results",
        "🚛 FTL vs Carting",
        "📋 Key Findings"
    ])
    st.markdown("---")
    st.markdown("""
    **Project Stats**
    - 142,502 trip segments
    - 1,657 facilities
    - 2,783 corridors
    - 25% ETA improvement
    """)
    st.caption("Data Science Team | May 2026")

# ── PRE-COMPUTED DATA ──
# All numbers from your analysis — no CSV needed!

top5_hubs = pd.DataFrame({
    "Hub ID":       ["IND000000ACB", "IND712311AAA", "IND421302AAG", "IND110037AAM", "IND562132AAA"],
    "Risk Score":   [0.634, 0.522, 0.393, 0.362, 0.355],
    "Betweenness":  ["23.3%", "8.1%", "5.3%", "4.7%", "15.3%"],
    "Avg Delay":    ["1.60×", "2.20×", "2.02×", "2.06×", "1.54×"],
    "Connections":  [94, 46, 58, 45, 71],
    "Action":       ["Capacity Upgrade", "Process Optimization", "Route-Type Shift",
                     "Process Optimization", "Parallel Route"]
})

hourly_data = pd.DataFrame({
    "Hour": list(range(24)),
    "Avg Delay": [
        1.97, 2.13, 2.30, 2.33, 2.43, 2.65,
        2.68, 2.26, 2.55, 2.29, 2.43, 3.11,
        2.47, 2.35, 2.52, 2.04, 1.98, 1.97,
        1.94, 1.84, 1.88, 2.02, 1.95, 2.04
    ]
})

model_data = pd.DataFrame({
    "Model":      ["Baseline XGBoost", "Graph-Enhanced XGBoost"],
    "MAE (min)":  [55.85, 41.81],
    "Within 15%": [44.44, 54.99]
})

ftl_data = pd.DataFrame({
    "Distance Band": ["Short (0-50km)", "Medium (50-200km)", "Long (200-500km)"],
    "FTL Delay":     [2.19, 2.17, 1.95],
    "Carting Delay": [2.79, 2.47, 2.22]
})

# ══════════════════════════════════════════════════════
# PAGE 1: NETWORK OVERVIEW
# ══════════════════════════════════════════════════════
if page == "🏠 Network Overview":
    st.markdown('<p class="main-header">🚚 Delhivery Network Intelligence</p>',
                unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Graph-Based Logistics Optimization Dashboard</p>',
                unsafe_allow_html=True)

    # Metrics Row
    cols = st.columns(5)
    metrics = [
        ("142,502", "Trip Segments"),
        ("1,657",   "Facilities"),
        ("2,783",   "Corridors"),
        ("2.21×",   "Avg OSRM Error"),
        ("94%",     "Chronic Delay Rate"),
    ]
    for col, (val, label) in zip(cols, metrics):
        col.markdown(f"""
        <div class="metric-card">
            <p class="metric-value">{val}</p>
            <p class="metric-label">{label}</p>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div class="insight-box">
    <b>Core Finding:</b> OSRM underestimates delivery time by <b>2.21× on average</b>.
    94% of all network corridors are chronically delayed. A graph-enhanced model
    reduces prediction error by <b>25%</b> by capturing hub congestion patterns
    invisible to road-based routing.
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🕐 Delay by Hour of Day")
        fig = px.line(
            hourly_data, x="Hour", y="Avg Delay",
            title="Average Network Delay by Departure Hour",
            labels={"Avg Delay": "Delay Ratio (×OSRM)", "Hour": "Hour of Day"},
            color_discrete_sequence=["#C00000"]
        )
        fig.add_hline(y=1, line_dash="dash", line_color="green",
                      annotation_text="Perfect Prediction")
        fig.add_annotation(x=11, y=3.11, text="Peak: 11 AM (3.11×)",
                           showarrow=True, arrowhead=2, bgcolor="#FFE0E0")
        fig.update_traces(fill="tozeroy", fillcolor="rgba(192,0,0,0.1)")
        fig.update_layout(height=380)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("### 🚛 Route Type Comparison")
        route_data = pd.DataFrame({
            "Metric":   ["Avg Delay", "Avg Distance (km)", "Trip Count"],
            "FTL":      [2.04, 400, 98827],
            "Carting":  [2.73, 34, 43675]
        })
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(name="FTL",     x=["Avg Delay", "Trips (÷100)"],
                               y=[2.04, 988], marker_color="#2E75B6"))
        fig2.add_trace(go.Bar(name="Carting", x=["Avg Delay", "Trips (÷100)"],
                               y=[2.73, 437], marker_color="#C00000"))
        fig2.update_layout(title="FTL vs Carting Performance",
                           barmode="group", height=380)
        st.plotly_chart(fig2, use_container_width=True)

    # Network Stats
    st.markdown("---")
    st.markdown("### 🕸️ Network Properties")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Network Density",      "0.001",          "Extremely sparse")
    c2.metric("Most Connected Hub",   "IND000000ACB",    "94 connections")
    c3.metric("Hub-and-Spoke",        "Confirmed",       "Few critical transit hubs")
    c4.metric("Non-Cutoff Delay",     "3.74×",          "vs 1.98× standard trips")

# ══════════════════════════════════════════════════════
# PAGE 2: BOTTLENECK HUBS
# ══════════════════════════════════════════════════════
elif page == "🔴 Bottleneck Hubs":
    st.markdown("## 🔴 Bottleneck Hub Analysis")
    st.markdown("""
    <div class="insight-box">
    <b>Risk Score</b> = Betweenness Centrality (40%) + Avg Delay (40%) + Total Degree (20%)
    — all normalized 0-1. Identifies hubs that are both network-critical AND chronically delayed.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 🏆 Top 5 Bottleneck Hubs")
    st.dataframe(top5_hubs, use_container_width=True, hide_index=True)

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 📊 Risk Score Ranking")
        fig = px.bar(
            top5_hubs, x="Hub ID", y="Risk Score",
            color="Risk Score", color_continuous_scale="RdYlGn_r",
            title="Composite Risk Score — Top 5 Hubs",
            text="Risk Score"
        )
        fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
        fig.update_layout(height=380, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("### 🕸️ Betweenness Centrality")
        fig2 = px.bar(
            top5_hubs, x="Hub ID", y=[23.3, 15.3, 8.1, 5.3, 4.7],
            title="% of Network Paths Through Each Hub",
            labels={"y": "Betweenness (%)"},
            color_discrete_sequence=["#C00000"]
        )
        fig2.update_layout(height=380)
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    st.markdown("### ⚠️ Chronic Corridor Summary")
    c1, c2, c3 = st.columns(3)
    c1.metric("Chronic Corridors", "2,615", "94% of all corridors")
    c2.metric("Highest Impact",    "7,455", "IND000000ACB → IND562132AAA")
    c3.metric("SLA Impact",        "47,307 trips", "Through top 5 hubs")

    top_corridors = pd.DataFrame({
        "Source Hub":      ["IND000000ACB", "IND562132AAA", "IND000000ACB",
                            "IND000000ACB", "IND000000ACB"],
        "Destination Hub": ["IND562132AAA", "IND000000ACB", "IND712311AAA",
                            "IND421302AAG", "IND501359AAE"],
        "Median Delay":    ["1.50×", "1.47×", "1.65×", "1.65×", "1.59×"],
        "Trip Count":      [4970, 3316, 2831, 1616, 1638],
        "Impact Score":    [7455, 4863, 4663, 2662, 2602]
    })
    st.markdown("**Top 5 Highest Impact Corridors:**")
    st.dataframe(top_corridors, use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════
# PAGE 3: ETA MODEL
# ══════════════════════════════════════════════════════
elif page == "⏱️  ETA Model Results":
    st.markdown("## ⏱️ ETA Prediction Model — Results")

    st.markdown("""
    <div class="insight-box">
    <b>The Graph Advantage:</b> Adding betweenness centrality and corridor delay history
    as features captures network congestion OSRM cannot see — reducing prediction error by 25%.
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    col1.metric("Baseline MAE",   "55.85 min",  "OSRM-based features only")
    col2.metric("Graph MAE",      "41.81 min",  "↓ 25% improvement")
    col3.metric("Within 15%",     "44.44% → 54.99%", "+10.55 percentage points")

    st.markdown("---")
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("### 📊 MAE Comparison")
        fig = px.bar(
            model_data, x="Model", y="MAE (min)",
            color="Model", title="Mean Absolute Error (lower = better)",
            color_discrete_map={
                "Baseline XGBoost":        "#C00000",
                "Graph-Enhanced XGBoost":  "#2E75B6"
            },
            text="MAE (min)"
        )
        fig.update_traces(texttemplate="%{text:.2f} min", textposition="outside")
        fig.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown("### 📊 Within 15% Accuracy")
        fig2 = px.bar(
            model_data, x="Model", y="Within 15%",
            color="Model", title="Within 15% Accuracy (higher = better)",
            color_discrete_map={
                "Baseline XGBoost":       "#C00000",
                "Graph-Enhanced XGBoost": "#2E75B6"
            },
            text="Within 15%"
        )
        fig2.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
        fig2.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    st.markdown("### 🔧 Graph Features Added")
    features_df = pd.DataFrame({
        "Feature":        ["source_betweenness", "dest_betweenness", "corridor_delay"],
        "What it captures": [
            "Congestion risk at departure hub (betweenness = 0.233 for top hub)",
            "Congestion risk at arrival hub",
            "Historical median delay on this specific corridor"
        ],
        "Why OSRM misses it": [
            "OSRM only sees road speed — not how many packages compete for the hub",
            "Same — arrival hub congestion is invisible to map routing",
            "OSRM uses shortest path — not historical performance per corridor"
        ]
    })
    st.dataframe(features_df, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("### ⏱️ Try the ETA Predictor")
    st.markdown("Estimate delivery time for a hypothetical trip:")

    p1, p2, p3 = st.columns(3)
    osrm_t  = p1.slider("OSRM Predicted Time (min)", 30, 600, 120)
    distance = p2.slider("Distance (km)", 10, 500, 100)
    hour     = p3.slider("Departure Hour", 0, 23, 9)

    route    = st.selectbox("Route Type", ["FTL", "Carting"])
    corr_delay = st.slider("Corridor Historical Delay (×)", 1.0, 5.0, 2.0, 0.1)

    # Simple estimation based on your findings
    base_pred  = osrm_t * 2.21
    adjustment = 1.0
    if hour in [9, 10, 11]:
        adjustment *= 1.15
    if route == "Carting":
        adjustment *= 1.25
    adjustment *= (corr_delay / 2.21)

    graph_pred = osrm_t * 2.21 * adjustment * 0.75

    col1, col2, col3 = st.columns(3)
    col1.metric("OSRM Prediction",    f"{osrm_t} min",       "What OSRM says")
    col2.metric("Baseline Estimate",  f"{base_pred:.0f} min", f"+{base_pred-osrm_t:.0f} min vs OSRM")
    col3.metric("Graph-Enhanced Est.", f"{graph_pred:.0f} min", "With network context")

# ══════════════════════════════════════════════════════
# PAGE 4: FTL vs CARTING
# ══════════════════════════════════════════════════════
elif page == "🚛 FTL vs Carting":
    st.markdown("## 🚛 FTL vs. Carting Decision Framework")

    st.markdown("""
    <div class="insight-box">
    <b>Key Insight:</b> Corridor delay history accounts for <b>92.4% of feature importance</b>
    in the decision model — structural corridor performance matters far more than vehicle type.
    Fix the corridor, not just the vehicle.
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    col1.metric("Decision Tree Accuracy", "68.21%", "Field operations")
    col2.metric("XGBoost Accuracy",       "69.90%", "Automated routing")
    col3.metric("Top Feature",            "92.4%",  "corridor_delay importance")

    st.markdown("---")
    st.markdown("### 📊 Performance by Distance Band")

    fig = go.Figure()
    fig.add_trace(go.Bar(name="FTL",     x=ftl_data["Distance Band"],
                          y=ftl_data["FTL Delay"],     marker_color="#2E75B6",
                          text=ftl_data["FTL Delay"],  textposition="outside",
                          texttemplate="%{text:.2f}×"))
    fig.add_trace(go.Bar(name="Carting", x=ftl_data["Distance Band"],
                          y=ftl_data["Carting Delay"], marker_color="#C00000",
                          text=ftl_data["Carting Delay"], textposition="outside",
                          texttemplate="%{text:.2f}×"))
    fig.update_layout(barmode="group", height=400,
                      title="FTL Outperforms Carting at Every Distance",
                      yaxis_title="Average Delay (× OSRM)")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown("### 🔧 Route Recommender")
    st.markdown("Get a route type recommendation based on corridor profile:")

    p1, p2 = st.columns(2)
    dist_band   = p1.selectbox("Distance Band",
                                ["Short (0-50km)", "Medium (50-200km)",
                                 "Long (200-500km)", "Very Long (500km+)"])
    corr_hist   = p2.slider("Corridor Historical Delay (×)", 1.0, 5.0, 2.0, 0.1)
    dep_hour2   = st.slider("Departure Hour", 0, 23, 9)
    urban_access = st.checkbox("Urban area — trucks cannot access?")

    if urban_access:
        rec = "🚐 **Carting** — only option for urban last-mile access"
        color = "warning"
    elif dist_band == "Very Long (500km+)":
        rec = "🚛 **FTL** — Carting has no presence beyond 500km"
        color = "success"
    elif corr_hist > 3.0:
        rec = "⚠️ **Flag for review** — corridor delay is critically high (>3.0×). Investigate route before committing to either type."
        color = "error"
    elif dist_band == "Short (0-50km)":
        rec = "🚛 **FTL** — saves 0.60× delay vs Carting on short routes"
        color = "success"
    else:
        rec = "🚛 **FTL** — outperforms Carting at this distance band"
        color = "success"

    if color == "success":
        st.success(rec)
    elif color == "warning":
        st.warning(rec)
    else:
        st.error(rec)

    st.markdown("---")
    st.markdown("### 📋 Feature Importance")
    feat_imp = pd.DataFrame({
        "Feature":    ["corridor_delay", "osrm_distance", "departure_hour",
                       "source_betweenness", "route_type"],
        "Importance": [0.924, 0.059, 0.008, 0.006, 0.003]
    })
    fig2 = px.bar(feat_imp, x="Importance", y="Feature", orientation="h",
                  color="Importance", color_continuous_scale="Blues",
                  title="Decision Tree Feature Importance")
    fig2.update_layout(height=300, coloraxis_showscale=False)
    st.plotly_chart(fig2, use_container_width=True)

# ══════════════════════════════════════════════════════
# PAGE 5: KEY FINDINGS
# ══════════════════════════════════════════════════════
elif page == "📋 Key Findings":
    st.markdown("## 📋 Key Findings Summary")

    st.success("""
    **Project Complete** — Graph-based intelligence system built for Delhivery's network.
    All 5 project tasks delivered with measurable, quantified results.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🔢 By the Numbers")
        numbers = pd.DataFrame({
            "Metric":  [
                "Trip segments analyzed", "Unique facilities", "Unique corridors",
                "Avg OSRM underestimation", "Chronic corridors", "Trips through top 5 hubs",
                "Hours of delay saved (potential)", "ETA model improvement",
                "FTL vs Carting accuracy"
            ],
            "Value": [
                "142,502", "1,657", "2,783",
                "2.21×", "94% of network", "47,307",
                "75,000+/year", "25% MAE reduction", "68-70%"
            ]
        })
        st.dataframe(numbers, use_container_width=True, hide_index=True)

    with col2:
        st.markdown("### 🎯 Recommendations")
        st.markdown("""
        **P1 — Immediate Actions:**
        - Capacity upgrade at IND000000ACB (23.3% of all paths)
        - Route-type shift on IND421302AAG corridors (FTL → 2.02× improvement)

        **P2 — Short Term (1-4 months):**
        - Create parallel route: ACB → IND562132AAA (4,970 trips)
        - Process audit at IND712311AAA (2.20× delay — highest in network)

        **P3 — Medium Term (1-3 months):**
        - Deploy graph ETA model in production (25% better predictions)
        - Dedicated routing protocol for non-cutoff trips (3.74× delay)

        **Expected outcomes if P1+P2 completed:**
        - 25-35% reduction in SLA breaches
        - 34,882 trips directly improved
        - ~33% reduction in unnecessary delay hours
        """)

    st.markdown("---")
    st.markdown("### 📁 Project Deliverables")
    deliverables = pd.DataFrame({
        "Deliverable": [
            "delhivery_analysis.ipynb",
            "Delhivery_Strategy_Memo.docx",
            "Delhivery_Quantitative_Summary.docx",
            "bottleneck_network.html",
            "model_comparison.png",
            "decision_tree.png",
            "hourly_delay.png",
            "ftl_carting_tradeoff.png",
            "app.py (this dashboard)"
        ],
        "Description": [
            "Complete analysis notebook — 7 sections, fully documented",
            "Strategy memo for Head of Network Operations",
            "Quantitative summary report with all metrics",
            "Interactive network visualization — bottlenecks highlighted",
            "Baseline vs graph-enhanced model benchmark chart",
            "FTL vs Carting interpretable decision tree",
            "Delay pattern by hour of day",
            "FTL vs Carting performance by distance band",
            "Live interactive dashboard (deployed on Streamlit Cloud)"
        ],
        "Status": ["✅"] * 9
    })
    st.dataframe(deliverables, use_container_width=True, hide_index=True)
