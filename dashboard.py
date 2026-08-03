"""
Soccer Simulation — Agent Emotional Analysis Dashboard
=======================================================
Requirements:
    pip install streamlit pandas plotly scipy statsmodels numpy

Run:
    streamlit run dashboard.py
"""

import io
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import chi2_contingency
from scipy.stats import chi2 as chi2_dist
import numpy as np
import plotly.express as px
import statsmodels.api as sm
import statsmodels.formula.api as smf
import patsy

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Simulation Analysis",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Colour maps ───────────────────────────────────────────────────────────────
EMOTION_COLOURS = {
    "Joy":             "#F9CA24",
    "Gratification":   "#6AB04C",
    "Satisfaction":    "#22A6B3",
    "Distress":        "#E55039",
    "Disappointment":  "#EB4D4B",
    "FearsConfirmed":  "#8E44AD",
}
ACTION_COLOURS = {
    "Celebrate":     "#F9CA24",
    "Chant":         "#6AB04C",
    "FormGroup":     "#22A6B3",
    "ComfortAlly":   "#48DBFB",
    "Boo":           "#E55039",
    "CalmSituation": "#9B59B6",
    "WatchCalmly":   "#95A5A6",
}

OCEAN_TRAITS = ["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"]
PSYCH_VARS = OCEAN_TRAITS + ["PrevMood_P", "PrevMood_A", "PrevMood_D", "EmotionIntensity"]
PROSOCIAL_ACTIONS = ["ComfortAlly", "CalmSituation"]

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    h1 { color: #1F3864; }
    .sec {
        font-size: 12px; font-weight: 700; letter-spacing: .1em;
        text-transform: uppercase; color: #2E75B6;
        border-bottom: 2px solid #2E75B6;
        padding-bottom: 4px; margin: 0 0 14px 0;
    }
    .filter-note {
        font-size: 12px; color: #888; font-style: italic; margin-bottom: 10px;
    }
    div[data-testid="metric-container"] {
        background: #f4f8fd;
        border: 1px solid #d0e4f7;
        border-radius: 8px;
        padding: 10px 14px;
    }
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────
def get_octant(p, a, d):
    ps  = "+P" if p >= 0 else "-P"
    as_ = "+A" if a >= 0 else "-A"
    ds  = "+D" if d >= 0 else "-D"
    name = {
        "+P+A+D": "Exuberant",  "+P+A-D": "Dependent",
        "+P-A+D": "Relaxed",    "+P-A-D": "Docile",
        "-P+A+D": "Hostile",    "-P+A-D": "Anxious",
        "-P-A+D": "Disdainful", "-P-A-D": "Bored",
    }
    key = ps + as_ + ds
    return f"{key} {name.get(key, '')}"


@st.cache_data
def load_df(content: bytes) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(content), sep=";", decimal=",", encoding="utf-8-sig")
    num_cols = [
        "Openness", "Conscientiousness", "Extraversion",
        "Agreeableness", "Neuroticism", "Stability", "EmotionIntensity",
        "PrevMood_P", "PrevMood_A", "PrevMood_D",
        "NewMood_P",  "NewMood_A",  "NewMood_D",
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df["DeltaP"] = df["NewMood_P"] - df["PrevMood_P"]
    df["DeltaA"] = df["NewMood_A"] - df["PrevMood_A"]
    df["DeltaD"] = df["NewMood_D"] - df["PrevMood_D"]
    df["PrevOctant"] = df.apply(
        lambda r: get_octant(r["PrevMood_P"], r["PrevMood_A"], r["PrevMood_D"]), axis=1)
    df["NewOctant"] = df.apply(
        lambda r: get_octant(r["NewMood_P"], r["NewMood_A"], r["NewMood_D"]), axis=1)
    df["OctantChanged"] = df["PrevOctant"] != df["NewOctant"]
    return df


def prob_bar(series: pd.Series, colour_map: dict, height=260) -> go.Figure:
    """Plain horizontal probability bar chart (used for emotions)."""
    counts = series.value_counts()
    if counts.empty:
        return go.Figure()
    probs   = counts / counts.sum()
    labels  = probs.index.tolist()
    values  = probs.values.tolist()
    colours = [colour_map.get(l, "#95A5A6") for l in labels]
    fig = go.Figure(go.Bar(
        x=values, y=labels,
        orientation="h",
        marker_color=colours,
        text=[f"{v:.1%}" for v in values],
        textposition="outside",
        cliponaxis=False,
    ))
    fig.update_layout(
        height=height,
        margin=dict(l=0, r=60, t=10, b=10),
        xaxis=dict(
            tickformat=".0%",
            range=[0, min(1.15, max(values) * 1.35)],
            showgrid=True, gridcolor="#eee",
        ),
        yaxis=dict(autorange="reversed"),
        plot_bgcolor="white",
        paper_bgcolor="white",
        showlegend=False,
        font=dict(size=12),
    )
    return fig


def action_prob_bar(sub: pd.DataFrame, colour_map: dict, height=260) -> go.Figure:
    """
    Horizontal probability bar chart for actions.

    Hovering over a bar shows the mean OCEAN trait values of the agents
    who performed that action under the current filters.
    """
    if sub.empty:
        return go.Figure()

    counts = sub["Action"].value_counts()
    probs  = counts / counts.sum()

    ocean_means = (
        sub.groupby("Action")[OCEAN_TRAITS]
           .mean()
           .reindex(counts.index)
    )

    labels  = counts.index.tolist()
    values  = probs.values.tolist()
    colours = [colour_map.get(l, "#95A5A6") for l in labels]

    hover_texts = []
    for action in labels:
        n   = counts[action]
        p   = probs[action]
        row = ocean_means.loc[action]
        tip = (
            f"<b>{action}</b><br>"
            f"P(action) = {p:.1%}  ({n} agents)<br>"
            f"<br>"
            f"<b>Mean OCEAN traits</b><br>"
            f"  Openness          {row['Openness']:.3f}<br>"
            f"  Conscientiousness {row['Conscientiousness']:.3f}<br>"
            f"  Extraversion      {row['Extraversion']:.3f}<br>"
            f"  Agreeableness     {row['Agreeableness']:.3f}<br>"
            f"  Neuroticism       {row['Neuroticism']:.3f}"
        )
        hover_texts.append(tip)

    fig = go.Figure(go.Bar(
        x=values,
        y=labels,
        orientation="h",
        marker_color=colours,
        text=[f"{v:.1%}" for v in values],
        textposition="outside",
        cliponaxis=False,
        hovertemplate="%{customdata}<extra></extra>",
        customdata=hover_texts,
    ))
    fig.update_layout(
        height=height,
        margin=dict(l=0, r=60, t=10, b=10),
        xaxis=dict(
            tickformat=".0%",
            range=[0, min(1.15, max(values) * 1.35)],
            showgrid=True, gridcolor="#eee",
        ),
        yaxis=dict(autorange="reversed"),
        plot_bgcolor="white",
        paper_bgcolor="white",
        showlegend=False,
        font=dict(size=12),
        hoverlabel=dict(
            bgcolor="white",
            bordercolor="#d0e4f7",
            font_size=13,
            font_family="monospace",
        ),
    )
    return fig


def multiselect_all(label, options, key):
    """Multiselect preceded by an All toggle."""
    use_all = st.checkbox(f"All {label}", value=True, key=f"all_{key}")
    if use_all:
        return list(options)
    chosen = st.multiselect(label, options, default=list(options), key=key)
    return chosen if chosen else list(options)


def delta_card(col, label, value):
    """Coloured delta metric card."""
    arrow  = "▲" if value > 0 else ("▼" if value < 0 else "—")
    colour = "#27AE60" if value > 0 else ("#E74C3C" if value < 0 else "#888")
    col.markdown(
        f"<div style='text-align:center;padding:10px;background:#f4f8fd;"
        f"border:1px solid #d0e4f7;border-radius:8px'>"
        f"<div style='font-size:11px;color:#555;font-weight:600'>{label}</div>"
        f"<div style='font-size:22px;font-weight:700;color:{colour}'>"
        f"{arrow} {abs(value):.4f}</div></div>",
        unsafe_allow_html=True,
    )


def cramers_v(confusion_matrix):
    chi2_stat, p, dof, expected = chi2_contingency(confusion_matrix)
    n = confusion_matrix.to_numpy().sum()
    phi2 = chi2_stat / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    rcorr = r - ((r - 1) ** 2) / (n - 1)
    kcorr = k - ((k - 1) ** 2) / (n - 1)
    return np.sqrt(phi2corr / min((kcorr - 1), (rcorr - 1)))


# ═════════════════════════════════════════════════════════════════════════════
# Upload
# ═════════════════════════════════════════════════════════════════════════════
st.title("⚽  Soccer Simulation — Emotional Agent Analysis")

uploaded = st.file_uploader("Upload simulation CSV", type="csv")
if not uploaded:
    st.info("Upload a CSV file to begin.")
    st.stop()

df = load_df(uploaded.read())
st.success(
    f"Loaded **{len(df):,} records** · "
    f"{df['AgentID'].nunique()} agents · "
    f"{df['Personality'].nunique()} personality types · "
    f"{df['EventID'].nunique()} event types"
)

st.markdown("---")

ALL_PERS  = sorted(df["Personality"].unique())
ALL_EVTS  = sorted(df["EventID"].unique())
ALL_EMOS  = sorted(df["Emotion"].unique())
ALL_ACTS  = sorted(df["Action"].unique())
ALL_OCT   = sorted(df["NewOctant"].unique())

# ═════════════════════════════════════════════════════════════════════════════
# Data explorer — independent filtered table
# ═════════════════════════════════════════════════════════════════════════════
st.markdown(
    '<div class="sec">Data explorer — independent filtered record table</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="filter-note">'
    'Filters: Personality · Event · Action · Emotion · Mood Octant. '
    'Independent from all sections above.'
    '</div>',
    unsafe_allow_html=True,
)

dt1, dt2 = st.columns(2)
dt3, dt4, dt5 = st.columns(3)

with dt1: sel_t_pers = multiselect_all("Personalities", ALL_PERS, "dt_pers")
with dt2: sel_t_evts = multiselect_all("Events",        ALL_EVTS, "dt_evt")
with dt3: sel_t_acts = multiselect_all("Actions",       ALL_ACTS, "dt_act")
with dt4: sel_t_emos = multiselect_all("Emotions",      ALL_EMOS, "dt_emo")
with dt5: sel_t_oct  = multiselect_all("Mood Octants",  ALL_OCT,  "dt_oct")

tbl = df[
    df["Personality"].isin(sel_t_pers) &
    df["EventID"].isin(sel_t_evts) &
    df["Action"].isin(sel_t_acts) &
    df["Emotion"].isin(sel_t_emos) &
    df["NewOctant"].isin(sel_t_oct)
].reset_index(drop=True)

DISPLAY_COLS = [
    "AgentID", "AgentName", "Team", "Personality",
    "Openness", "Conscientiousness", "Extraversion", "Agreeableness",
    "Neuroticism", "Stability",
    "EventID", "Emotion", "EmotionIntensity",
    "PrevMood_P", "PrevMood_A", "PrevMood_D", "PrevOctant",
    "NewMood_P",  "NewMood_A",  "NewMood_D",  "NewOctant",
    "DeltaP", "DeltaA", "DeltaD", "OctantChanged",
    "Action",
    "Time"
]
tbl_view = tbl[[c for c in DISPLAY_COLS if c in tbl.columns]]

st.markdown(f"**{len(tbl_view):,} records** match the current filters.")

st.dataframe(
    tbl_view,
    use_container_width=True,
    hide_index=True,
    height=420,
    column_config={
        "EmotionIntensity": st.column_config.ProgressColumn(
            "Intensity", min_value=0, max_value=1, format="%.3f"),
        "OctantChanged": st.column_config.CheckboxColumn("Octant shifted?"),
        "DeltaP":  st.column_config.NumberColumn("ΔP",  format="%.4f"),
        "DeltaA":  st.column_config.NumberColumn("ΔA",  format="%.4f"),
        "DeltaD":  st.column_config.NumberColumn("ΔD",  format="%.4f"),
        "PrevMood_P": st.column_config.NumberColumn("Prev P", format="%.3f"),
        "PrevMood_A": st.column_config.NumberColumn("Prev A", format="%.3f"),
        "PrevMood_D": st.column_config.NumberColumn("Prev D", format="%.3f"),
        "NewMood_P":  st.column_config.NumberColumn("New P",  format="%.3f"),
        "NewMood_A":  st.column_config.NumberColumn("New A",  format="%.3f"),
        "NewMood_D":  st.column_config.NumberColumn("New D",  format="%.3f"),
        "Openness":          st.column_config.NumberColumn(format="%.3f"),
        "Conscientiousness": st.column_config.NumberColumn(format="%.3f"),
        "Extraversion":      st.column_config.NumberColumn(format="%.3f"),
        "Agreeableness":     st.column_config.NumberColumn(format="%.3f"),
        "Neuroticism":       st.column_config.NumberColumn(format="%.3f"),
        "Stability":         st.column_config.NumberColumn(format="%.3f"),
    },
)

# ═════════════════════════════════════════════════════════════════════════════
# General metrics
# ═════════════════════════════════════════════════════════════════════════════
st.markdown(
    '<div class="sec">General metrics — simulation-wide frequencies</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="filter-note">No filters — reflects all records in the uploaded file.</div>',
    unsafe_allow_html=True,
)

g1, g2, g3 = st.columns(3)

with g1:
    st.markdown("**Most common new mood octants**")
    o = df["NewOctant"].value_counts().reset_index()
    o.columns = ["Octant", "Count"]
    o["Share"] = (o["Count"] / len(df)).map(lambda x: f"{x:.1%}")
    st.dataframe(o, use_container_width=True, hide_index=True, height=300)

with g2:
    st.markdown("**Most common emotions**")
    e = df["Emotion"].value_counts().reset_index()
    e.columns = ["Emotion", "Count"]
    e["Share"] = (e["Count"] / len(df)).map(lambda x: f"{x:.1%}")
    st.dataframe(e, use_container_width=True, hide_index=True, height=300)

with g3:
    st.markdown("**Most common actions**")
    a = df["Action"].value_counts().reset_index()
    a.columns = ["Action", "Count"]
    a["Share"] = (a["Count"] / len(df)).map(lambda x: f"{x:.1%}")
    st.dataframe(a, use_container_width=True, hide_index=True, height=300)

st.markdown("---")

# ═════════════════════════════════════════════════════════════════════════════
# Emotional state variation
# ═════════════════════════════════════════════════════════════════════════════
st.markdown(
    '<div class="sec">Emotional state variation — before vs after stimulus</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="filter-note">'
    'Filters: Personality · Event — '
    'Results: mean ΔP / ΔA / ΔD, % octant shift, and P(Emotion).'
    '</div>',
    unsafe_allow_html=True,
)

c1a, c1b = st.columns(2)
with c1a:
    sel1_pers = multiselect_all("Personalities", ALL_PERS, "s1_pers")
with c1b:
    sel1_evts = multiselect_all("Events", ALL_EVTS, "s1_evt")

sub1 = df[df["Personality"].isin(sel1_pers) & df["EventID"].isin(sel1_evts)]

if sub1.empty:
    st.warning("No records match these filters.")
else:
    n1 = len(sub1)

    m1, m2, m3, m4 = st.columns(4)
    delta_card(m1, "Δ Pleasure (mean)",  sub1["DeltaP"].mean())
    delta_card(m2, "Δ Arousal (mean)",   sub1["DeltaA"].mean())
    delta_card(m3, "Δ Dominance (mean)", sub1["DeltaD"].mean())

    pct = sub1["OctantChanged"].mean()
    m4.markdown(
        f"<div style='text-align:center;padding:10px;background:#f4f8fd;"
        f"border:1px solid #d0e4f7;border-radius:8px'>"
        f"<div style='font-size:11px;color:#555;font-weight:600'>Octant shifted</div>"
        f"<div style='font-size:22px;font-weight:700;color:#2E75B6'>{pct:.1%}</div>"
        f"<div style='font-size:10px;color:#888'>of agents changed mood octant</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    t1a, t1b = st.columns(2)

    with t1a:
        st.markdown("**Mood octant — before → after**")
        oct_tbl = (
            sub1.groupby(["PrevOctant", "NewOctant"])
                .size()
                .reset_index(name="Count")
        )
        oct_tbl["Share"] = (oct_tbl["Count"] / n1).map(lambda x: f"{x:.1%}")
        st.dataframe(
            oct_tbl.sort_values("Count", ascending=False).reset_index(drop=True),
            use_container_width=True, hide_index=True, height=260,
        )

    with t1b:
        st.markdown("**P(Emotion) — probability each emotion is elicited**")
        st.plotly_chart(
            prob_bar(sub1["Emotion"], EMOTION_COLOURS, height=260),
            use_container_width=True,
        )

    st.markdown(
        f"<div class='filter-note'>Based on {n1:,} records.</div>",
        unsafe_allow_html=True,
    )

st.markdown("---")


# ═════════════════════════════════════════════════════════════════════════════
# Action probability by personality
# ═════════════════════════════════════════════════════════════════════════════
st.markdown(
    '<div class="sec">Action probability by personality</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="filter-note">'
    'Required: Personality · Event — '
    'Optional: Emotion · Mood Octant (enable below). '
    'Hover over a bar to see the mean OCEAN trait values of agents who performed that action.'
    '</div>',
    unsafe_allow_html=True,
)

c2a, c2b = st.columns(2)
with c2a:
    sel2_pers = multiselect_all("Personalities", ALL_PERS, "s2_pers")
with c2b:
    sel2_evts = multiselect_all("Events", ALL_EVTS, "s2_evt")

use_emo = st.checkbox("Enable Emotion filter (optional)", value=False, key="use_emo")
use_oct = st.checkbox("Enable Mood Octant filter (optional)", value=False, key="use_oct")

sel2_emos = ALL_EMOS
sel2_oct  = ALL_OCT

if use_emo:
    sel2_emos = st.multiselect("Emotions", ALL_EMOS, default=ALL_EMOS, key="s2_emo")
    if not sel2_emos:
        sel2_emos = ALL_EMOS

if use_oct:
    sel2_oct = st.multiselect("New Mood Octant", ALL_OCT, default=ALL_OCT, key="s2_oct")
    if not sel2_oct:
        sel2_oct = ALL_OCT

sub2 = df[
    df["Personality"].isin(sel2_pers) &
    df["EventID"].isin(sel2_evts) &
    df["Emotion"].isin(sel2_emos) &
    df["NewOctant"].isin(sel2_oct)
]

if sub2.empty:
    st.warning("No records match these filters.")
else:
    n2       = len(sub2)
    counts2  = sub2["Action"].value_counts()
    probs2   = counts2 / counts2.sum()

    ocean_means2 = (
        sub2.groupby("Action")[OCEAN_TRAITS]
            .mean()
            .round(3)
            .reindex(counts2.index)
    )

    c2c, c2d = st.columns([2, 1])

    with c2c:
        st.markdown("**P(Action)** — hover for mean OCEAN values")
        st.plotly_chart(
            action_prob_bar(sub2, ACTION_COLOURS,
                            height=max(220, len(counts2) * 54)),
            use_container_width=True,
        )

    with c2d:
        st.markdown("**Action breakdown + mean OCEAN**")
        act_tbl = pd.DataFrame({
            "Action":    counts2.index,
            "Count":     counts2.values,
            "P(action)": [f"{v:.1%}" for v in probs2.values],
            "O":  ocean_means2["Openness"].values,
            "C":  ocean_means2["Conscientiousness"].values,
            "E":  ocean_means2["Extraversion"].values,
            "A":  ocean_means2["Agreeableness"].values,
            "N":  ocean_means2["Neuroticism"].values,
        })
        st.dataframe(
            act_tbl.reset_index(drop=True),
            use_container_width=True,
            hide_index=True,
            height=max(220, len(counts2) * 54),
            column_config={
                "O": st.column_config.NumberColumn("O", format="%.3f"),
                "C": st.column_config.NumberColumn("C", format="%.3f"),
                "E": st.column_config.NumberColumn("E", format="%.3f"),
                "A": st.column_config.NumberColumn("A", format="%.3f"),
                "N": st.column_config.NumberColumn("N", format="%.3f"),
            },
        )

    st.markdown(
        f"<div class='filter-note'>Based on {n2:,} records.</div>",
        unsafe_allow_html=True,
    )

st.markdown("---")

# ═════════════════════════════════════════════════════════════════════════════
# Statistical validation — stratified by event
# ═════════════════════════════════════════════════════════════════════════════
st.markdown(
    '<div class="sec">Statistical validation of personality-action association</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="filter-note">'
    'Action probabilities differ strongly by Event, so a single pooled Chi² would '
    'confound "which event happened" with "which personality reacted". '
    'The Personality × Action association is therefore tested '
    '<b>separately within each event</b>.'
    '</div>',
    unsafe_allow_html=True,
)

if sub2.empty:
    st.warning("No records match the filters above.")
else:
    MIN_N_PER_EVENT = 30  # minimum sample size to trust a per-event chi-square

    event_rows = []
    for ev in sorted(sub2["EventID"].unique()):
        ev_df = sub2[sub2["EventID"] == ev]
        if len(ev_df) < MIN_N_PER_EVENT:
            continue
        ct = pd.crosstab(ev_df["Personality"], ev_df["Action"])
        ct = ct.loc[(ct.sum(axis=1) > 0), (ct.sum(axis=0) > 0)]
        if ct.shape[0] < 2 or ct.shape[1] < 2:
            continue
        try:
            chi2_e, p_e, dof_e, _ = chi2_contingency(ct)
            v_e = cramers_v(ct)
        except ValueError:
            continue
        event_rows.append({
            "Event": ev, "n": len(ev_df), "Chi2": chi2_e, "dof": dof_e,
            "p-value": p_e, "Cramer's V": v_e,
            "Significant (p<.05)": "Yes" if p_e < 0.05 else "No",
        })

    event_stats_df = pd.DataFrame(event_rows)

    st.markdown("**Personality × Action association, tested separately within each event**")
    if event_stats_df.empty:
        st.warning(
            f"No individual event has at least {MIN_N_PER_EVENT} matching "
            "records under the current filters, so per-event chi-square tests "
            "cannot be computed reliably (expected cell counts would be too low)."
        )
    else:
        st.dataframe(
            event_stats_df.sort_values("p-value").reset_index(drop=True),
            use_container_width=True, hide_index=True,
            column_config={
                "Chi2": st.column_config.NumberColumn(format="%.2f"),
                "p-value": st.column_config.NumberColumn(format="%.5f"),
                "Cramer's V": st.column_config.NumberColumn(format="%.3f"),
            },
        )
        st.markdown(
            f"<div class='filter-note'>Events with fewer than {MIN_N_PER_EVENT} "
            "matching records are omitted.</div>",
            unsafe_allow_html=True,
        )

    with st.expander("Pooled Chi² across all selected events (reference only — confounds Event, kept for comparison)"):
        contingency = pd.crosstab(sub2["Personality"], sub2["Action"])
        fig = px.imshow(contingency, text_auto=True, aspect="auto", color_continuous_scale="Blues")
        fig.update_layout(xaxis_title="Action", yaxis_title="Personality", height=450)
        st.plotly_chart(fig, use_container_width=True)

        chi2_pool, p_pool, dof_pool, expected_pool = chi2_contingency(contingency)
        v_pool = cramers_v(contingency)

        pc1, pc2, pc3, pc4 = st.columns(4)
        pc1.metric("Chi²", f"{chi2_pool:.2f}")
        pc2.metric("Degrees of freedom", dof_pool)
        pc3.metric("p-value", f"{p_pool:.5f}")
        pc4.metric("Cramer's V", f"{v_pool:.3f}")

        if v_pool < 0.10:
            strength = "negligible"
        elif v_pool < 0.30:
            strength = "weak"
        elif v_pool < 0.50:
            strength = "moderate"
        else:
            strength = "strong"
        st.info(f"Pooled association strength: {strength} (Cramer's V = {v_pool:.3f}). Interpret with caution — see note above.")

st.markdown("---")

# ═════════════════════════════════════════════════════════════════════════════
# Inferential Analysis of Emergent Behaviors
# ═════════════════════════════════════════════════════════════════════════════
st.markdown(
    '<div class="sec">Inferential Analysis of Emergent Behaviors</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="filter-note">'
    'Identifies and inferentially analyzes instances where an agent’s observed action '
    'deviates from the deterministic expectation of their baseline personality. Evaluates which '
    'situational variables and mood shifts (ΔP, ΔA, ΔD, Emotion Intensity) act as '
    'triggers for emergent responses using Robust Logistic Regression (clustered by AgentID).'
    '</div>',
    unsafe_allow_html=True,
)

em1, em2 = st.columns(2)
with em1:
    em_mode = st.radio(
        "Emergence Criterion:",
        options=[
            "Psychological (High Neuroticism / Low Agreeableness performing prosocial/peaceful action)",
            "Statistical (Action with low expected probability given baseline OCEAN)"
        ],
        key="em_mode_choice"
    )

with em2:
    if "Psychological" in em_mode:
        n_thresh = st.slider("High Neuroticism Threshold (N ≥)", 0.0, 1.0, 0.5, 0.05, key="n_thresh")
    else:
        prob_thresh = st.slider("Unexpected Probability Threshold (P <)", 0.05, 0.30, 0.15, 0.01, key="p_thresh")

sub_em = sub2.dropna(subset=["Action", "EventID", "AgentID"] + PSYCH_VARS + ["DeltaP", "DeltaA", "DeltaD"]).copy()

if sub_em.empty or len(sub_em) < 50:
    st.warning("Not enough records under current filters to perform inferential emergence analysis.")
else:
    PROSOCIAL_NON_AGGRESSIVE = ["ComfortAlly", "CalmSituation", "Boo", "WatchCalmly"]
    
    if "Psychological" in em_mode:
        # Agent with high Neuroticism or low Agreeableness performing peaceful/prosocial action
        sub_em["IsEmergent"] = (
            ((sub_em["Neuroticism"] >= n_thresh) | (sub_em["Agreeableness"] <= 0.4)) &
            (sub_em["Action"].isin(PROSOCIAL_NON_AGGRESSIVE))
        ).astype(int)
    else:
        # Occurrence with low relative probability in global personality distribution
        act_counts = sub_em.groupby(["Personality", "Action"]).size() / sub_em.groupby("Personality").size()
        sub_em["BaseP"] = sub_em.apply(lambda r: act_counts.get((r["Personality"], r["Action"]), 0.0), axis=1)
        sub_em["IsEmergent"] = (sub_em["BaseP"] < prob_thresh).astype(int)

    n_em = sub_em["IsEmergent"].sum()
    rate_em = sub_em["IsEmergent"].mean()

    m_em1, m_em2, m_em3 = st.columns(3)
    m_em1.metric("Emergent Records", f"{n_em:,}")
    m_em2.metric("Emergence Rate", f"{rate_em:.1%}")
    top_act = sub_em[sub_em["IsEmergent"] == 1]["Action"].mode()
    m_em3.metric("Primary Emergent Action", top_act.iloc[0] if not top_act.empty else "N/A")

    if n_em < 10 or (len(sub_em) - n_em) < 10:
        st.warning("Requires at least 10 emergent and 10 non-emergent records to fit logistic regression.")
    else:
        st.markdown("**Robust Logistic Regression: Situational Triggers of Emergent Behavior**")
        
        # Standardize predictors for clean OR calculation
        for col in ["DeltaP", "DeltaA", "DeltaD", "EmotionIntensity"]:
            std_v = sub_em[col].std()
            sub_em[f"{col}_z"] = (sub_em[col] - sub_em[col].mean()) / (std_v if std_v > 0 else 1.0)
        
        try:
            logit_mod = smf.logit(
                "IsEmergent ~ DeltaP_z + DeltaA_z + DeltaD_z + EmotionIntensity_z + C(EventID)",
                data=sub_em
            ).fit(cov_type="cluster", cov_kwds={"groups": sub_em["AgentID"]}, disp=False)
            
            params = logit_mod.params
            bse = logit_mod.bse
            pvalues = logit_mod.pvalues
            
            sit_vars = ["DeltaP_z", "DeltaA_z", "DeltaD_z", "EmotionIntensity_z"]
            var_names_clean = {
                "DeltaP_z": "Δ Pleasure (Mood)",
                "DeltaA_z": "Δ Arousal (Activation)",
                "DeltaD_z": "Δ Dominance (Control)",
                "EmotionIntensity_z": "Emotion Intensity"
            }
            
            em_rows = []
            for v in sit_vars:
                if v in params.index:
                    coef = params[v]
                    se = bse[v]
                    pv = pvalues[v]
                    or_val = np.exp(coef)
                    ci_low = np.exp(coef - 1.96 * se)
                    ci_high = np.exp(coef + 1.96 * se)
                    em_rows.append({
                        "Predictor": var_names_clean.get(v, v),
                        "Coef": coef,
                        "SE": se,
                        "p-value": pv,
                        "Odds Ratio (OR)": or_val,
                        "CI_low": ci_low,
                        "CI_high": ci_high,
                        "Significant": pv < 0.05
                    })
            
            em_res_df = pd.DataFrame(em_rows)
            
            f_em = go.Figure()
            f_em.add_trace(go.Scatter(
                x=em_res_df["Odds Ratio (OR)"],
                y=em_res_df["Predictor"],
                mode="markers",
                marker=dict(
                    size=12,
                    color=["#2E75B6" if s else "#888" for s in em_res_df["Significant"]],
                ),
                error_x=dict(
                    type="data",
                    symmetric=False,
                    array=em_res_df["CI_high"] - em_res_df["Odds Ratio (OR)"],
                    arrayminus=em_res_df["Odds Ratio (OR)"] - em_res_df["CI_low"],
                ),
                hovertemplate="<b>%{y}</b><br>Odds Ratio: %{x:.3f}<br>p-value: %{customdata:.4f}<extra></extra>",
                customdata=em_res_df["p-value"]
            ))
            f_em.add_vline(x=1.0, line_dash="dash", line_color="#999")
            f_em.update_layout(
                height=260,
                xaxis_title="Odds Ratio (OR) of Emergence Trigger (OR > 1 increases likelihood)",
                plot_bgcolor="white", paper_bgcolor="white",
                margin=dict(l=10, r=10, t=10, b=10)
            )
            st.plotly_chart(f_em, use_container_width=True)
            
            st.dataframe(
                em_res_df[["Predictor", "Odds Ratio (OR)", "p-value", "CI_low", "CI_high", "Significant"]],
                use_container_width=True, hide_index=True,
                column_config={
                    "Odds Ratio (OR)": st.column_config.NumberColumn(format="%.3f"),
                    "p-value": st.column_config.NumberColumn(format="%.4f"),
                    "CI_low": st.column_config.NumberColumn("95% CI Low", format="%.3f"),
                    "CI_high": st.column_config.NumberColumn("95% CI High", format="%.3f"),
                }
            )
            
            sig_triggers = em_res_df[em_res_df["Significant"]]
            if not sig_triggers.empty:
                trig_list = ", ".join(sig_triggers["Predictor"].tolist())
                st.success(
                    f"**Inferential Conclusion:** Situational factors that significantly trigger "
                    f"emergent behavior in agents are: **{trig_list}** (p < 0.05, adjusted for AgentID clusters)."
                )
            else:
                st.info("No single situational factor reached individual significance at p < 0.05 under this filter subset.")
                
        except Exception as ex:
            st.warning(f"Could not fit robust logistic regression model for emergence: {ex}")

st.markdown("---")

