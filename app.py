"""RallyIQ: replay a pro badminton rally and see what every shot was worth.

Run locally with:  streamlit run app.py
"""
import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from xgboost import XGBClassifier

st.set_page_config(page_title="RallyIQ", page_icon="🏸", layout="wide")

GOLD, INK, BLUE = "#B08A4F", "#151515", "#4C78A8"
OUTCOME_LABEL = {0: "returned", 1: "winner", 2: "error"}


@st.cache_resource
def load_model():
    model = XGBClassifier()
    model.load_model("data/xgb_model.json")
    return model


@st.cache_data
def load_data():
    shots = pd.read_csv("data/test_shots.csv.gz")
    matches = pd.read_csv("data/matches.csv")
    usage = pd.read_csv("data/shot_usage.csv", index_col=0)
    columns = json.load(open("data/feature_columns.json"))
    metrics = json.load(open("data/metrics.json"))
    return shots, matches, usage, columns, metrics


model = load_model()
shots, matches, usage, columns, metrics = load_data()
feature_cols = ["f__" + c for c in columns]
shot_cols = [c for c in columns if c.startswith("shot_")]


def recommend(row):
    """Try every realistic shot from this exact situation and rank them."""
    base = row[feature_cols].to_numpy(dtype=float)
    area = row["hit_area"]
    if area in usage.index:
        allowed = set(usage.columns[usage.loc[area] >= 0.03])
    else:
        allowed = set(usage.columns)

    options, names = [], []
    for col in shot_cols:
        name = col.replace("shot_", "")
        if name not in allowed or "serve" in name or name == "unknown":
            continue
        x = base.copy()
        for other in shot_cols:
            x[columns.index(other)] = 0.0
        x[columns.index(col)] = 1.0
        options.append(x)
        names.append(name)
    if not options:
        return pd.DataFrame()
    probs = model.predict_proba(np.vstack(options))
    out = pd.DataFrame({"shot": names, "winner": probs[:, 1], "error": probs[:, 2]})
    out["edge"] = out["winner"] - out["error"]
    return out.sort_values("edge", ascending=False).reset_index(drop=True)


def court_polygon(m):
    pts = np.array([[m.upleft_x, m.upleft_y], [m.upright_x, m.upright_y],
                    [m.downright_x, m.downright_y], [m.downleft_x, m.downleft_y]], dtype=float)
    center = pts.mean(axis=0)
    order = np.argsort(np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0]))
    pts = pts[order]
    return np.vstack([pts, pts[:1]])


def net_line(corners):
    """The net crosses the court's center, which in a camera view is where the diagonals meet."""
    top = corners[np.argsort(corners[:, 1])[:2]]
    bottom = corners[np.argsort(corners[:, 1])[2:]]
    tl, tr = top[np.argsort(top[:, 0])]
    bl, br = bottom[np.argsort(bottom[:, 0])]
    d1, d2 = br - tl, bl - tr
    t_ = np.cross(tr - tl, d2) / np.cross(d1, d2)
    cy = (tl + t_ * d1)[1]
    side = lambda a, b: a + (b - a) * (cy - a[1]) / (b[1] - a[1])
    return np.vstack([side(tl, bl), side(tr, br)])


# ---------- sidebar: pick a rally ----------
st.sidebar.title("🏸 RallyIQ")
st.sidebar.caption("Pick a rally from a match the model never trained on.")

matches = matches.sort_values(["year", "tournament"], ascending=[False, True])
labels = {r.match_key: f"{r.winner} vs {r.loser} · {r.tournament} {r.year}" for r in matches.itertuples()}
match_key = st.sidebar.selectbox("Match", list(labels), format_func=labels.get)
m = matches.set_index("match_key").loc[match_key]
names = {"A": str(m.winner).title(), "B": str(m.loser).title()}

in_match = shots[shots["match_key"] == match_key]
set_name = st.sidebar.selectbox("Set", sorted(in_match["set"].unique()))
in_set = in_match[in_match["set"] == set_name]
lengths = in_set.groupby("rally").size()
rally_list = sorted(lengths.index)
# start on a mid-length rally that ended in a winner, it makes the best first demo
ended_winner = in_set.sort_values("ball_round").groupby("rally")["outcome"].last() == 1
good = [r_ for r_ in rally_list if ended_winner[r_] and 6 <= lengths[r_] <= 14]
rally = st.sidebar.selectbox("Rally", rally_list, index=rally_list.index(good[0]) if good else 0,
                             format_func=lambda r_: f"Rally {r_}  ({lengths[r_]} shots)")
r = in_set[in_set["rally"] == rally].sort_values("ball_round").reset_index(drop=True)

with st.sidebar.expander("How good is the model?"):
    st.write(f"Trained on **{metrics['matches'] - metrics['test_matches']} matches**, tested on "
             f"**{metrics['test_matches']} unseen matches** ({metrics['test_shots']:,} shots).")
    table = pd.DataFrame(metrics["models"]).set_index("model")[["winner_auc", "error_auc", "log_loss"]]
    st.dataframe(table, width="stretch")
    st.caption("AUC: 0.5 is random, 1.0 is perfect. Log loss: lower is better "
               f"(always guessing the average scores {metrics['models'][0]['log_loss_guessing']}).")

# ---------- header ----------
start_a, start_b = int(r["start_A"].iloc[0]), int(r["start_B"].iloc[0])
last = r.iloc[-1]
won_by = names["A"] if (last["outcome"] == 1) == (last["player"] == "A") else names["B"]
st.title("RallyIQ")
st.markdown(f"**{names['A']}** vs **{names['B']}** · {m.tournament} {m['round']} · {set_name.replace('set', 'Set ')} · "
            f"score {start_a}–{start_b} · **{len(r)} shots** · point to **{won_by}**")

step = st.slider("Replay the rally", 1, len(r), len(r), help="Drag to step through the rally shot by shot")
shown = r.iloc[:step]
cur = r.iloc[step - 1]

left, right = st.columns([1.15, 1])

# ---------- court ----------
with left:
    fig = go.Figure()
    poly = court_polygon(m)
    fig.add_trace(go.Scatter(x=poly[:, 0], y=poly[:, 1], mode="lines", line=dict(color="#9C9588", width=2),
                             fill="toself", fillcolor="rgba(92,140,90,0.10)", hoverinfo="skip", showlegend=False))
    net = net_line(poly[:-1])
    fig.add_trace(go.Scatter(x=net[:, 0], y=net[:, 1], mode="lines",
                             line=dict(color=INK, width=3), hoverinfo="skip", showlegend=False))
    for _, row in shown.iterrows():
        color = GOLD if row["player"] == "A" else BLUE
        is_cur = row["ball_round"] == cur["ball_round"]
        x0, y0 = row["player_location_x"], row["player_location_y"]
        x1, y1 = row["landing_x"], row["landing_y"]
        if np.isnan([x0, y0, x1, y1]).any():
            continue
        fig.add_annotation(x=x1, y=y1, ax=x0, ay=y0, xref="x", yref="y", axref="x", ayref="y",
                           arrowhead=2, arrowsize=1.2, arrowwidth=3.5 if is_cur else 1.5,
                           arrowcolor=color, opacity=1 if is_cur else 0.45)
    fig.add_trace(go.Scatter(x=[cur["player_location_x"]], y=[cur["player_location_y"]], mode="markers+text",
                             marker=dict(size=14, color=GOLD if cur["player"] == "A" else BLUE),
                             text=[names[cur["player"]].split()[0]], textposition="top center", showlegend=False))
    fig.add_trace(go.Scatter(x=[cur["opponent_location_x"]], y=[cur["opponent_location_y"]], mode="markers",
                             marker=dict(size=14, color="white", line=dict(color=INK, width=2)), showlegend=False))
    fig.update_yaxes(autorange="reversed", visible=False, scaleanchor="x")
    fig.update_xaxes(visible=False)
    fig.update_layout(height=470, margin=dict(l=0, r=0, t=10, b=0), plot_bgcolor="white")
    st.plotly_chart(fig, width="stretch")
    st.caption(f"Gold = {names['A']}, blue = {names['B']}. Arrows run from the hitter to where the shuttle landed.")

# ---------- current shot + recommendation ----------
with right:
    hitter = names[cur["player"]]
    st.subheader(f"Shot {int(cur['ball_round'])}: {hitter} plays a {cur['shot']}")
    c1, c2, c3 = st.columns(3)
    c1.metric("Winner chance", f"{cur['ens_winner']:.0%}")
    c2.metric("Error chance", f"{cur['ens_error']:.0%}")
    c3.metric("What happened", OUTCOME_LABEL[int(cur["outcome"])])

    st.markdown("**Best realistic options from this spot**")
    rec = recommend(cur)
    if rec.empty:
        st.info("No recommendation for serves.")
    else:
        rec_show = rec.head(5).copy()
        rec_show["shot"] = [f"{s}  ← played" if s == cur["shot"] else s for s in rec_show["shot"]]
        st.dataframe(
            rec_show.drop(columns="edge").rename(columns={"winner": "winner chance", "error": "error chance"}),
            hide_index=True, width="stretch",
            column_config={
                "winner chance": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1),
                "error chance": st.column_config.NumberColumn(format="percent"),
            })
        st.caption("Each option is scored by the XGBoost model with everything else held fixed. Only shots pros "
                   "actually play from this court zone are shown, ranked by winner chance minus error chance.")

# ---------- shot-by-shot chart ----------
st.markdown("#### Shot by shot")
chart = go.Figure()
chart.add_trace(go.Bar(x=r["ball_round"], y=r["ens_winner"], name="winner chance",
                       marker_color=[GOLD if p == "A" else BLUE for p in r["player"]]))
chart.add_trace(go.Scatter(x=r["ball_round"], y=r["ens_error"], name="error chance",
                           mode="lines+markers", line=dict(color="#C0504D")))
chart.add_vline(x=cur["ball_round"], line_dash="dot", line_color=INK)
chart.update_layout(height=260, margin=dict(l=0, r=0, t=10, b=0), yaxis_tickformat=".0%",
                    xaxis_title="shot number", legend=dict(orientation="h", y=1.15), plot_bgcolor="white")
st.plotly_chart(chart, width="stretch")

st.caption("Data: ShuttleSet and ShuttleSet22 (CoachAI, MIT License). Predictions are an average of an XGBoost "
           "model and a causal Transformer that reads the rally so far.")
