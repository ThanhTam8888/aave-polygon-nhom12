"""Plot real fitted surfaces, decision paths and scores; no illustrative fake trees."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from xgboost import DMatrix

from aave_core import label, load_model, summary

TEAL = "#69DED0"
RED = "#FF7D90"
AMBER = "#F5C777"
INK = "#E7EDF5"
MUTED = "#A6B1C2"
BLUE = "#8CAEFF"
GRID = "rgba(166,177,194,.13)"
PANEL = "#15191F"


def _scene(x_title, y_title, z_title, *, camera=None, z_range=None):
    # WebGL scene grid alpha varies by renderer; solid dark strokes stay subtle.
    axis = dict(backgroundcolor=PANEL, showbackground=True, gridcolor="#303844",
                linecolor="#46515F", zerolinecolor="#303844",
                tickfont=dict(size=11, color=MUTED),
                title=dict(font=dict(size=12, color=INK)),
                showspikes=False, nticks=5)
    result = dict(
        bgcolor="rgba(0,0,0,0)", dragmode="turntable",
        aspectmode="manual", aspectratio=dict(x=1.2, y=1.1, z=.85),
        xaxis={**axis, "title": dict(text=x_title, font=dict(size=12, color=INK))},
        yaxis={**axis, "title": dict(text=y_title, font=dict(size=12, color=INK))},
        zaxis={**axis, "title": dict(text=z_title, font=dict(size=12, color=INK))},
        camera=camera or dict(eye=dict(x=1.65, y=1.7, z=1.2), projection=dict(type="perspective")),
    )
    if z_range is not None:
        result["zaxis"]["range"] = z_range
    return result


def polish(fig, height=420, **kwargs):
    defaults = dict(template="plotly_dark", height=height,
                    margin=dict(l=15, r=24, t=30, b=55),
                    font=dict(family="Source Sans, Segoe UI, Arial, sans-serif", color=INK, size=12),
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    colorway=[TEAL, AMBER, RED, BLUE],
                    hoverlabel=dict(bgcolor="#222832", bordercolor="#465466",
                                    font=dict(color=INK, size=12)),
                    legend=dict(orientation="h", y=-.2, x=0, font=dict(size=11, color=MUTED),
                                bgcolor="rgba(0,0,0,0)", itemclick="toggleothers"),
                    modebar=dict(bgcolor="rgba(0,0,0,0)", color=MUTED, activecolor=TEAL))
    defaults.update(kwargs)
    fig.update_layout(**defaults)
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID,
                     tickfont=dict(color=MUTED, size=11), title_font=dict(size=12, color=INK),
                     automargin=True)
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID,
                     tickfont=dict(color=MUTED, size=11), title_font=dict(size=12, color=INK),
                     automargin=True)
    return fig


def hf_bar(hf):
    fig = go.Figure()
    for start, end, color in [(0, 1, RED), (1, 1.3, AMBER), (1.3, 3, TEAL)]:
        fig.add_shape(type="rect", x0=start, x1=end, y0=.2, y1=.65, fillcolor=color, opacity=.32, line_width=0)
    for boundary, color in [(1, RED), (1.3, AMBER)]:
        fig.add_shape(type="line", x0=boundary, x1=boundary, y0=.15, y1=.75,
                      line=dict(color=color, width=1.5, dash="dot"))
    fig.add_trace(go.Scatter(x=[min(float(hf), 3)], y=[.86], mode="markers", marker=dict(symbol="triangle-down", size=20, color=INK,
                                                                                                  line=dict(color=PANEL, width=2)),
                             hovertemplate=f"HF thật: {float(hf):.5g}<extra></extra>"))
    fig.update_xaxes(range=[-.03, 3.05], tickvals=[0, 1, 1.3, 2, 3], ticktext=["0", "1", "1,3", "2", "3+"],
                     title="Health Factor · mốc 1 là ngưỡng giao thức", showgrid=False)
    fig.update_yaxes(visible=False, range=[0, 1.1])
    return polish(fig, 165, showlegend=False, margin=dict(l=10, r=18, t=8, b=45))


def hf_debt_cloud(df, n=4500):
    sample = df.sample(min(n, len(df)), random_state=42)
    fig = go.Figure()
    groups = [(sample.health_factor < 1, RED, "HF < 1"),
              (sample.health_factor.between(1, 1.3), AMBER, "HF 1–1,3"),
              (sample.health_factor > 1.3, TEAL, "HF > 1,3")]
    for mask, color, name in groups:
        g = sample.loc[mask]
        fig.add_trace(go.Scattergl(x=np.log10(1+g.health_factor), y=np.log10(1+g.total_debt_usdc_equivalent),
                                  mode="markers", name=name, marker=dict(color=color, size=5, opacity=.58),
                                  customdata=np.column_stack([g.health_factor, g.total_debt_usdc_equivalent]),
                                  hovertemplate="HF %{customdata[0]:.4g}<br>Nợ %{customdata[1]:,.2f} USDC.e<extra>%{fullData.name}</extra>"))
    fig.add_vrect(x0=0, x1=np.log10(2), fillcolor=RED, opacity=.045, line_width=0, layer="below")
    fig.add_vline(x=np.log10(2), line_color=RED, line_width=1.2, line_dash="dot")
    ticks = np.array([0, 1, 10, 100, 1000, 10000, 100000, 1000000, 10000000])
    fig.update_xaxes(title="Health Factor · thang log(1 + HF)", tickvals=np.log10(1+ticks),
                     ticktext=["0", "1", "10", "100", "1k", "10k", "100k", "1M", "10M"])
    fig.update_yaxes(title="Nợ USDC.e · thang log(1 + nợ)", tickvals=np.log10(1+ticks),
                     ticktext=["0", "1", "10", "100", "1k", "10k", "100k", "1M", "10M"])
    return polish(fig, 400)


def weekly_chart(weekly):
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=weekly.cutoff, y=weekly.snapshots, name="Snapshot còn nợ", marker_color="rgba(140,174,255,.25)",
                         hovertemplate="%{x|%d/%m/%Y}<br>%{y:,} snapshot còn nợ<extra></extra>"), secondary_y=False)
    fig.add_trace(go.Scatter(x=weekly.cutoff, y=weekly.positives, mode="lines+markers", name="Mẫu thanh lý 7 ngày sau", line=dict(color=TEAL, width=2.5),
                             marker=dict(size=5, line=dict(color=PANEL, width=1)),
                             hovertemplate="%{x|%d/%m/%Y}<br>%{y} mẫu thanh lý 7 ngày sau<extra></extra>"), secondary_y=True)
    fig.update_yaxes(title="Snapshot", secondary_y=False)
    fig.update_yaxes(title="Mẫu dương", secondary_y=True, rangemode="tozero")
    fig.update_xaxes(tickformat="%d/%m", nticks=9)
    return polish(fig, 335, hovermode="x unified")


def waterfall(contrib, baseline, margin):
    ordered = contrib.reindex(contrib.contribution.abs().sort_values(ascending=False).index)
    top = ordered.head(7)
    rest = float(ordered.iloc[7:].contribution.sum())
    remaining = len(ordered) - len(top)
    fig = go.Figure(go.Waterfall(orientation="h", y=["Điểm nền"]+top.label.tolist()+[f"{remaining} biến còn lại", "Margin cuối"],
                                x=[baseline]+top.contribution.tolist()+[rest, 0],
                                measure=["absolute"]+["relative"]*8+["total"],
                                increasing=dict(marker=dict(color=RED)), decreasing=dict(marker=dict(color=TEAL)),
                                totals=dict(marker=dict(color=BLUE)), connector=dict(line=dict(color="#566274", width=1)),
                                text=[f"{v:+.2f}" for v in [baseline]+top.contribution.tolist()+[rest]]+[f"{margin:.2f}"],
                                textposition="outside", textfont=dict(color=INK, size=11), cliponaxis=False,
                                hovertemplate="%{y}<br>Đóng góp %{x:.4f}<extra></extra>"))
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(title="Log-odds của đầu ra mô hình · không phải % nguy cơ")
    return polish(fig, 465, showlegend=False, margin=dict(l=10, r=55, t=20, b=55))


def boosting_path(row):
    obj = load_model("XGBoost")
    X = np.array([[row[f] for f in obj["features"]]], np.float32)
    X = obj["pipeline"].named_steps["impute"].transform(X)
    dm = DMatrix(X)
    booster = obj["pipeline"].named_steps["clf"].get_booster()
    count = booster.num_boosted_rounds()
    stages = np.unique(np.r_[1, np.geomspace(min(5, count), count, 13).round().astype(int)]).tolist()
    scores = [float(booster.predict(dm, iteration_range=(0, k))[0]) for k in stages]
    fig = go.Figure(go.Scatter(x=stages, y=scores, mode="lines+markers", line=dict(color=TEAL, width=2.5),
                              marker=dict(size=6, line=dict(color=PANEL, width=1)), name="Điểm của ca",
                              hovertemplate="%{x} cây đầu<br>Điểm %{y:.5f}<extra></extra>",
                              fill="tozeroy", fillcolor="rgba(105,222,208,.07)"))
    fig.add_hline(y=obj["threshold"], line_dash="dash", line_color=RED, line_width=1.2,
                  annotation_text="Ngưỡng khóa · tham chiếu ở 400 cây", annotation_position="top left", annotation_font=dict(color=RED, size=10))
    fig.update_xaxes(title=f"Số cây đầu của mô hình {count} cây đã fit")
    fig.update_yaxes(title="Điểm", range=[0, 1])
    return polish(fig, 310, showlegend=False, hovermode="x unified")


def lr_contribution_bar(diagnostics: dict) -> go.Figure:
    """Show an additive decomposition that exactly reconstructs the Logistic margin."""
    frame = diagnostics["contributions"]
    ordered = frame.reindex(frame.contribution.abs().sort_values(ascending=False).index)
    top, rest = ordered.head(8), ordered.iloc[8:]
    names = ["Intercept"] + [label(feature) for feature in top.feature] + [f"{len(rest)} biến còn lại"]
    values = [diagnostics["intercept"]] + top.contribution.tolist() + [float(rest.contribution.sum())]
    fig = go.Figure(go.Bar(y=names, x=values, orientation="h",
                          marker_color=[RED if value > 0 else TEAL for value in values],
                          text=[f"{value:+.3f}" for value in values], textposition="outside", cliponaxis=False,
                          hovertemplate="%{y}<br>Đóng góp vào z: %{x:.5f}<extra></extra>"))
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(title="Đóng góp vào z = intercept + Σ β × x chuẩn hóa")
    fig.add_vline(x=0, line_color=MUTED, line_width=1)
    return polish(fig, 450, showlegend=False, margin=dict(l=10, r=65, t=15, b=55))


def rf_vote_histogram(diagnostics: dict) -> go.Figure:
    """Display the distribution of fitted-tree scores, not fictitious binary votes."""
    fig = go.Figure(go.Histogram(x=diagnostics["scores"], xbins=dict(start=0, end=1, size=.05),
                                marker=dict(color=AMBER, line=dict(color="#15191F", width=1)),
                                hovertemplate="Khoảng điểm %{x}<br>%{y} cây<extra></extra>"))
    fig.add_vline(x=diagnostics["score"], line_color=TEAL, line_width=2,
                  annotation_text=f"Điểm rừng {diagnostics['score']:.3f}", annotation_position="top left")
    fig.add_vline(x=diagnostics["threshold"], line_color=INK, line_dash="dash",
                  annotation_text=f"Ngưỡng {diagnostics['threshold']:.3f}", annotation_position="top right")
    fig.update_xaxes(title="Điểm do từng cây chấm cho cùng vị thế", range=[0, 1])
    fig.update_yaxes(title="Số cây", dtick=50)
    return polish(fig, 360, showlegend=False, margin=dict(l=15, r=15, t=45, b=55))
