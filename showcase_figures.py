"""Figures for a lecturer-facing explanation: real scores, own thresholds, one borrower."""
from typing import Mapping

import numpy as np
import pandas as pd
import plotly.graph_objects as go

import aave_core as core
from charts import AMBER, INK, TEAL, _scene, polish

SCENARIO = "#C4ADFF"


def borrower_surface(row: Mapping, response: dict, name: str, point: dict) -> go.Figure:
    """Use observed-day and count axes, and the chosen model's raw score on the vertical axis."""
    ages, counts = response["ages"], response["ks"]
    values = response["models"][name]["scores"]
    fig = go.Figure()
    for i, count in enumerate(counts):
        x = np.tile([count-.32, count+.32], (len(ages), 1))
        y = np.tile(ages[:, None], (1, 2))
        z = np.tile(values[i, :, None], (1, 2))
        data = np.column_stack([np.full(len(ages), count), ages, values[i], response["support"][i]])
        data = np.repeat(data[:, None, :], 2, axis=1)
        fig.add_trace(go.Surface(x=x, y=y, z=z, customdata=data, cmin=0, cmax=1,
                                colorscale=[[0, "#293A50"], [.5, "#4297A3"], [1, "#9DE8DA"]],
                                showscale=False, opacity=.95,
                                lighting=dict(ambient=.9, diffuse=.5, specular=.08),
                                hovertemplate="Thanh lý 30 ngày: %{customdata[0]:.0f} lần"
                                "<br>Lịch sử vay quan sát: %{customdata[1]:.1f} ngày"
                                "<br>Điểm mô hình: %{customdata[2]:.5f}"
                                "<br>Mẫu tham chiếu theo hai trục: %{customdata[3]:.0f}<extra></extra>"))
    threshold = float(core.summary()["models"][name]["threshold"])
    plane_name = f"Ngưỡng cảnh báo {threshold:.3f}"
    fig.add_trace(go.Surface(x=[counts.min()-.4, counts.max()+.4], y=[ages.min(), ages.max()],
                            z=np.full((2, 2), threshold), opacity=.17, showscale=False,
                            colorscale=[[0, AMBER], [1, AMBER]], name=plane_name,
                            hovertemplate=plane_name+"<extra></extra>"))
    fig.add_trace(go.Scatter3d(x=[counts.max()+.35], y=[ages.min()], z=[threshold], mode="text",
                              text=[f"Ngưỡng {threshold:.3f}"], textfont=dict(size=12, color=AMBER),
                              textposition="top right", showlegend=False, hoverinfo="skip"))
    original = float(row[core.SCORE_COLUMNS[name]])
    _add_point(fig, row["past_liquidation_count_30d"], row["borrower_history_days"], original,
               "Ca thật", INK, threshold, connector=point["is_original"])
    if not point["is_original"]:
        changed = point["features"]
        score = point["results"][name]["score"]
        _add_point(fig, changed["past_liquidation_count_30d"], changed["borrower_history_days"], score,
                   "Đang thử", SCENARIO, threshold, connector=True)
    scene = _scene("Thanh lý 30 ngày trước<br>(số lần)", "Lịch sử vay quan sát<br>(ngày)", "Điểm mô hình", z_range=[0, 1],
                   camera=dict(eye=dict(x=1.65, y=1.75, z=1.2)))
    scene["xaxis"].update(tickvals=counts.tolist(), range=[counts.min()-.5, counts.max()+.6])
    days = sorted(set([float(ages.min()), 365., 730., 1095., float(ages.max())]))
    days = [day for day in days if ages.min() <= day <= ages.max()]
    scene["yaxis"].update(tickvals=days, ticktext=[f"{day:,.0f}" for day in days])
    scene["zaxis"].update(tickvals=[0, .25, .5, .75, 1], tickformat=".2f")
    polish(fig, 540, margin=dict(l=10, r=30, t=10, b=50))
    fig.update_layout(scene=scene, uirevision="borrower-lesson-"+name,
                      legend=dict(orientation="h", y=-.08, x=0, font=dict(size=11), itemclick="toggle"))
    return fig


def _add_point(fig: go.Figure, count: float, age: float, score: float,
               label: str, color: str, threshold: float, connector: bool) -> None:
    fig.add_trace(go.Scatter3d(x=[count], y=[age], z=[score], mode="markers", name=label,
                              marker=dict(color=color, size=8, symbol="diamond", line=dict(color="#17212B", width=2)),
                              hovertemplate=f"{label}<br>Điểm {score:.5f}<extra></extra>"))
    annotations = list(fig.layout.scene.annotations or [])
    annotations.append(dict(x=count, y=age, z=score, text=f"{label}: {score:.3f}", showarrow=False,
                            bgcolor="#18212D", bordercolor=color, borderwidth=.7, borderpad=3,
                            yshift=18, font=dict(size=11, color=color)))
    fig.update_layout(scene_annotations=annotations)
    if connector:
        fig.add_trace(go.Scatter3d(x=[count, count], y=[age, age], z=[score, threshold], mode="lines",
                                  line=dict(color=color, dash="dot", width=4), showlegend=False, hoverinfo="skip"))


def detection_chart(comparison: pd.DataFrame, positive_count: int) -> go.Figure:
    """Compare observed detections without hiding the missed positives or differing alert loads."""
    names = comparison["Phương án"].tolist()
    found = comparison["Phát hiện đúng"].to_numpy()
    missed = positive_count - found
    fig = go.Figure()
    fig.add_trace(go.Bar(y=names, x=found, name="Phát hiện", orientation="h", marker_color=TEAL,
                         text=[str(int(value)) for value in found], textposition="inside",
                         hovertemplate="%{y}<br>Phát hiện %{x} mẫu<extra></extra>"))
    fig.add_trace(go.Bar(y=names, x=missed, name="Bỏ sót", orientation="h", marker_color="#344052",
                         text=[str(int(value)) for value in missed], textposition="inside",
                         hovertemplate="%{y}<br>Bỏ sót %{x} mẫu<extra></extra>"))
    fig.update_xaxes(title="Số mẫu có thanh lý trong 7 ngày", range=[0, positive_count])
    fig.update_yaxes(autorange="reversed")
    polish(fig, 300, margin=dict(l=10, r=20, t=15, b=70), barmode="stack")
    return fig
