"""Comparable real-data ranking views and checked model sensitivity; no UI dependencies."""
from typing import Mapping

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import aave_core as core
from charts import AMBER, BLUE, INK, MUTED, RED, TEAL, _scene, polish

MODEL_COLORS = {"Logistic": BLUE, "Random Forest": AMBER, "XGBoost": TEAL}
RANK_TICKS = ([0, 1, 2, 3, 4, 5], ["100%", "top 10%", "top 1%", "top 0,1%", "top 0,01%", "top 0,001%"])
LEVEL_COLUMNS = {name: "rank_" + column for name, column in core.SCORE_COLUMNS.items()}
TOP_COLUMNS = {name: "top_" + column for name, column in core.SCORE_COLUMNS.items()}


def alert_codes(df: pd.DataFrame) -> np.ndarray:
    """Encode the three locked model alerts without changing thresholds."""
    codes = np.zeros(len(df), dtype=np.uint8)
    for i, (name, column) in enumerate(core.SCORE_COLUMNS.items()):
        mask = df[column].to_numpy() >= core.summary()["models"][name]["threshold"]
        codes |= mask.astype(np.uint8) << i
    return codes


def rank_space_points(df: pd.DataFrame, scales: dict, n_background: int = 5000, seed: int = 42) -> pd.DataFrame:
    """Keep every positive/false-alert snapshot and a reproducible negative background sample."""
    codes = alert_codes(df)
    keep = df.target.to_numpy().astype(bool) | (codes > 0)
    candidates = np.flatnonzero(~keep)
    chosen = np.random.default_rng(seed).choice(candidates, min(n_background, len(candidates)), replace=False)
    keep[chosen] = True
    points = df.iloc[np.flatnonzero(keep)].copy()
    points["alert_code"] = codes[keep]
    for name, column in core.SCORE_COLUMNS.items():
        points[LEVEL_COLUMNS[name]] = scales[name].level(points[column].to_numpy())
        points[TOP_COLUMNS[name]] = scales[name].top_fraction(points[column].to_numpy())
    points.attrs["background_total"] = len(candidates)
    return points


def overlap_table(df: pd.DataFrame) -> pd.DataFrame:
    """Count positive snapshots by exact alert combination and liquidation history."""
    codes, positive = alert_codes(df), df.target.to_numpy() == 1
    history = df.past_liquidation_count_90d.to_numpy() > 0
    groups = [(7, "Cả ba"), (3, "Logistic + RF"), (5, "Logistic + XGBoost"), (6, "RF + XGBoost"),
              (1, "Chỉ Logistic"), (2, "Chỉ RF"), (4, "Chỉ XGBoost"), (0, "Không mô hình nào")]
    rows = []
    for code, name in groups:
        mask = positive & (codes == code)
        rows.append({"Mô hình cảnh báo": name, "Có tiền sử 90 ngày": int((mask & history).sum()),
                     "Chưa có tiền sử": int((mask & ~history).sum()), "Tổng": int(mask.sum())})
    return pd.DataFrame(rows)


def _customdata(df: pd.DataFrame) -> list[list]:
    """Format historical context and model-specific tail ranks for a compact hover."""
    fields = []
    for _, row in df.iterrows():
        models = "<br>".join(f"{name}: {row[column]:.4f} · top {100*row[TOP_COLUMNS[name]]:.4g}%"
                              for name, column in core.SCORE_COLUMNS.items())
        fields.append([str(row.wallet)[:8] + "…" + str(row.wallet)[-6:], str(row.cutoff)[:10],
                       float(row.health_factor), float(row.total_debt_usdc_equivalent), models,
                       "Có thanh lý trong 7 ngày" if row.target else "Không ghi nhận thanh lý 7 ngày",
                       int(row.past_liquidation_count_90d)])
    return fields


def _rank_scatter(frame: pd.DataFrame, name: str, color: str, size: float,
                  opacity: float, symbol: str = "circle", detailed: bool = True) -> go.Scatter3d:
    coordinates = [frame[LEVEL_COLUMNS[m]] for m in core.SCORE_COLUMNS]
    hover = ("Ví %{customdata[0]} · %{customdata[1]} UTC<br>HF %{customdata[2]:.4g} · Nợ "
             "%{customdata[3]:,.2f} USDC.e<br>%{customdata[4]}<br>%{customdata[5]}"
             "<br>Thanh lý 90 ngày trước: %{customdata[6]}<extra></extra>")
    return go.Scatter3d(x=coordinates[0], y=coordinates[1], z=coordinates[2], mode="markers", name=name,
                        marker=dict(color=color, size=size, opacity=opacity, symbol=symbol,
                                    line=dict(color=color, width=1 if symbol == "circle-open" else 0)),
                        customdata=_customdata(frame) if detailed else None,
                        hovertemplate=hover if detailed else "Mẫu âm không bị cảnh báo<extra></extra>")


def rank_space_figure(points: pd.DataFrame, scales: dict, active_row: Mapping) -> go.Figure:
    """Show model ranks on common empirical tail axes, with locked threshold planes."""
    positive = points.target == 1
    history = points.past_liquidation_count_90d > 0
    background = points[(~positive) & points.alert_code.eq(0)]
    false_alerts = points[(~positive) & points.alert_code.gt(0)]
    fig = go.Figure()
    fig.add_trace(_rank_scatter(background, f"Âm, không cảnh báo · mẫu {len(background):,}/"
                               f"{points.attrs['background_total']:,}", "#6D798A", 2, .16, detailed=False))
    fig.add_trace(_rank_scatter(false_alerts, f"Cảnh báo sai · {len(false_alerts):,} mẫu", "#BECADB", 3, .28))
    fig.add_trace(_rank_scatter(points[positive & history], f"Thanh lý · có tiền sử ({(positive & history).sum()})",
                               RED, 6, 1))
    fig.add_trace(_rank_scatter(points[positive & ~history], f"Thanh lý · chưa có tiền sử ({(positive & ~history).sum()})",
                               RED, 7, 1, "circle-open"))
    active = pd.DataFrame([dict(active_row)])
    for name, column in core.SCORE_COLUMNS.items():
        active[LEVEL_COLUMNS[name]] = scales[name].level(float(active_row[column]))
        active[TOP_COLUMNS[name]] = scales[name].top_fraction(float(active_row[column]))
    selected = _rank_scatter(active, "Vị thế đang xem", INK, 9, 1, "diamond")
    selected.update(mode="markers+text", text=["Ca đang xem"], textposition="top center", textfont=dict(size=12))
    fig.add_trace(selected)
    maximum = np.ceil(np.log10(next(iter(scales.values())).n) * 10) / 10
    for axis, name in enumerate(core.SCORE_COLUMNS):
        _add_rank_plane(fig, scales[name], name, axis, maximum)
    scene = _scene("Logistic", "Random Forest", "XGBoost", camera=dict(eye=dict(x=1.6, y=1.6, z=1.1)))
    for axis, (name, color) in zip(["xaxis", "yaxis", "zaxis"], MODEL_COLORS.items()):
        scene[axis].update(range=[0, maximum], tickvals=RANK_TICKS[0], ticktext=RANK_TICKS[1],
                           title=dict(text=name, font=dict(size=13, color=color)), nticks=6)
    scene.update(aspectmode="cube")
    polish(fig, 680, margin=dict(l=0, r=0, t=15, b=125))
    fig.update_layout(scene=scene, uirevision="rank-space", legend=dict(x=0, y=-.07, itemclick="toggle",
                      itemdoubleclick="toggleothers", orientation="h", font=dict(size=11)))
    return fig


def _add_rank_plane(fig: go.Figure, scale: core.RankScale, name: str, axis: int, maximum: float) -> None:
    threshold = core.summary()["models"][name]["threshold"]
    level = float(scale.level(threshold))
    vertices = np.array([[0, 0], [maximum, 0], [maximum, maximum], [0, maximum]])
    xyz = [np.full(4, level) if i == axis else vertices[:, i if i < axis else i-1] for i in range(3)]
    text = f"Ngưỡng {name} · top {100*scale.top_fraction(threshold):.3g}%"
    fig.add_trace(go.Mesh3d(x=xyz[0], y=xyz[1], z=xyz[2], i=[0, 0], j=[1, 2], k=[2, 3],
                           color=MODEL_COLORS[name], opacity=.12, name=text, showlegend=True,
                           hovertemplate=text + "<extra></extra>"))


def response_data(df: pd.DataFrame, row: Mapping, scales: dict, n_ages: int = 30) -> dict:
    """Evaluate only checked sensitivity rows and reference density on the two displayed axes."""
    low, maximum = core.history_floor(row), float(df.borrower_history_days.max())
    if low >= .9 * maximum:
        return {"available": False, "floor": low, "maximum": maximum}
    ages = np.geomspace(low, maximum, n_ages) if low > 0 else np.r_[0., np.geomspace(.001, maximum, n_ages-1)]
    ks = np.arange(int(df.past_liquidation_count_30d.max()) + 1)
    grid = core.liquidation_history_grid(row, ks, ages)
    transformed = np.log1p(ages)
    edges = np.r_[transformed[0], (transformed[:-1] + transformed[1:]) / 2, transformed[-1]]
    support = np.array([np.histogram(np.log1p(df.loc[df.past_liquidation_count_30d.eq(k),
                                                   "borrower_history_days"]), bins=edges)[0] for k in ks])
    result = {"available": True, "floor": low, "maximum": maximum, "ages": ages, "ks": ks,
              "grid": grid, "support": support, "models": {}, "level_max": np.ceil(np.log10(len(df))*10)/10}
    for name in core.SCORE_COLUMNS:
        obj = core.load_model(name)
        scores = obj["pipeline"].predict_proba(grid[obj["features"]].to_numpy(np.float32))[:, 1]
        result["models"][name] = {"scores": scores.reshape(len(ks), len(ages)),
                                   "levels": scales[name].level(scores).reshape(len(ks), len(ages)),
                                   "top": scales[name].top_fraction(scores).reshape(len(ks), len(ages)),
                                   "alerts": (scores >= obj["threshold"]).reshape(len(ks), len(ages))}
    return result


def response_surfaces_figure(row: Mapping, scales: dict, result: dict,
                             models: list[str] | None = None) -> go.Figure:
    """Plot separate integer-count strips rather than interpolate fractional liquidation counts."""
    names = models or list(core.SCORE_COLUMNS)
    fig = make_subplots(rows=1, cols=len(names), specs=[[{"type": "scene"}] * len(names)],
                        subplot_titles=names if len(names) > 1 else None, horizontal_spacing=.04)
    for column, name in enumerate(names, start=1):
        _add_response_model(fig, row, name, scales[name], result, column, column == len(names))
    margin = dict(l=20, r=125, t=20, b=35) if len(names) == 1 else dict(l=65, r=45, t=45, b=60)
    polish(fig, 590 if len(names) == 1 else 550, margin=margin, showlegend=False)
    if len(names) > 1:
        for i in range(1, len(names)+1):
            key = "scene" if i == 1 else "scene"+str(i)
            scene = fig.layout[key]
            scene.camera = dict(eye=dict(x=2.1, y=2.2, z=1.5))
            scene.aspectmode = "cube"
            for axis in ["xaxis", "yaxis", "zaxis"]:
                scene[axis].tickfont.size = 10
                scene[axis].title.font.size = 11
    for annotation, name in zip(fig.layout.annotations, names):
        annotation.font.update(color=MODEL_COLORS[name], size=15)
    fig.update_layout(uirevision="checked-response-" + "-".join(names))
    return fig


def _add_response_model(fig: go.Figure, row: Mapping, name: str, scale: core.RankScale,
                        result: dict, column: int, colorbar: bool) -> None:
    ages, ks, values = result["ages"], result["ks"], result["models"][name]
    colors = [[0, "#29394C"], [.3, "#41708D"], [.6, TEAL], [1, AMBER]]
    for i, k in enumerate(ks):
        x = np.tile([k-.32, k+.32], (len(ages), 1))
        y = np.tile(np.log1p(ages)[:, None], (1, 2))
        z = np.tile(values["levels"][i, :, None], (1, 2))
        top_labels = [f"{100*f:.4g}%" if f else f"< {100/scale.n:.4g}%" for f in values["top"][i]]
        custom = np.empty((len(ages), 5), dtype=object)
        custom[:, 0], custom[:, 1], custom[:, 2] = k, ages, values["scores"][i]
        custom[:, 3], custom[:, 4] = top_labels, result["support"][i]
        custom = np.repeat(custom[:, None, :], 2, axis=1)
        fig.add_trace(go.Surface(x=x, y=y, z=z, customdata=custom, colorscale=colors, cmin=0,
                                cmax=result["level_max"], showscale=colorbar and i == len(ks)-1,
                                colorbar=dict(title="Thứ hạng", tickvals=RANK_TICKS[0], ticktext=RANK_TICKS[1],
                                              thickness=9, len=.65, x=1.01),
                                lighting=dict(ambient=.9, diffuse=.5, specular=.08), opacity=.95,
                                hovertemplate="Thanh lý 30 ngày: %{customdata[0]:.0f}<br>Tuổi lịch sử: "
                                "%{customdata[1]:.1f} ngày<br>Điểm: %{customdata[2]:.5f} · top "
                                "%{customdata[3]}<br>Mẫu thật trong ô 2D: %{customdata[4]:.0f}<extra></extra>"),
                      row=1, col=column)
    threshold = core.summary()["models"][name]["threshold"]
    level = float(scale.level(threshold))
    fig.add_trace(go.Surface(x=[ks.min()-.4, ks.max()+.4], y=np.log1p([ages.min(), ages.max()]),
                            z=np.full((2, 2), level), colorscale=[[0, MODEL_COLORS[name]], [1, MODEL_COLORS[name]]],
                            opacity=.18, showscale=False, hovertemplate=f"Ngưỡng {name}: {threshold:.5f}"
                            f" · top {100*scale.top_fraction(threshold):.4g}%<extra></extra>"), row=1, col=column)
    fig.add_trace(go.Scatter3d(x=[row["past_liquidation_count_30d"]], y=[np.log1p(row["borrower_history_days"])],
                              z=[float(scale.level(row[core.SCORE_COLUMNS[name]]))], mode="markers+text",
                              marker=dict(color=INK, size=7, symbol="diamond"), text=["Ca thật"],
                              textposition="top center", hovertemplate="Vị thế thật tại cutoff<extra></extra>"),
                  row=1, col=column)
    scene = _scene("Thanh lý 30 ngày<br>(số lần)", "Tuổi lịch sử vay<br>(ngày · thang log)", "Mức xếp hạng")
    scene["xaxis"].update(tickvals=ks.tolist(), range=[ks.min()-.5, ks.max()+.5])
    days = sorted(set([0, 1, 7, 30, 90, 180, 365, 730, int(np.ceil(ages.max()))]))
    days = [day for day in days if ages.min() <= day <= np.ceil(ages.max())]
    scene["yaxis"].update(tickvals=np.log1p(days).tolist(), ticktext=[f"{day:,}" for day in days])
    scene["zaxis"].update(range=[0, result["level_max"]], tickvals=RANK_TICKS[0], ticktext=RANK_TICKS[1])
    fig.update_layout(**{("scene" if column == 1 else "scene" + str(column)): scene})
