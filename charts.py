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


def _wrapped(text, width=23):
    """Wrap prose axis labels without changing the feature or plotted values."""
    lines, current = [], ""
    for word in text.split():
        if current and len(current)+len(word)+1 > width:
            lines.append(current)
            current = word
        else:
            current = (current+" "+word).strip()
    return "<br>".join(lines+[current])


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
    fig = go.Figure(go.Waterfall(orientation="h", y=["Điểm nền"]+top.label.tolist()+["27 biến còn lại", "Margin cuối"],
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


def logistic_surface(row, xname, yname, x_max, y_max):
    obj = load_model("Logistic")
    features = obj["features"]
    xs, ys = np.linspace(0, x_max, 28), np.linspace(0, y_max, 28)
    xx, yy = np.meshgrid(xs, ys)
    X = np.tile(np.array([row[f] for f in features], np.float32), (xx.size, 1))
    X[:, features.index(xname)], X[:, features.index(yname)] = xx.ravel(), yy.ravel()
    zz = obj["pipeline"].predict_proba(X)[:, 1].reshape(xx.shape)
    point = float(obj["pipeline"].predict_proba(np.array([[row[f] for f in features]], np.float32))[0, 1])
    fig = go.Figure(go.Surface(x=xs, y=ys, z=zz,
                               colorscale=[[0, "#254F64"], [.3, "#358D9B"], [.6, TEAL], [.82, AMBER], [1, RED]],
                               cmin=0, cmax=1, colorbar=dict(title=dict(text="Điểm", side="right"), len=.55, thickness=9, x=1.01,
                                                           tickvals=[0, .5, 1], tickfont=dict(color=MUTED, size=10), outlinewidth=0),
                               lighting=dict(ambient=.82, diffuse=.8, specular=.18, roughness=.85),
                               contours=dict(z=dict(show=True, color="rgba(231,237,245,.22)", width=1)),
                               opacity=.96, name="Bề mặt điểm",
                               hovertemplate=f"{label(xname)}: %{{x:.2f}}<br>{label(yname)}: %{{y:.2f}}<br>Điểm: %{{z:.4f}}<extra></extra>"))
    fig.add_trace(go.Scatter3d(x=[row[xname]], y=[row[yname]], z=[point], mode="markers+text",
                              marker=dict(color=INK, size=7, line=dict(color="#18202A", width=3)), name="Ca đang xem",
                              text=["Ca đang xem"], textposition="top center", textfont=dict(color=INK, size=11),
                              hovertemplate=f"Ca đang xem<br>Điểm Logistic: {point:.5f}<extra></extra>"))
    polish(fig, 570, margin=dict(l=0, r=35, t=12, b=48))
    fig.update_layout(scene=_scene(_wrapped(label(xname)), _wrapped(label(yname)), "Điểm Logistic", z_range=[0, 1]),
                      legend=dict(orientation="h", x=0, y=-.04, font=dict(color=MUTED, size=11)),
                      uirevision=f"logistic-{xname}-{yname}")
    return fig


def tree3d(row, tree_index=0, max_depth=4):
    obj = load_model("Random Forest")
    clf = obj["pipeline"].named_steps["clf"]
    x = np.array([[row[f] for f in obj["features"]]], np.float32)
    x = obj["pipeline"].named_steps["impute"].transform(x)
    estimator = clf.estimators_[tree_index]
    t = estimator.tree_
    path = set(estimator.decision_path(x).indices.tolist())
    nodes, edges = [], []
    def visit(node, depth, xpos, span):
        vals = t.value[node, 0]
        score = float(vals[1] / vals.sum())
        feature = obj["features"][t.feature[node]] if t.feature[node] >= 0 else None
        desc = f"Nút {node}<br>" + (f"{label(feature)} ≤ {t.threshold[node]:.4g}" if feature else "Lá") + f"<br>Mẫu ở nút: {t.n_node_samples[node]}<br>Điểm dương ở nút: {score:.4f}"
        if depth == max_depth and t.children_left[node] != -1:
            desc += "<br>Hình dừng ở độ sâu chọn; cây vẫn tiếp tục"
        nodes.append(dict(id=node, x=xpos, y=score, z=-depth, text=desc, path=node in path, depth=depth))
        if depth < max_depth and t.children_left[node] != -1:
            for child, shift in [(t.children_left[node], -span), (t.children_right[node], span)]:
                visit(child, depth+1, xpos+shift, span/2)
                edges.append((node, child))
    visit(0, 0, 0., 1.)
    lookup = {n["id"]: n for n in nodes}
    fig = go.Figure()
    for highlighted in [False, True]:
        xline, yline, zline = [], [], []
        for parent, child in edges:
            if ((parent in path and child in path) == highlighted):
                a, b = lookup[parent], lookup[child]
                xline += [a["x"], b["x"], None]
                yline += [a["y"], b["y"], None]
                zline += [a["z"], b["z"], None]
        fig.add_trace(go.Scatter3d(x=xline, y=yline, z=zline, mode="lines", hoverinfo="skip",
                                   opacity=1 if highlighted else .45,
                                   line=dict(color=AMBER if highlighted else "#5A6C7D", width=7 if highlighted else 2),
                                   name="Đường đi của ca" if highlighted else "Nhánh khác"))
    for highlighted in [False, True]:
        group = [n for n in nodes if n["path"] == highlighted]
        fig.add_trace(go.Scatter3d(x=[n["x"] for n in group], y=[n["y"] for n in group], z=[n["z"] for n in group],
                                  mode="markers+text" if highlighted else "markers",
                                  text=[str(n["id"]) for n in group] if highlighted else None,
                                  textfont=dict(size=11, color=INK), textposition="top center",
                                  hovertext=[n["text"] for n in group], hovertemplate="%{hovertext}<extra></extra>", showlegend=False,
                                  marker=dict(size=8 if highlighted else 4.5, color=AMBER if highlighted else BLUE,
                                              opacity=1 if highlighted else .68,
                                              line=dict(color="#18202A", width=1))))
    polish(fig, 600, margin=dict(l=0, r=15, t=16, b=35))
    scene = _scene("Các nhánh của cây", "Điểm dương<br>ở nút", "Độ sâu", camera=dict(eye=dict(x=.6, y=2.65, z=.75)))
    scene["xaxis"].update(showticklabels=False, showgrid=False)
    scene["yaxis"].update(range=[0, 1], nticks=3)
    scene["zaxis"].update(tickvals=[-d for d in range(max_depth+1)], ticktext=[str(d) for d in range(max_depth+1)])
    scene["aspectratio"] = dict(x=1.8, y=.6, z=1.15)
    fig.update_layout(scene=scene, uirevision=f"forest-{tree_index}-{max_depth}",
                      legend=dict(orientation="h", x=0, y=-.02, font=dict(color=MUTED, size=11)))
    steps = []
    for node in estimator.decision_path(x).indices:
        fidx = t.feature[node]
        if fidx >= 0:
            feature = obj["features"][fidx]
            value = float(x[0, fidx])
            steps.append({"Nút": int(node), "Biến": label(feature), "Giá trị ca": value, "Ngưỡng": float(t.threshold[node]),
                          "Rẽ": "Trái (≤)" if value <= t.threshold[node] else "Phải (>)"})
    leaf = int(estimator.apply(x)[0])
    score = float(estimator.predict_proba(x)[0, 1])
    forest_score = float(clf.predict_proba(x)[0, 1])
    return fig, pd.DataFrame(steps), {"leaf": leaf, "tree_score": score, "forest_score": forest_score, "actual_depth": estimator.get_depth(), "nodes": t.node_count}


def xgb_cloud(sample, tree_limit=400):
    obj = load_model("XGBoost")
    X = sample[obj["features"]].to_numpy(np.float32)
    X = obj["pipeline"].named_steps["impute"].transform(X)
    booster = obj["pipeline"].named_steps["clf"].get_booster()
    scores = booster.predict(DMatrix(X), iteration_range=(0, tree_limit))
    fig = go.Figure()
    for target, color, name in [(0, BLUE, "Không ghi nhận thanh lý"), (1, RED, "Có thanh lý 7 ngày sau")]:
        mask = sample.target.to_numpy() == target
        g = sample.loc[mask]
        fig.add_trace(go.Scatter3d(x=np.log1p(g.borrow_count_30d), y=np.log1p(g.borrower_history_days), z=scores[mask], mode="markers", name=name,
                                  marker=dict(color=color, size=6 if target else 2.7, opacity=1 if target else .22,
                                              line=dict(color="#FFE6EB" if target else color, width=.7 if target else 0)),
                                  customdata=np.column_stack([g.borrow_count_30d, g.borrower_history_days]),
                                  hovertemplate="Vay 30 ngày: %{customdata[0]:.0f}<br>Tuổi lịch sử: %{customdata[1]:.1f} ngày<br>Điểm: %{z:.5f}<extra>%{fullData.name}</extra>"))
    polish(fig, 580, margin=dict(l=0, r=15, t=16, b=55))
    scene = _scene("Vay 30 ngày<br>ln(1 + số lần)", "Lịch sử vay<br>ln(1 + số ngày)", "Điểm XGBoost", z_range=[0, 1],
                   camera=dict(eye=dict(x=1.65, y=1.85, z=1.25)))
    fig.update_layout(scene=scene, uirevision="xgb-cloud",
                      legend=dict(orientation="h", x=0, y=-.05, font=dict(color=MUTED, size=11)))
    return fig


def boosting_path(row):
    obj = load_model("XGBoost")
    X = np.array([[row[f] for f in obj["features"]]], np.float32)
    X = obj["pipeline"].named_steps["impute"].transform(X)
    dm = DMatrix(X)
    booster = obj["pipeline"].named_steps["clf"].get_booster()
    stages = [1, 5, 10, 20, 40, 60, 80, 100, 150, 200, 250, 300, 350, 400]
    scores = [float(booster.predict(dm, iteration_range=(0, k))[0]) for k in stages]
    fig = go.Figure(go.Scatter(x=stages, y=scores, mode="lines+markers", line=dict(color=TEAL, width=2.5),
                              marker=dict(size=6, line=dict(color=PANEL, width=1)), name="Điểm của ca",
                              hovertemplate="%{x} cây đầu<br>Điểm %{y:.5f}<extra></extra>",
                              fill="tozeroy", fillcolor="rgba(105,222,208,.07)"))
    fig.add_hline(y=obj["threshold"], line_dash="dash", line_color=RED, line_width=1.2,
                  annotation_text="Ngưỡng khóa · tham chiếu ở 400 cây", annotation_position="top left", annotation_font=dict(color=RED, size=10))
    fig.update_xaxes(title="Số cây đầu của mô hình 400 cây đã fit")
    fig.update_yaxes(title="Điểm", range=[0, 1])
    return polish(fig, 310, showlegend=False, hovermode="x unified")
