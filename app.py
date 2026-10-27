from pathlib import Path
import html
import io
import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import precision_recall_curve, roc_curve

import aave_core as core
import charts

ROOT = Path(__file__).resolve().parent
NAV = ["Tổng quan", "Theo dõi HF", "Chấm điểm vị thế", "Mô hình 3D", "Đối chiếu kết quả", "Hướng dẫn demo"]
st.set_page_config(page_title="Aave Polygon · Nhóm 12", page_icon="◈", layout="wide", initial_sidebar_state="expanded")
st.markdown("<style>" + (ROOT / "assets/theme.css").read_text(encoding="utf-8") + "</style>", unsafe_allow_html=True)


def render_chart(fig, *, key, width="stretch"):
    """Keep the authored accessible palette; Streamlit must not recolor these plots."""
    is_3d = any(t.type in {"scatter3d", "surface"} for t in fig.data)
    config = {"displaylogo": False, "displayModeBar": True,
              "scrollZoom": is_3d, "responsive": True,
              "toImageButtonOptions": {"format": "png", "filename": key, "scale": 2},
              "modeBarButtonsToRemove": [] if is_3d else ["lasso2d", "select2d"]}
    if is_3d:
        camera = st.session_state.get("lab_camera", "Phối cảnh")
        if camera == "Chính diện":
            fig.update_layout(scene_camera=dict(eye=dict(x=.05, y=2.8, z=.4), up=dict(x=0, y=0, z=1)))
        elif camera == "Từ trên":
            fig.update_layout(scene_camera=dict(eye=dict(x=.05, y=.05, z=3), up=dict(x=0, y=1, z=0)))
        fig.update_layout(uirevision=f"{fig.layout.uirevision}-{camera}-{st.session_state.get('camera_revision', 0)}")
    st.plotly_chart(fig, width=width, key=key, theme=None, config=config)


@st.cache_resource
def dataset():
    # Shared immutable public data; every editable object is created as a copy.
    paths = sorted((ROOT / "data").glob("test_2024_*.parquet"))
    df = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
    df["wallet"] = df.wallet.astype("category")
    df["version"] = df.version.astype("category")
    return df.set_index("sid", drop=False)


@st.cache_data
def weekly_data():
    return pd.read_csv(ROOT / "data/weekly.csv", parse_dates=["cutoff"])


@st.cache_data
def pr_roc(name):
    df = dataset()
    score = df[core.SCORE_COLUMNS[name]]
    p, r, _ = precision_recall_curve(df.target, score)
    fpr, tpr, _ = roc_curve(df.target, score)
    ind = np.unique(np.linspace(0, len(p)-1, min(len(p), 700)).astype(int))
    ri = np.unique(np.linspace(0, len(fpr)-1, min(len(fpr), 700)).astype(int))
    return p[ind], r[ind], fpr[ri], tpr[ri]


def num(value):
    return f"{int(value):,}".replace(",", ".")


def money(value):
    return f"{float(value):,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


def short_wallet(value):
    return str(value)[:8] + "…" + str(value)[-6:]


def serial(value):
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return str(value)


def hero(kicker, title, subtitle):
    st.markdown(f'<div class="hero"><div class="eyebrow">{html.escape(kicker)}</div><h1>{html.escape(title)}</h1><p>{html.escape(subtitle)}</p></div>', unsafe_allow_html=True)


def kpis(items):
    for col, (title, value, note) in zip(st.columns(len(items)), items):
        col.markdown(f'<div class="kpi"><div class="kpi-label">{html.escape(title)}</div><div class="kpi-value">{html.escape(str(value))}</div><div class="kpi-note">{html.escape(note)}</div></div>', unsafe_allow_html=True)


def navigate(page, sid=None):
    st.session_state["nav"] = page
    if sid is not None:
        st.session_state["active_sid"] = int(sid)
        st.session_state["from_table"] = True
        st.session_state.pop("custom_prediction", None)


def use_preset():
    picked = next(p for p in PRESETS if p["label"] == st.session_state["preset_label"])
    st.session_state["active_sid"] = picked["sid"]
    st.session_state["from_table"] = False
    st.session_state.pop("custom_prediction", None)


def case_banner(row):
    status = core.hf_status(row["health_factor"])
    badge = f'<span class="status" style="background:{status["color"]}16;color:{status["color"]}">{status["label"]} · HF {float(row["health_factor"]):.5g}</span>'
    st.markdown(f'<div class="case"><div class="case-title">{badge}{html.escape(row["version"])} · {html.escape(short_wallet(row["wallet"]))}</div><div class="case-meta">Cutoff {row["cutoff"]:%d/%m/%Y %H:%M} UTC · Block {int(row["state_block"]):,} · Nợ {money(row["total_debt_usdc_equivalent"])} USDC.e</div></div>', unsafe_allow_html=True)


def model_cards(row):
    results = [core.predict(row, name) for name in core.SCORE_COLUMNS]
    for col, result in zip(st.columns(3), results):
        with col:
            st.metric(result["model"], f'{result["score"]:.4f}', "CÓ CẢNH BÁO" if result["alert"] else "KHÔNG CẢNH BÁO", delta_color="off")
            st.caption(f'Ngưỡng validation đã khóa: {result["threshold"]:.4f}')
    st.caption("Các giá trị 0–1 là điểm xếp hạng của từng mô hình, chưa phải xác suất thanh lý đã hiệu chỉnh. Không so trực tiếp độ lớn điểm giữa các mô hình.")
    return results


def home(df, row):
    hero("NHÓM 12 · AAVE POLYGON", "Theo dõi HF. Kiểm chứng cảnh báo.",
         "Từ lịch sử giao dịch đến danh sách vị thế cần rà soát trong 7 ngày. Khám phá dữ liệu, chạy ba mô hình và xem các quyết định bằng 3D.")
    st.markdown('<span class="pill">BACKTEST 2024</span><span class="pill">V2 + V3 · POLYGON</span><span class="pill">DỮ LIỆU & MÔ HÌNH THẬT</span>', unsafe_allow_html=True)
    kpis([("Mẫu test còn nợ", "320.974", "Đơn vị: vị thế–tuần"), ("Mẫu có thanh lý", "80", "Trong 7 ngày sau cutoff"),
          ("XGBoost phát hiện", "10 / 80", "Tại ngưỡng đã khóa"), ("Precision của XGBoost", "3,33%", "10 đúng / 300 cảnh báo")])
    a, b = st.columns([1.1, 1])
    with a:
        st.subheader("Bắt đầu bằng một vị thế thật")
        case_banner(row)
        st.write("HF trả lời về trạng thái hiện tại. Mô hình dùng 34 biến hành vi để tạo điểm ưu tiên. Nhãn được kiểm tra bằng sự kiện trong bảy ngày sau đó.")
        st.button("Xem vị thế và Health Factor", type="primary", on_click=navigate, args=("Theo dõi HF",), width="stretch")
        st.button("Mở phòng mô hình 3D", on_click=navigate, args=("Mô hình 3D",), width="stretch")
    with b:
        st.subheader("HF thấp đi cùng quy mô nợ nào?")
        focus_hf = st.toggle("Xem gần ngưỡng HF (0–3)", value=True, key="home_hf_focus")
        cloud = charts.hf_debt_cloud(df)
        if focus_hf:
            ticks = np.array([0, 1, 1.3, 2, 3])
            cloud.update_xaxes(range=[0, np.log10(4)], tickvals=np.log10(1+ticks), ticktext=["0", "1", "1,3", "2", "3"])
        render_chart(cloud, width="stretch", key="home_cloud")
        st.caption("Mẫu đều 4.500 snapshot để vẽ. Bật xem gần ngưỡng chỉ hiển thị HF 0–3; tắt để xem toàn miền. Mốc 1,3 là quy ước demo; xanh không bảo đảm an toàn trong 7 ngày.")
    st.subheader("Một năm kiểm chứng theo từng tuần")
    render_chart(charts.weekly_chart(weekly_data()), width="stretch", key="home_weekly")
    st.info("Kết quả chính: mô hình được chọn theo validation vẫn không có HF trong đầu vào. Tín hiệu phụ thuộc mạnh vào tiền sử thanh lý; chưa có bằng chứng đủ để vận hành cảnh báo tổng quát.")


def hf_page(df, row):
    hero("01 / VỊ THẾ & TRẠNG THÁI", "Nhìn HF trước. Đọc cảnh báo sau.", "Tìm ví trong tập lịch sử, lọc theo vùng HF và mở từng vị thế. Đây là dữ liệu test 2024, không phải truy vấn số dư hiện tại.")
    case_banner(row)
    a, b = st.columns([1.1, 1])
    with a:
        st.subheader("Health Factor tại cutoff")
        render_chart(charts.hf_bar(row["health_factor"]), width="stretch", key="hf_bar")
        st.caption("Đỏ: HF < 1 · Vàng: 1 ≤ HF ≤ 1,3 · Xanh: HF > 1,3. Trên 3, con trỏ dừng ở mép thang; giá trị gốc vẫn được giữ.")
        st.write("HF < 1 là ranh giới đủ điều kiện thanh lý xét theo HF; nó khác sự kiện thanh lý được thực thi trong tương lai. Không coi màu xanh là cam kết an toàn.")
    with b:
        st.subheader("Mô phỏng biến động HF")
        shock = st.slider("Giá trị thế chấp thay đổi (%)", -50, 50, 0, 1, key="hf_shock")
        debt = st.slider("Dư nợ thay đổi (%)", -20, 100, 0, 1, key="hf_debt_change")
        new_hf = core.hf_scenario(row["health_factor"], shock, debt)
        status = core.hf_status(new_hf)
        st.metric("HF giả định", f"{new_hf:.4f}", status["label"], delta_color="off")
        st.caption("Giả định mọi thế chấp đổi cùng tỷ lệ, ngưỡng thanh lý giữ nguyên. Đây là phép tính độ nhạy, không dự báo giá hay giao dịch thật.")
        st.caption("Điểm của mô hình 34 biến không đổi chỉ vì kéo HF: HF và nợ không phải đầu vào của mô hình được chọn.")
    st.divider()
    st.subheader("Danh sách vị thế lịch sử")
    c1, c2, c3, c4 = st.columns([1.2, 1, 1.2, 1])
    search = c1.text_input("Tìm địa chỉ ví", placeholder="0x…", key="wallet_search").strip().lower()
    versions = c2.multiselect("Phiên bản", ["V2", "V3"], default=["V2", "V3"], key="filter_versions")
    band = c3.selectbox("Vùng HF", ["Tất cả", "Đỏ · HF < 1", "Vàng · 1–1,3", "Xanh · HF > 1,3"], key="hf_band")
    min_debt = c4.number_input("Nợ tối thiểu (USDC.e)", min_value=0., value=0., step=10., key="min_debt")
    dates = ["Tất cả"] + sorted(df.cutoff.dt.strftime("%Y-%m-%d").unique().tolist())
    c5, c6 = st.columns([1, 2])
    date = c5.selectbox("Tuần chấm (UTC)", dates, key="cutoff_filter")
    show_labels = c6.toggle("Hiện nhãn hậu kiểm và thời điểm thanh lý sau cutoff", key="show_future_label")
    mask = df.version.isin(versions) & df.total_debt_usdc_equivalent.ge(min_debt)
    if search:
        mask &= df.wallet.str.contains(search, case=False, regex=False, na=False)
    if date != "Tất cả":
        mask &= df.cutoff.eq(pd.Timestamp(date))
    if band.startswith("Đỏ"):
        mask &= df.health_factor.lt(1)
    elif band.startswith("Vàng"):
        mask &= df.health_factor.between(1, 1.3)
    elif band.startswith("Xanh"):
        mask &= df.health_factor.gt(1.3)
    filtered = df.loc[mask]
    st.caption(f"{num(len(filtered))} snapshot phù hợp · Hiển thị tối đa 300 dòng theo điểm XGBoost giảm dần. Chọn ô ở đầu dòng rồi bấm nút mở vị thế.")
    if filtered.empty:
        st.info("Không có vị thế phù hợp. Địa chỉ mới ngoài tập 2024 không được tự động truy vấn on-chain.")
        return
    shown = filtered.sort_values("score_xgb", ascending=False).head(300)
    view = shown[["version", "wallet", "cutoff", "health_factor", "total_debt_usdc_equivalent", "score_xgb"]].copy()
    view["Cảnh báo ML"] = shown.score_xgb.ge(core.summary()["models"]["XGBoost"]["threshold"]).map({True: "Có", False: "Không"})
    if show_labels:
        view["Thanh lý trong 7 ngày"] = shown.target.map({1: "Có quan sát", 0: "Không ghi nhận"})
        view["Thời điểm sự kiện kế tiếp"] = shown.next_liquidation
        st.caption("Cột hậu kiểm chỉ để đánh giá. Nó không được đưa vào 34 biến khi chấm điểm.")
    view = view.rename(columns={"version": "Pool", "wallet": "Ví", "cutoff": "Cutoff UTC", "health_factor": "HF", "total_debt_usdc_equivalent": "Nợ USDC.e", "score_xgb": "Điểm XGBoost"})
    event = st.dataframe(view.style.map(lambda v: f'color:{core.hf_status(v)["color"]};font-weight:600', subset=["HF"]), hide_index=True, width="stretch", height=390,
                         on_select="rerun", selection_mode="single-row", key="position_table",
                         column_config={"HF": st.column_config.NumberColumn(format="%.4f"), "Nợ USDC.e": st.column_config.NumberColumn(format="%.2f"), "Điểm XGBoost": st.column_config.NumberColumn(format="%.5f")})
    if event.selection.rows and event.selection.rows[0] < len(shown):
        picked = int(shown.iloc[event.selection.rows[0]].sid)
        st.button("Mở vị thế đã chọn và chấm ba mô hình", type="primary", on_click=navigate, args=("Chấm điểm vị thế", picked), key="open_position")
    st.download_button("Tải bảng đang hiển thị (CSV)", view.to_csv(index=False).encode("utf-8-sig"), "vi_the_dang_xem.csv", "text/csv")


def show_scored(row, hypothetical=False):
    results = model_cards(row)
    a, b = st.columns([1.1, 1])
    with a:
        st.subheader("Vì sao XGBoost chấm điểm này?")
        contrib, base, margin = core.xgb_contributions(row)
        render_chart(charts.waterfall(contrib, base, margin), width="stretch", key="score_shap_"+str(hypothetical))
        st.caption("TreeSHAP được tính trực tiếp từ booster đã lưu; 7 biến lớn nhất + phần còn lại cộng đủ margin. Đây là giải thích mô hình, không phải nguyên nhân thực tế. HF không có trong 34 biến.")
    with b:
        st.subheader("Đọc kết quả đúng phạm vi")
        if hypothetical:
            st.info("Tình huống giả định: không có nhãn tương lai để biết đúng/sai. Chấm điểm không xác minh một vị thế hiện tại ngoài blockchain.")
        else:
            outcome = "Có quan sát thanh lý trong 7 ngày" if int(row["target"]) else "Không ghi nhận thanh lý trong 7 ngày"
            st.write("**Hậu kiểm:** " + outcome)
            if int(row["target"]):
                st.write(f'**Sự kiện kế tiếp:** {row["next_liquidation"]:%d/%m/%Y %H:%M} UTC · cách cutoff {row["lead_days"]:.2f} ngày.')
            st.write(f'**Tiền sử 90 ngày:** {int(row["past_liquidation_count_90d"])} lần.')
            st.write("**Điểm đầu vào:** chỉ dùng dữ liệu trước hoặc đúng cutoff; nhãn tương lai chỉ xuất hiện ở phần hậu kiểm.")
        st.write("Một cảnh báo đúng không đồng nghĩa một vụ thanh lý đã được ngăn chặn. Kết quả hiện tại chủ yếu phát hiện tái thanh lý.")
        packet = {"kind": "hypothetical" if hypothetical else "historical_snapshot", "features": {f: row[f] for f in core.summary()["features"]}, "results": results, "HF_is_model_input": False}
        st.download_button("Tải đầu vào & kết quả chấm (JSON)", json.dumps(packet, ensure_ascii=False, indent=2, default=serial), "ket_qua_cham.json", "application/json", key="download_score_"+str(hypothetical))


def scoring_page(df, row):
    hero("02 / CHẠY SUY LUẬN THẬT", "Một vị thế. Ba góc nhìn.", "Chấm trực tiếp các pipeline đã huấn luyện. Mô hình, bộ biến và ngưỡng được giữ theo thực nghiệm; không huấn luyện lại khi bấm nút.")
    case_banner(row)
    mode = st.radio("Cách nhập dữ liệu", ["Ca lịch sử thật", "Giả định / chỉnh sửa biến", "Chấm CSV"], horizontal=True, key="input_mode")
    if mode == "Ca lịch sử thật":
        show_scored(row)
        with st.expander("Xem đủ 34 đầu vào tại cutoff"):
            st.dataframe(pd.DataFrame({"Biến": [core.label(f) for f in core.summary()["features"]], "Giá trị": [row[f] for f in core.summary()["features"]]}), hide_index=True, width="stretch")
    elif mode == "Giả định / chỉnh sửa biến":
        st.info("Đây là mô phỏng từ ca nền, không phải số dư hoặc lịch sử mới được thu thập. Bốn tỷ số/chỉ số nhịp được tự tính lại từ số đếm.")
        editable = [f for f in core.summary()["features"] if f not in core.DERIVED]
        frame = pd.DataFrame({"Biến": [core.label(f) for f in editable], "Giá trị": [float(row[f]) for f in editable]}, index=editable)
        with st.form("scenario_form"):
            edited = st.data_editor(frame, disabled=["Biến"], hide_index=True, height=460, width="stretch", key=f"editor_{row['sid']}",
                                    column_config={"Giá trị": st.column_config.NumberColumn(required=True, format="%.5f")})
            submitted = st.form_submit_button("Chấm tình huống giả định", type="primary")
        if submitted:
            candidate = dict(row)
            candidate.update({f: float(edited.loc[f, "Giá trị"]) for f in editable})
            try:
                candidate = core.recompute_derived(candidate)
                errors = core.validate_features(candidate)
            except (ZeroDivisionError, ValueError, TypeError):
                errors = ["Các số đếm làm mẫu số không hợp lệ. Kiểm tra cửa sổ 7/30 ngày."]
            if errors:
                for error in errors:
                    st.error(error)
                st.session_state.pop("custom_prediction", None)
            else:
                st.session_state["custom_prediction"] = candidate
        if "custom_prediction" in st.session_state:
            show_scored(st.session_state["custom_prediction"], hypothetical=True)
    else:
        st.write("CSV cần đủ 34 cột feature, tối đa 5.000 dòng. Có thể dùng tệp mẫu từ các ca thật của demo; HF/nợ là metadata, không đưa vào mô hình history34.")
        st.download_button("Tải CSV đầu vào mẫu", (ROOT / "data/example_input.csv").read_bytes(), "aave_input_mau.csv", "text/csv", key="sample_csv")
        upload = st.file_uploader("Chọn CSV để chấm", type=["csv"], key="batch_upload")
        if upload and st.button("Chạy ba mô hình trên CSV", type="primary"):
            try:
                incoming = pd.read_csv(upload)
                result = core.predict_frame(incoming)
                st.session_state["batch_prediction"] = result
            except (ValueError, TypeError, KeyError, pd.errors.ParserError) as exc:
                st.session_state.pop("batch_prediction", None)
                st.error(str(exc))
        if "batch_prediction" in st.session_state:
            result = st.session_state["batch_prediction"]
            st.success(f"Đã chấm {len(result)} dòng. Không dùng cột nhãn tương lai nếu CSV có chứa.")
            st.dataframe(result, hide_index=True, width="stretch")
            st.download_button("Tải kết quả CSV", result.to_csv(index=False).encode("utf-8-sig"), "aave_ket_qua.csv", "text/csv", key="batch_result")


def lab_page(df, row):
    hero("03 / PHÒNG MÔ HÌNH 3D", "Xoay mô hình. Theo từng quyết định.", "Các bề mặt, nhánh cây và điểm ở đây lấy từ mô hình thật. Kéo để xoay, cuộn để phóng to; đọc bảng bên dưới để nối hình ảnh với phép tính.")
    case_banner(row)
    name = st.radio("Chọn thuật toán", list(core.SCORE_COLUMNS), horizontal=True, key="lab_model")
    view_col, reset_col = st.columns([3, 1], vertical_alignment="bottom")
    with view_col:
        st.segmented_control("Góc nhìn 3D", ["Phối cảnh", "Chính diện", "Từ trên"], default="Phối cảnh", key="lab_camera")
    with reset_col:
        if st.button("Đặt lại góc nhìn", width="stretch", key="reset_camera"):
            st.session_state["camera_revision"] = st.session_state.get("camera_revision", 0) + 1
    st.markdown('<div class="chart-guide"><strong>Kéo</strong> để xoay · <strong>Cuộn trong hình</strong> để phóng to · <strong>Rê chuột</strong> để đọc giá trị · Dùng nút góc nhìn nếu khó quan sát. Mọi hình đều lấy từ mô hình đã lưu.</div>', unsafe_allow_html=True)
    if name == "Logistic":
        a, b = st.columns([1.8, 1])
        features = core.summary()["features"]
        with b:
            st.subheader("Tổ hợp tuyến tính → sigmoid")
            axes = ["borrow_count_30d", "repay_count_30d", "past_liquidation_count_30d", "past_liquidation_count_90d", "borrower_history_days", "days_since_last_borrow", "active_days_30d", "activity_count_30d"]
            xname = st.selectbox("Trục X", axes, index=0, format_func=core.label, key="lr_x")
            yoptions = [f for f in axes if f != xname]
            yname = st.selectbox("Trục Y", yoptions, index=yoptions.index("past_liquidation_count_90d") if "past_liquidation_count_90d" in yoptions else 0, format_func=core.label, key="lr_y")
            x_max = max(5., min(float(df[xname].quantile(.99)), 5000.), float(row[xname])*1.1)
            y_max = max(5., min(float(df[yname].quantile(.99)), 5000.), float(row[yname])*1.1)
            st.latex(r"z=\beta_0+\sum_j\beta_j\widetilde{x}_j,\qquad s=\frac{1}{1+e^{-z}}")
            obj = core.load_model(name)
            X = np.array([[row[f] for f in features]], np.float32)
            imp = obj["pipeline"].named_steps["impute"].transform(X)
            scaled = obj["pipeline"].named_steps["scale"].transform(imp)
            clf = obj["pipeline"].named_steps["clf"]
            z = float(clf.decision_function(scaled)[0])
            st.metric("z của ca đang xem", f"{z:.4f}")
            st.metric("Điểm sau sigmoid", f"{1/(1+np.exp(-z)):.5f}")
            st.caption("Bề mặt chỉ thay hai cột, giữ 32 cột khác ở ca nền. Đây là minh họa độ nhạy của hàm, không phải vị thế giả định đã được kiểm tra nhất quán hay mô phỏng nhân quả.")
        with a:
            render_chart(charts.logistic_surface(row, xname, yname, x_max, y_max), width="stretch", key="lr_surface")
        contrib = clf.coef_[0]*scaled[0]
        top = pd.DataFrame({"Biến": [core.label(f) for f in features], "Đóng góp β × x chuẩn hóa": contrib}).iloc[np.argsort(np.abs(contrib))[::-1][:8]]
        st.dataframe(top, hide_index=True, width="stretch")
        st.caption("C = 0,1 · chuẩn hóa học từ train · phạt L2. Điểm 0–1 chưa phải xác suất thực tế đã kiểm chứng.")
    elif name == "Random Forest":
        a, b = st.columns([1.8, 1])
        with b:
            st.subheader("Một cây trong rừng 300 cây")
            tree_index = st.slider("Chọn cây", 1, 300, 1, key="rf_tree") - 1
            depth = st.slider("Hiển thị đến độ sâu", 2, 6, 4, key="rf_depth")
            st.write("Màu vàng là đường đi của ca đang xem. Rê chuột vào nút để thấy biến, ngưỡng chia, số mẫu và điểm tại nút.")
            st.caption("Trái: giá trị ≤ ngưỡng. Phải: giá trị > ngưỡng. Cắt hình ở độ sâu chọn chỉ để dễ xem; cây thật vẫn được chấm đến lá.")
        fig, steps, info = charts.tree3d(row, tree_index, depth)
        with a:
            render_chart(fig, width="stretch", key="rf_tree3d")
        kpis([("Điểm cây đang xem", f'{info["tree_score"]:.4f}', f'Lá {info["leaf"]}'),
              ("Điểm cả rừng", f'{info["forest_score"]:.4f}', "Trung bình của 300 cây"),
              ("Độ sâu thật của cây", str(info["actual_depth"]), f'{info["nodes"]} nút thật')])
        st.subheader("Theo dấu đường đi đến lá")
        st.dataframe(steps, hide_index=True, width="stretch")
        st.caption("RF_300_d12: 300 cây, sâu tối đa 12, tối thiểu 30 mẫu/lá, sqrt(34) ≈ 5 biến ứng viên/nút, tối đa 100.000 lượt bootstrap/cây. Các ngưỡng trong bảng là ngưỡng thật đã học.")
    else:
        a, b = st.columns([1.8, 1])
        with b:
            st.subheader("Các cây cộng dồn thành điểm")
            n_trees = st.slider("Xem các cây đầu của mô hình 400 cây", 1, 400, 400, key="xgb_trees")
            st.latex(r"z_M=b+\sum_{m=1}^{M}\Delta_m,\qquad s=\sigma(z_M)")
            st.write("Mỗi cây thêm một cập nhật đã được thu nhỏ bởi learning rate 0,05. Cây sâu tối đa 3; số cây và độ sâu là hai tham số khác nhau.")
            st.caption("Đây là các cây đầu của mô hình 400 cây đã fit, không phải huấn luyện một mô hình mới có M cây. Giảm M không thay mô hình được chọn trong báo cáo.")
        sample = pd.concat([df[df.target.eq(0)].sample(4000, random_state=42), df[df.target.eq(1)]]).copy()
        with a:
            render_chart(charts.xgb_cloud(sample, n_trees), width="stretch", key="xgb_cloud")
        st.caption("Hình 3D gồm 4.000 âm lấy mẫu đều và toàn bộ 80 dương. Mật độ không đại diện tỷ lệ của toàn tập; cả 34 biến vẫn được dùng để tính điểm, không chỉ hai trục đang vẽ.")
        st.subheader("Điểm ca đang xem khi cộng thêm cây")
        render_chart(charts.boosting_path(row), width="stretch", key="boosting_path")
        st.caption("Ngưỡng khóa chỉ được chọn cho cấu hình đủ 400 cây. Đường nét đứt trên các giai đoạn chỉ là mốc tham chiếu.")


def results_page(df, row):
    hero("04 / BẰNG CHỨNG THỰC NGHIỆM", "Được chọn chưa có nghĩa là đủ tốt.", "Đối chiếu toàn bộ test còn nợ. Thay ngưỡng để hiểu đánh đổi giữa cảnh báo sai và bỏ sót; AP vẫn đo cùng một thứ tự xếp hạng.")
    model = st.selectbox("Mô hình để kiểm tra", list(core.SCORE_COLUMNS), index=2, key="metric_model")
    locked = st.checkbox("Dùng ngưỡng báo cáo đã khóa từ validation", value=True, key="locked_threshold")
    original = core.summary()["models"][model]["threshold"]
    threshold = st.slider("Ngưỡng điểm cảnh báo", 0., 1., float(original), .001, disabled=locked, key=f"metric_threshold_{model}")
    threshold = original if locked else threshold
    m = core.metrics(df.target, df[core.SCORE_COLUMNS[model]], threshold)
    if not locked:
        st.info("Đang khám phá một ngưỡng hậu kiểm trên test. Thao tác này không cập nhật mô hình, ngưỡng khóa hoặc kết quả báo cáo.")
    kpis([("Precision", f'{100*m["precision"]:.2f}%', "Đúng / tổng cảnh báo"), ("Recall", f'{100*m["recall"]:.2f}%', "Tìm được / 80 dương"),
          ("F₂", f'{m["F2"]:.5f}', "Thước đo nhấn recall"), ("Cảnh báo", num(m["alerts"]), f'{m["tp"]} đúng · {num(m["fp"])} sai')])
    a, b = st.columns(2)
    with a:
        st.subheader("Ma trận nhầm lẫn")
        z = [[m["tn"], m["fp"]], [m["fn"], m["tp"]]]
        texts = [[f'TN · {num(m["tn"])}', f'FP · {num(m["fp"])}'], [f'FN · {num(m["fn"])}', f'TP · {num(m["tp"])}']]
        fig = go.Figure(go.Heatmap(z=np.log1p(z), text=texts, texttemplate="%{text}", x=["Không cảnh báo", "Cảnh báo"], y=["Nhãn 0", "Nhãn 1"], colorscale=[[0, "#202A37"], [.45, "#2C6471"], [1, "#66D5C8"]], textfont=dict(size=17, color="#F3F7FC"), xgap=6, ygap=6, showscale=False, hovertemplate="%{text}<extra></extra>"))
        fig.update_yaxes(autorange="reversed", showgrid=False)
        fig.update_xaxes(showgrid=False, side="bottom")
        render_chart(charts.polish(fig, 325), width="stretch", key="confusion")
        st.caption("Màu theo log(1 + số mẫu) để nhìn được lớp hiếm; số hiển thị là số đếm thật.")
    with b:
        st.subheader(f'Precision–Recall · AP {m["AP"]:.5f}')
        focus_pr = st.toggle("Phóng vùng precision thấp (0–10%)", value=True, key="pr_focus")
        p, r, _, _ = pr_roc(model)
        fig = go.Figure(go.Scatter(x=r, y=p, mode="lines", line=dict(color=charts.TEAL, width=3), name=model))
        fig.add_trace(go.Scatter(x=[m["recall"]], y=[m["precision"]], mode="markers", marker=dict(color=charts.AMBER, size=11), name="Ngưỡng đang xem"))
        fig.update_xaxes(title="Recall", range=[0, 1])
        fig.add_hline(y=float(df.target.mean()), line_dash="dot", line_color=charts.MUTED,
                      annotation_text="Tỷ lệ dương nền", annotation_position="top right", annotation_yshift=7, annotation_font=dict(size=10, color=charts.MUTED))
        fig.update_yaxes(title="Precision", range=[0, .1 if focus_pr else 1], tickformat=".0%")
        fig.update_xaxes(tickformat=".0%")
        render_chart(charts.polish(fig, 325), width="stretch", key="precision_recall")
        st.caption("AP tính trên đủ 320.974 dòng; đường cong rút gọn điểm để hiển thị. Điểm vàng là precision/recall tại ngưỡng đang xem. Khi bật phóng vùng, phần precision trên 10% nằm ngoài khung; tắt để xem toàn đường cong.")
    st.subheader("Cùng 34 biến, cùng cohort, cấu hình chọn theo validation")
    family_rows = []
    for name, meta in core.summary()["models"].items():
        t = meta["test"]
        family_rows.append({"Mô hình": name, "AP validation": meta["validation_AP"], "AP test": t["AP"], "Precision": t["precision"], "Recall": t["recall"], "F₂": t["F2"], "Cảnh báo": t["alerts"]})
    st.dataframe(pd.DataFrame(family_rows), hide_index=True, width="stretch", column_config={"Precision": st.column_config.NumberColumn(format="percent"), "Recall": st.column_config.NumberColumn(format="percent")})
    st.subheader("So với quy tắc đơn giản thì sao?")
    compare = pd.read_csv(ROOT / "data/hf_model_comparison.csv")
    compare = compare[compare.name.isin(["history34__XGB_400_d3", "recurrence90", "HF_rank"])][["name", "test_AP", "test_tp", "test_fp", "test_alerts", "test_F2"]]
    compare.columns = ["Phương án", "AP test", "TP", "FP", "Cảnh báo", "F₂"]
    compare["Phương án"] = compare["Phương án"].replace({"history34__XGB_400_d3": "XGBoost 400/sâu 3", "recurrence90": "Tiền sử thanh lý 90 ngày", "HF_rank": "Xếp hạng −HF"})
    st.dataframe(compare, hide_index=True, width="stretch")
    st.write("**Cách đọc:** XGBoost có AP nhỉnh hơn quy tắc 90 ngày, nhưng quy tắc tìm 24/80 mẫu thay vì 10/80 và có F₂ cao hơn; hai tải cảnh báo khác nhau. Không dùng một chỉ số để tuyên bố mô hình tốt nhất cho vận hành.")
    with st.expander("Cùng ngân sách 0,5% / 1% / 5% mỗi tuần"):
        st.dataframe(pd.read_csv(ROOT / "data/hf_fixed_budget_comparison.csv"), hide_index=True, width="stretch")
        st.write("Ở top 1%, XGBoost tìm 19 ca và HF tìm 3; ở top 5%, HF tìm 52 còn XGBoost tìm 29. Ưu thế phụ thuộc ngân sách.")
    with st.expander("Thiết kế dữ liệu, tiền sử và giới hạn cần trình bày"):
        st.markdown("""
- Train 2021–2022; validation 2023; test hồi cứu 2024. Purge cửa sổ nhãn bảy ngày chạm ranh giới tập.
- Train còn nợ sau lấy mẫu: 131.272 / 511 dương; validation: 552.742 / 30; test: 320.974 / 80.
- 5 bộ biến × (2 Logistic + 3 RF + 3 XGBoost) = 40 cấu hình. Bản demo giữ ba đại diện cùng history34.
- Cả 10 ca đúng ở ngưỡng khóa có tiền sử thanh lý 90 ngày, thuộc 7 ví; 9 có HF > 1 tại cutoff.
- 7.593/7.991 mẫu HF < 1 có nợ dưới 1 USDC.e. Quy tắc nợ ≥ 10 rồi xét HF là hậu kiểm, không thay lựa chọn khóa.
- HF/nợ/giá đã được bổ sung để thử nghiệm nhưng mô hình được chọn không sử dụng chúng. Chưa chứng minh giá trị tăng thêm ổn định.
- Test 2024 đã từng được xem; nhãn có giới hạn bao phủ. Điểm không phải xác suất đã hiệu chỉnh; SHAP không chứng minh nhân quả; TP không phải số vụ ngăn được.
""")


def demo_page():
    hero("05 / KỊCH BẢN THUYẾT TRÌNH", "Một đường demo để thầy theo dõi được.", "Dùng các ca thật, để từng thao tác trả lời một câu hỏi. Khoảng 6–8 phút cho bản demo đầy đủ.")
    steps = [
        ("1 · Bài toán, 45 giây", "Tổng quan", "Chúng em kiểm tra dự báo sự kiện thanh lý trong bảy ngày. Dữ liệu là lịch sử 2024; HF hiện tại, điểm mô hình và nhãn tương lai là ba đại lượng khác nhau."),
        ("2 · HF và ca thật, 60 giây", "Theo dõi HF", "Chọn ca HF đỏ/nợ nhỏ rồi ca xanh. Kéo biến động thế chấp để thấy HF thay đổi. Mốc vàng 1–1,3 là quy ước theo dõi, không bảo đảm an toàn."),
        ("3 · Chạy ba mô hình, 60 giây", "Chấm điểm vị thế", "Mở ca cảnh báo đúng. Điểm được tính trực tiếp từ pipeline. Đọc ngưỡng riêng từng mô hình và SHAP; HF không nằm trong 34 biến."),
        ("4 · Ba mô hình 3D, 2 phút", "Mô hình 3D", "Logistic: hai trục trên bề mặt sigmoid. RF: đổi cây, đọc nhánh vàng và lá thật. XGBoost: kéo số cây đầu, quan sát điểm phân tán và đường cộng dồn."),
        ("5 · Kiểm chứng và phản biện, 90 giây", "Đối chiếu kết quả", "Đọc 10 TP, 290 FP, 70 FN; so quy tắc 90 ngày. Bỏ chọn ngưỡng khóa và thử cắt khác để thấy precision/recall đổi nhưng AP không đổi."),
        ("6 · Kết luận, 30 giây", "Đối chiếu kết quả", "Nhóm đã xây quy trình và kiểm tra tín hiệu; mô hình phụ thuộc tiền sử. Cần dữ liệu đầy đủ hơn và kiểm chứng thời gian mới trước khi vận hành."),
    ]
    for title, page, text in steps:
        st.subheader(title)
        a, b = st.columns([4, 1])
        a.write(text)
        b.button("Mở màn hình", on_click=navigate, args=(page,), key="demo_"+title, width="stretch")
    st.divider()
    st.subheader("Câu trả lời ngắn khi thầy hỏi")
    with st.expander("Vì sao kéo HF mà điểm XGBoost không đổi?"):
        st.write("Vì mô hình được chọn sử dụng 34 biến hành vi và tiền sử. HF là trạng thái/đối chứng riêng, không được tự thêm vào đầu vào mô hình đã fit.")
    with st.expander("Cây 3D có phải hình minh họa tự vẽ không?"):
        st.write("Đồ thị lấy cấu trúc, ngưỡng, số mẫu và đường đi từ chính cây RF đã fit. XGBoost cũng chạy booster đã lưu, không tạo điểm ngẫu nhiên.")
    with st.expander("Nhập một địa chỉ ví mới thì dự báo ngay được không?"):
        st.write("Chưa. Phải có lịch sử và trạng thái phù hợp để tạo đủ feature. App tra cứu các snapshot lịch sử đã chuẩn bị; CSV/tình huống tự nhập là suy luận trên đầu vào cung cấp, không xác minh on-chain trực tiếp.")
    st.download_button("Tải kịch bản demo (Markdown)", (ROOT / "KICH_BAN_DEMO.md").read_bytes(), "KICH_BAN_DEMO.md", "text/markdown")


PRESETS = json.loads((ROOT / "data/presets.json").read_text(encoding="utf-8"))
st.session_state.setdefault("active_sid", PRESETS[0]["sid"])
with st.sidebar:
    st.markdown('<div class="brand">◈ AAVE / POLYGON<small>NHÓM 12 · MACHINE LEARNING LAB</small></div>', unsafe_allow_html=True)
    st.caption("Dữ liệu lịch sử · kiểm chứng trong 7 ngày")
    st.divider()
    page = st.radio("Đi đến", NAV, key="nav", label_visibility="collapsed")
    st.divider()
    st.selectbox("Ca minh họa nhanh", [p["label"] for p in PRESETS], key="preset_label", on_change=use_preset)
    st.caption("Đang xem vị thế bạn mở từ bảng. Đổi ca minh họa để quay về ca dựng sẵn." if st.session_state.get("from_table") else "Ca được chọn dùng chung cho HF, chấm điểm và mô hình 3D.")
    st.divider()
    st.caption("XGBoost 400 cây / sâu 3\n\n34 biến hành vi · không có HF\n\nMô hình & ngưỡng giữ nguyên từ thực nghiệm.")

DATA = dataset()
ROW = DATA.loc[int(st.session_state["active_sid"])].to_dict()
if page == "Tổng quan":
    home(DATA, ROW)
elif page == "Theo dõi HF":
    hf_page(DATA, ROW)
elif page == "Chấm điểm vị thế":
    scoring_page(DATA, ROW)
elif page == "Mô hình 3D":
    lab_page(DATA, ROW)
elif page == "Đối chiếu kết quả":
    results_page(DATA, ROW)
else:
    demo_page()
st.divider()
st.caption("NHÓM 12 · Aave V2/V3 Polygon · Backtest 2024. Mỗi dòng là một snapshot phiên bản–ví–tuần. Dữ liệu demo không phải vị thế hiện tại; cảnh báo đúng chưa đồng nghĩa ngăn được thanh lý.")
