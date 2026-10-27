"""A guided borrower-to-decision demonstration for a nontechnical presentation."""
import html
from pathlib import Path
from typing import Callable

import pandas as pd
import streamlit as st

import aave_core as core
import charts
import showcase_figures as figures
from viz3d import MODEL_COLORS

ROOT = Path(__file__).resolve().parent


def _reset_inputs(sid: int, count: int, age: float) -> None:
    st.session_state[f"lesson_k_{sid}"] = count
    st.session_state[f"lesson_h_{sid}"] = age


def _score_cards(point: dict, active_model: str) -> None:
    label = "Ca thật tại thời điểm chấm" if point["is_original"] else "Đầu vào đang thử · không có nhãn tương lai"
    st.caption(label + ". Mỗi mô hình dùng ngưỡng riêng; không so điểm như xác suất.")
    for column, (name, result) in zip(st.columns(3), point["results"].items()):
        decision = "CẢNH BÁO" if result["alert"] else "CHƯA CẢNH BÁO"
        color = MODEL_COLORS[name]
        active = " selected" if name == active_model else ""
        column.markdown(
            f'<div class="lesson-model{active}" style="--model-color:{color}">'
            f'<div class="lesson-model-name">{html.escape(name)}</div>'
            f'<div class="lesson-model-score">{result["score"]:.3f}</div>'
            f'<div class="lesson-model-rule">Ngưỡng {result["threshold"]:.3f} · {decision}</div></div>',
            unsafe_allow_html=True)


def explore(row: dict, response: dict, render_chart: Callable, navigate: Callable) -> None:
    """Follow one set of inputs through all scores, one response plot, and its mechanism."""
    if not response["available"]:
        st.info("Ca này có quá ít miền tuổi lịch sử hợp lệ để vẽ bề mặt; chọn một ca khác ở thanh bên.")
        return
    cards = st.container()
    name = st.segmented_control("Chọn mô hình để giải thích", list(core.SCORE_COLUMNS), default="Logistic",
                                required=True, key="showcase_model")
    left, right = st.columns([1, 2.5], gap="large")
    with left:
        st.markdown("**Thử cùng một vị thế**")
        sid = int(row["sid"])
        original_count = int(row["past_liquidation_count_30d"])
        original_age = float(row["borrower_history_days"])
        count_key, age_key = f"lesson_k_{sid}", f"lesson_h_{sid}"
        st.session_state.setdefault(count_key, original_count)
        st.session_state.setdefault(age_key, original_age)
        st.button("Trở về ca thật", key="lesson_reset", width="stretch", on_click=_reset_inputs,
                  args=(sid, original_count, original_age))
        count = st.selectbox("Thanh lý trong 30 ngày trước", response["ks"].tolist(),
                             format_func=lambda k: f"{k} lần", key=count_key)
        age = st.slider("Lịch sử vay quan sát (ngày)", float(response["ages"].min()),
                        float(response["ages"].max()), step=.1, format="%.1f", key=age_key)
        earlier = int(row["past_liquidation_count_90d"] - original_count)
        st.caption(f"Giữ {earlier} lần ở ngày 31–90. Tổng 90 ngày tự thành {count+earlier} lần. "
                   f"{len(core.summary()['features'])-3} đầu vào khác giữ nguyên.")
        point = core.lesson_point(row, int(count), float(age))
        result = point["results"][name]
        sign = "≥" if result["alert"] else "<"
        decision = "cảnh báo" if result["alert"] else "chưa cảnh báo"
        st.markdown(f'**{name}: {decision}**  \nĐiểm {result["score"]:.4f} {sign} ngưỡng {result["threshold"]:.4f}.')
        st.caption("Đây là thử đầu vào của mô hình, không dự báo tác động của hành động ngoài đời.")
    with cards:
        _score_cards(point, name)
    with right:
        st.markdown("**Đổi đầu vào, quan sát điểm có vượt ngưỡng không**")
        render_chart(figures.borrower_surface(row, response, name, point), key="borrower_surface")
        st.caption("Ngang: số lần thanh lý gần đây · chiều sâu: số ngày lịch sử vay · chiều cao: điểm. "
                   "Mặt phẳng vàng là ngưỡng; trắng là ca thật, tím là đầu vào đang thử.")
    _explain_mechanism(name, point, render_chart, navigate)
    with st.expander("Vì sao chọn hai đầu vào này, và hình cho phép kết luận gì?"):
        st.write("Tuổi lịch sử vay và số lần thanh lý 30 ngày có mean |SHAP| lớn nhất trên các ca dương của "
                 "XGBoost đã phân tích. Cả ba mô hình đều nhận những cột này. Ta dùng cùng hai trục để nhìn "
                 "khác biệt phản ứng, nhưng so mỗi điểm với ngưỡng riêng của mô hình đó.")
        st.write("Mỗi hàng lưới qua kiểm tra ràng buộc. Số lần 90 ngày được cập nhật cùng số lần 30 ngày; "
                 "các cột khác giữ nguyên. Kiểm tra này không xác nhận kịch bản đã từng tồn tại. Các dải nối "
                 "minh họa giữa mức tuổi đã chấm; điểm trắng/tím luôn được tính trực tiếp bằng pipeline.")
        st.write("HF và nợ của ca nền không nằm trong 34 đầu vào. Điểm là chỉ số xếp hạng của mô hình, "
                 "không phải xác suất thanh lý đã hiệu chỉnh hoặc bằng chứng về nguyên nhân.")


def _explain_mechanism(name: str, point: dict, render_chart: Callable, navigate: Callable) -> None:
    values = point["features"]
    result = point["results"][name]
    feature_count = len(core.summary()["features"])
    if name == "Logistic":
        details = core.logistic_diagnostics(values)
        stages = [f"{feature_count} chỉ số hành vi", "Chuẩn hóa, cộng theo trọng số",
                  f'Tổng = {details["margin"]:.3f}', f'Điểm {result["score"]:.3f}']
        sentence = "Logistic cộng các tín hiệu theo trọng số, rồi dùng sigmoid để đưa tổng về thang điểm từ 0 đến 1."
    elif name == "Random Forest":
        details = core.forest_diagnostics(values)
        stages = [f'{details["tree_count"]} cây đã học', "Mỗi cây chấm cùng đầu vào", "Lấy trung bình các điểm",
                  f'Điểm {result["score"]:.3f}']
        sentence = "Random Forest hỏi nhiều cây rồi lấy trung bình điểm; một cây không đại diện cho toàn rừng."
    else:
        obj = core.load_model(name)
        count = obj["pipeline"].named_steps["clf"].get_booster().num_boosted_rounds()
        stages = [f"{count} cây đã học", "Cộng dần các điều chỉnh", "Chuyển tổng thành điểm",
                  f'Điểm {result["score"]:.3f}']
        sentence = "XGBoost cộng các cập nhật của những cây kế tiếp vào kết quả trước đó; các cây không chấm độc lập như RF."
    st.markdown("**Mô hình đang làm gì?**")
    st.write(sentence)
    st.markdown('<div class="lesson-flow">'+"".join(f'<div>{html.escape(s)}</div>' for s in stages)+"</div>",
                unsafe_allow_html=True)
    with st.expander("Mở số liệu để giải thích cơ chế"):
        if name == "Logistic":
            render_chart(charts.lr_contribution_bar(details), key="lesson_lr_mechanism")
            st.caption("Các thanh cộng đủ z. Biến tương quan không cho phép diễn giải từng hệ số riêng lẻ như nguyên nhân.")
        elif name == "Random Forest":
            above = int((details["scores"] >= details["threshold"]).sum())
            st.write(f'{above}/{details["tree_count"]} cây chấm trên ngưỡng. Quyết định cuối dùng trung bình '
                     f'{details["score"]:.4f}, không dùng đa số cây bỏ phiếu.')
            render_chart(charts.rf_vote_histogram(details), key="lesson_rf_mechanism")
            if st.toggle("Xem đường đi thật của cây thứ nhất", key="showcase_rf_path"):
                st.dataframe(details["steps"], hide_index=True, width="stretch")
                st.caption("Đây là một cây để minh họa; quyết định của rừng vẫn dùng trung bình tất cả cây.")
        else:
            render_chart(charts.boosting_path(values), key="lesson_xgb_mechanism")
            st.caption("Đây là các giai đoạn của một booster đã lưu. Ngưỡng chỉ được khóa cho cấu hình đủ số cây.")
            if point["is_original"]:
                st.button("Xem SHAP của ca thật", on_click=navigate, args=("Chấm điểm vị thế",), key="lesson_to_shap")
    with st.expander("Gợi ý nói với giảng viên trong 30 giây"):
        decision = "đạt ngưỡng nên cảnh báo" if result["alert"] else "chưa đạt ngưỡng nên chưa cảnh báo"
        st.write(f'“Đây là cùng một vị thế nền. {sentence} Với đầu vào đang xem, điểm là '
                 f'{result["score"]:.3f}, ngưỡng là {result["threshold"]:.3f}; mô hình {decision}. '
                 'Em thử đổi số lần thanh lý gần đây hoặc số ngày lịch sử để thấy hàm đã học phản ứng ra sao. '
                 'Điều đó chưa chứng minh thay đổi hành vi sẽ làm vị thế được an toàn.”')


def comparison_frame() -> pd.DataFrame:
    """Read all frozen outcomes from the report artifacts, retaining the simple rule baseline."""
    rows = []
    for name, meta in core.summary()["models"].items():
        test = meta["test"]
        rows.append({"Phương án": name, "Phát hiện đúng": test["tp"], "Cảnh báo": test["alerts"],
                     "Cảnh báo sai": test["fp"], "AP test": test["AP"]})
    rule = pd.read_csv(ROOT/"data/hf_model_comparison.csv")
    rule = rule.loc[rule.name.eq("recurrence90")].iloc[0]
    rows.append({"Phương án": "Quy tắc tiền sử 90 ngày", "Phát hiện đúng": int(rule.test_tp),
                 "Cảnh báo": int(rule.test_alerts), "Cảnh báo sai": int(rule.test_fp), "AP test": rule.test_AP})
    return pd.DataFrame(rows)


def validate_story(df: pd.DataFrame, row: dict, render_chart: Callable, navigate: Callable) -> None:
    st.subheader("Với ca thật, sau thời điểm chấm đã xảy ra điều gì?")
    if int(row["target"]):
        st.write(f'Ca nền có HF {row["health_factor"]:.4f} tại cutoff; ghi nhận thanh lý sau '
                 f'{row["lead_days"]:.2f} ngày, lúc {row["next_liquidation"]:%d/%m/%Y %H:%M} UTC.')
    else:
        st.write("Không ghi nhận sự kiện thanh lý trong 7 ngày sau cutoff của ca nền này.")
    case_rows = []
    for name, column in core.SCORE_COLUMNS.items():
        alert = row[column] >= core.summary()["models"][name]["threshold"]
        correct = bool(alert) == bool(row["target"])
        outcome = ("Cảnh báo đúng" if correct else "Cảnh báo sai") if alert else ("Không báo, không ghi nhận" if correct else "Bỏ sót")
        case_rows.append({"Mô hình": name, "Điểm ca thật": row[column],
                          "Ngưỡng": core.summary()["models"][name]["threshold"], "Hậu kiểm": outcome})
    st.dataframe(pd.DataFrame(case_rows), hide_index=True, width="stretch")
    st.caption("Bảng này chỉ xét ca nền thật; các đầu vào đang thử ở tab trước không có nhãn tương lai tương ứng.")
    positives = int(df.target.sum())
    st.subheader(f"Trên toàn bộ test, phát hiện bao nhiêu trong {positives} mẫu có thanh lý?")
    comparison = comparison_frame()
    render_chart(figures.detection_chart(comparison, positives), key="lesson_test_detection")
    st.dataframe(comparison, hide_index=True, width="stretch")
    st.write("Các phương án dùng ngưỡng riêng nên tải cảnh báo khác nhau. XGBoost được chọn theo validation; "
             "chưa thể gọi là tốt nhất cho vận hành khi quy tắc đơn giản vẫn phát hiện nhiều mẫu hơn. "
             "Cảnh báo đúng trong backtest chưa đồng nghĩa ngăn được thanh lý.")
    st.button("Mở đầy đủ tiêu chí đánh giá", on_click=navigate, args=("Đối chiếu kết quả",), key="lesson_to_results")
