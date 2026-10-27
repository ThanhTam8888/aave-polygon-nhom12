"""Presentation of common model ranks, valid sensitivity rows and exact fitted mechanisms."""
from typing import Callable

import pandas as pd
import streamlit as st

import aave_core as core
import viz3d
import showcase


@st.cache_resource
def rank_scales(_df: pd.DataFrame) -> dict:
    return {name: core.RankScale(_df[column].to_numpy()) for name, column in core.SCORE_COLUMNS.items()}


@st.cache_data
def rank_points(_df: pd.DataFrame, _scales: dict) -> pd.DataFrame:
    return viz3d.rank_space_points(_df, _scales)


@st.cache_data
def rank_tables(_df: pd.DataFrame) -> tuple:
    correlation = _df[list(core.SCORE_COLUMNS.values())].corr(method="spearman")
    names = {column: name for name, column in core.SCORE_COLUMNS.items()}
    return viz3d.overlap_table(_df), correlation.rename(index=names, columns=names)


@st.cache_data(max_entries=8)
def sensitivity(sid: int, _df: pd.DataFrame, _scales: dict) -> dict:
    return viz3d.response_data(_df, _df.loc[sid].to_dict(), _scales)


def show_model_lab(df: pd.DataFrame, row: dict, case_banner: Callable,
                   render_chart: Callable, navigate: Callable) -> None:
    st.title("Ba mô hình đánh giá một vị thế vay")
    st.caption("Chọn ca thật, xem cách tính điểm, thử đầu vào, rồi đối chiếu sự kiện trong 7 ngày.")
    case_banner(row)
    scales = rank_scales(df)
    tabs = st.tabs(["Khám phá ba mô hình", "Kiểm chứng kết quả", "Phân tích sâu"])
    with tabs[0]:
        showcase.explore(row, sensitivity(int(row["sid"]), df, scales), render_chart, navigate)
    with tabs[1]:
        showcase.validate_story(df, row, render_chart, navigate)
    with tabs[2]:
        st.caption("Các hình phục vụ phản biện về thứ hạng, độ nhạy và giới hạn dữ liệu.")
        advanced = st.tabs(["Bản đồ xếp hạng 3D", "Bề mặt theo thứ hạng"])
        with advanced[0]:
            _rank_view(df, row, scales, render_chart)
        with advanced[1]:
            _response_view(df, row, scales, render_chart)


def _rank_view(df: pd.DataFrame, row: dict, scales: dict, render_chart: Callable) -> None:
    st.subheader("Ba mô hình có cùng ưu tiên các ca thanh lý không?")
    st.write("Mỗi điểm là một mẫu pool–ví–tuần; mỗi trục là thứ hạng của một mô hình trên toàn bộ tập test. "
             "Phía top nhỏ hơn nghĩa là được ưu tiên cao hơn.")
    points = rank_points(df, scales)
    render_chart(viz3d.rank_space_figure(points, scales, row), key="rank_space")
    table, correlation = rank_tables(df)
    missed = table.loc[table["Mô hình cảnh báo"].eq("Không mô hình nào")].iloc[0]
    no_history = int(table["Chưa có tiền sử"].sum())
    caught_history = int(table["Có tiền sử 90 ngày"].sum() - missed["Có tiền sử 90 ngày"])
    st.info(f"{int(missed['Chưa có tiền sử'])}/{no_history} ca thanh lý chưa có tiền sử 90 ngày bị cả ba mô hình bỏ sót. "
            f"Trong {int(table['Có tiền sử 90 ngày'].sum())} ca có tiền sử, {caught_history} ca được ít nhất một mô hình cảnh báo.")
    left, right = st.columns([1.55, 1])
    with left:
        st.subheader(f"{int(df.target.sum())} ca thanh lý: mô hình nào cảnh báo?")
        st.dataframe(table, hide_index=True, width="stretch")
    with right:
        st.subheader("Mức đồng thuận thứ hạng")
        st.dataframe(correlation.style.format("{:.2f}"), width="stretch")
        st.caption("Tương quan Spearman trên toàn bộ tập test; tương quan cao không đồng nghĩa phát hiện tốt.")
    with st.expander("Cách đọc", key="rank_reading"):
        st.markdown("""
- Điểm ba mô hình khác thang và chưa được hiệu chỉnh thành xác suất; trục chung là tỷ lệ snapshot có điểm cao hơn hoặc bằng ca đang xét, đổi sang thang log để nhìn rõ vùng đuôi.
- Mặt phẳng màu đánh dấu ngưỡng đã khóa từ validation, áp dụng đúng cho các điểm lịch sử đang vẽ. Bấm tên mặt phẳng trong chú giải để ẩn/hiện.
- Điểm vượt cả ba mặt phẳng là ca được cả ba cảnh báo; chấm đỏ đặc/rỗng phân biệt có/chưa có tiền sử 90 ngày. Không phải mọi chấm đỏ có tiền sử đều được phát hiện.
- Vị thế đang xem là hình thoi trắng; rê chuột để đọc HF, nợ, điểm từng mô hình và nhãn hậu kiểm. HF/nợ là thông tin bối cảnh.
- Giữ tất cả ca dương và tất cả cảnh báo sai, chỉ lấy mẫu phần âm không cảnh báo. Mật độ hình không biểu diễn tỷ lệ thật của tập test.
""")


def _response_view(df: pd.DataFrame, row: dict, scales: dict, render_chart: Callable) -> None:
    st.subheader("Cùng một vị thế, đổi tiền sử và tuổi lịch sử thì điểm thay đổi ra sao?")
    st.write("Hai trục được chọn theo hai biến có mean |SHAP| lớn nhất trên các ca dương của XGBoost: "
             "tuổi lịch sử vay và số lần thanh lý trong 30 ngày trước.")
    result = sensitivity(int(row["sid"]), df, scales)
    if not result["available"]:
        st.info(f"Tuổi lịch sử tối thiểu của ca này là {result['floor']:.1f} ngày, gần mức tối đa "
                f"{result['maximum']:.1f} ngày trong dữ liệu. Miền thay đổi quá hẹp để vẽ bề mặt hữu ích; chọn một ca khác.")
        return
    left, right = st.columns([2, 1])
    with left:
        model = st.segmented_control("Mô hình cần xem", list(core.SCORE_COLUMNS), default="XGBoost",
                                     key="response_model") or "XGBoost"
    compare = right.toggle("Xem cả ba bề mặt", value=False, key="compare_surfaces")
    names = list(core.SCORE_COLUMNS) if compare else [model]
    render_chart(viz3d.response_surfaces_figure(row, scales, result, names), key="response_surface")
    st.caption("Các dải tách rời ứng với số lần nguyên. Hình thoi là ca thật; mặt phẳng màu là mốc ngưỡng. "
               "Rê chuột để xem điểm và số mẫu thật tham chiếu trong mỗi ô.")
    earlier = int(row["past_liquidation_count_90d"] - row["past_liquidation_count_30d"])
    st.write(f"Giữ {earlier} lần thanh lý ở ngày 31–90; khi đặt 30 ngày = k, 90 ngày tự đổi thành k + {earlier}. "
             f"Tuổi lịch sử chạy từ {result['ages'].min():.2f} đến {result['ages'].max():.1f} ngày; "
             f"{len(core.summary()['features'])-3} đầu vào còn lại giữ nguyên.")
    rows = [{"Thanh lý 30 ngày (lần)": int(k),
             **{name: f"{int(result['models'][name]['alerts'][i].sum())}/{len(result['ages'])} mức tuổi"
                for name in core.SCORE_COLUMNS}} for i, k in enumerate(result["ks"])]
    st.subheader("Bao nhiêu mức tuổi lịch sử tạo cảnh báo cho ca nền này?")
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    with st.expander("Cách đọc", key="response_reading"):
        st.markdown("""
- Đây là độ nhạy của hàm đã học trên một vị thế, không chứng minh thay đổi hành vi sẽ gây ra hoặc ngăn thanh lý.
- Có ba cột đầu vào thay đổi: số lần thanh lý 30 ngày, số lần 90 ngày được cập nhật tương ứng, và tuổi lịch sử. Mọi hàng đều qua kiểm tra ràng buộc trước khi chấm.
- Tuổi lịch sử thấp nhất phải phù hợp các cửa sổ vay và độ gần đây của hoạt động quan sát. Kiểm tra này không bảo đảm kịch bản từng tồn tại ngoài đời.
- “Mẫu thật trong ô” chỉ đếm theo hai trục hiển thị, không xác nhận toàn bộ tổ hợp đầu vào. Vùng ít mẫu cần đọc thận trọng.
- Bảng cảnh báo dùng điểm gốc và ngưỡng đã khóa. Thứ hạng thực nghiệm có thể đồng hạng, nên mặt phẳng là mốc tham chiếu cho kịch bản chưa xuất hiện trong tập test.
""")


