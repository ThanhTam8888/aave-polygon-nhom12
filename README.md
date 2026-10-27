# Aave Polygon · Nhóm 12

**Demo:** https://aave-polygon-nhom12.streamlit.app/

Demo nghiên cứu: kiểm tra khả năng dự báo sự kiện thanh lý trong 7 ngày từ lịch sử giao dịch on-chain.

Ứng dụng Streamlit dùng dữ liệu hồi cứu 2024: 320.974 snapshot còn nợ, 80 mẫu dương; ba pipeline Logistic, Random Forest và XGBoost được giữ nguyên từ thực nghiệm. Mô hình được chọn dùng 34 biến hành vi, không có HF trong đầu vào.

## Trình diễn ba mô hình

Trong **Mô hình 3D**, chọn một ca thật để so điểm với ngưỡng riêng của ba mô hình. Thử số lần thanh lý gần đây và tuổi lịch sử vay để thấy điểm cập nhật từ pipeline đã lưu; đầu vào đang thử không có nhãn tương lai. Logistic được giải thích qua tổng có trọng số, RF qua trung bình điểm các cây, XGBoost qua quá trình cộng dồn. Tab Kiểm chứng kết quả đối chiếu ca gốc và toàn test; Phân tích sâu giữ các đồ thị thứ hạng.

Lời dẫn 3 phút và lý do thiết kế nằm trong `PHUONG_AN_SHOWCASE_3D.md`.

## Chạy

Python 3.12. Cài `pip install -r requirements.txt`, sau đó `streamlit run app.py`.

Streamlit Community Cloud: nhánh `main`, entrypoint `app.py`, Python 3.12. Giữ đủ các module `model_lab.py`, `viz3d.py`, `showcase.py`, `showcase_figures.py` cùng `assets/`, `data/`, `models/` và `.streamlit/config.toml`. Không cần secrets hoặc RPC. Bản Linux x86_64 dùng `xgboost-cpu` cùng phiên bản để tránh cài thư viện GPU không sử dụng.

## Phạm vi

- HF là trạng thái lịch sử tại cutoff; điểm ML và nhãn tương lai là các đại lượng khác nhau.
- HF 1,3 là quy ước theo dõi trong demo, không bảo đảm an toàn.
- Điểm mô hình chưa là xác suất đã hiệu chỉnh; SHAP không chứng minh nhân quả.
- XGBoost tìm 10/80 mẫu tại ngưỡng khóa; quy tắc tiền sử 90 ngày tìm 24/80, với tải cảnh báo khác nhau.
- Dữ liệu test đã từng được xem. App là công cụ trình bày và kiểm chứng hồi cứu, chưa chứng minh khả năng vận hành cảnh báo trực tiếp.

Kịch bản trình bày nằm trong `KICH_BAN_DEMO.md`. Không huấn luyện lại hoặc thay ngưỡng khi mở app.
