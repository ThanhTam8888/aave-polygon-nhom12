# Aave Polygon · Nhóm 12

Demo nghiên cứu: kiểm tra khả năng dự báo sự kiện thanh lý trong 7 ngày từ lịch sử giao dịch on-chain.

Ứng dụng Streamlit dùng dữ liệu hồi cứu 2024: 320.974 snapshot còn nợ, 80 mẫu dương; ba pipeline Logistic, Random Forest và XGBoost được giữ nguyên từ thực nghiệm. Mô hình được chọn dùng 34 biến hành vi, không có HF trong đầu vào.

## Chạy

Python 3.12. Cài `pip install -r requirements.txt`, sau đó `streamlit run app.py`.

Streamlit Community Cloud: nhánh `main`, entrypoint `app.py`, Python 3.12. Giữ đủ `assets/`, `data/`, `models/` và `.streamlit/config.toml`. Không cần secrets hoặc RPC. Bản Linux x86_64 dùng `xgboost-cpu` cùng phiên bản để tránh cài thư viện GPU không sử dụng.

## Phạm vi

- HF là trạng thái lịch sử tại cutoff; điểm ML và nhãn tương lai là các đại lượng khác nhau.
- HF 1,3 là quy ước theo dõi trong demo, không bảo đảm an toàn.
- Điểm mô hình chưa là xác suất đã hiệu chỉnh; SHAP không chứng minh nhân quả.
- XGBoost tìm 10/80 mẫu tại ngưỡng khóa; quy tắc tiền sử 90 ngày tìm 24/80, với tải cảnh báo khác nhau.
- Dữ liệu test đã từng được xem. App là công cụ trình bày và kiểm chứng hồi cứu, chưa chứng minh khả năng vận hành cảnh báo trực tiếp.

Kịch bản trình bày nằm trong `KICH_BAN_DEMO.md`. Không huấn luyện lại hoặc thay ngưỡng khi mở app.
