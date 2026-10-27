# Showcase 3D: một vị thế thật đi qua ba mô hình

## Mục đích của phần trình diễn

Sau khoảng 3 phút, giảng viên cần thấy được bốn việc: nhóm có dữ liệu gì tại thời điểm chấm, mỗi thuật toán tạo điểm như thế nào, khi đổi một giả định thì điểm phản ứng ra sao, và dự báo có khớp sự kiện trong 7 ngày không.

Prompt của nhóm có phần đối chiếu số liệu và kiểm tra ràng buộc hữu ích. Bản đồ thứ hạng, Spearman và bề mặt trên thang log phù hợp với phần phản biện chuyên sâu. Chúng đòi hỏi quá nhiều giải thích nếu dùng làm màn hình đầu tiên cho người xem mới.

## Cách tiếp cận đã chọn

| Hướng trình bày | Giúp hiểu điều gì? | Cách sử dụng |
| --- | --- | --- |
| Một vị thế qua ba mô hình | Cùng đầu vào, cách tính và ngưỡng khác nhau tạo quyết định khác nhau | Màn hình mặc định |
| Bản đồ nhiều vị thế theo thứ hạng | Mô hình đồng thuận đến đâu, bỏ sót nhóm nào | Tab Phân tích sâu |
| Sơ đồ nhánh của một cây | Cách một cây chia dữ liệu | Bảng đường đi có thể mở thêm; không dùng một cây để giải thích toàn rừng |

Màn hình chính giữ cùng một vị thế. Chuyển Logistic, Random Forest và XGBoost để giải thích từng thuật toán. Điểm của cả ba luôn hiện cùng ngưỡng riêng. Thử đổi đầu vào thì điểm, vị trí chấm 3D và phần giải thích cơ chế đều lấy từ cùng đầu vào đó.

## Ba chiều có nghĩa cụ thể

- **Ngang:** số lần thanh lý trong 30 ngày trước, chỉ nhận số nguyên quan sát trong dữ liệu.
- **Chiều sâu:** số ngày lịch sử vay được quan sát. Đây không phải tuổi tạo địa chỉ ví.
- **Chiều cao:** điểm do pipeline đang chọn tính. Mặt phẳng vàng là ngưỡng đã khóa của chính mô hình ấy.

Trắng là ca gốc thật; tím là đầu vào đang thử. Bề mặt nối các mức đầu vào đã được chấm. Hai điểm đánh dấu được tính trực tiếp, không đọc bằng cách nội suy hình vẽ.

Số lần 90 ngày phải đổi theo số lần 30 ngày để giữ phần 31–90 ngày. Nếu còn thay cả tuổi lịch sử thì có ba cột đầu vào thay đổi, 31 cột khác giữ nguyên. Mọi đầu vào đều được kiểm tra ràng buộc trước khi chấm.

## Mỗi mô hình cần giải thích bằng một câu

| Mô hình | Câu nói với giảng viên | Số liệu mở thêm khi được hỏi |
| --- | --- | --- |
| Logistic Regression | Cộng các tín hiệu theo trọng số, rồi đổi tổng thành điểm | Các đóng góp cộng đủ tổng z trước sigmoid |
| Random Forest | Nhiều cây chấm cùng đầu vào, điểm cuối là trung bình các cây | Phân bố điểm của 300 cây và một đường đi thật |
| XGBoost | Cộng dần các điều chỉnh của những cây đã học nối tiếp | Điểm qua các giai đoạn của booster 400 cây |

Các số 300/400 lấy từ mô hình đã lưu. Việc chọn cấu hình được giải thích bằng thực nghiệm validation trong báo cáo; không phải cứ nhiều cây hơn là tốt hơn.

## Kịch bản 3 phút với ca đầu tiên

**0:00–0:20 — Nêu câu hỏi.** “Đây là một vị thế lịch sử trên Aave Polygon. Nhóm kiểm tra liệu lịch sử trước thời điểm chấm có giúp cảnh báo sự kiện thanh lý trong 7 ngày tiếp theo không.”

**0:20–0:55 — Logistic.** Chỉ điểm 0,716 và ngưỡng 0,899: chưa cảnh báo. Mở số liệu cơ chế nếu cần để chỉ các tín hiệu cộng thành tổng có trọng số. Hình 3D cho thấy điểm thay đổi theo hai đầu vào, các cột khác giữ ở ca nền.

**0:55–1:25 — Random Forest.** Chỉ 300 cây tạo trung bình 0,463, dưới ngưỡng 0,651. Có 144/300 cây chấm trên ngưỡng nhưng rừng vẫn chưa cảnh báo: rừng lấy trung bình điểm, không lấy số đông cây để quyết định ở ngưỡng đó.

**1:25–1:55 — XGBoost.** Điểm 0,707 đạt ngưỡng 0,349 nên cảnh báo. Điểm Logistic cao hơn nhưng lại không cảnh báo vì hai mô hình có ngưỡng riêng. Không gọi hai điểm là xác suất thực tế.

**1:55–2:30 — Thử một giả định.** Giữ tuổi lịch sử khoảng 637,1 ngày, đổi số lần thanh lý gần đây từ 0 sang 1. XGBoost đổi từ khoảng 0,0477 sang 0,6673 và vượt ngưỡng. Phần 31–90 ngày vẫn giữ 2 lần, nên “0 lần trong 30 ngày” không có nghĩa là chưa từng thanh lý trong 90 ngày. Nhấn Trở về ca thật để kết thúc thử.

**2:30–3:00 — Kiểm chứng.** Mở Kiểm chứng kết quả. Ca gốc ghi nhận thanh lý khoảng 2,75 ngày sau cutoff; XGBoost cảnh báo đúng trong ca này. Trên toàn test, XGBoost chỉ tìm 10/80 mẫu và có 290 cảnh báo sai; quy tắc tiền sử 90 ngày tìm 24/80 với tải cảnh báo khác. Kết luận cần dựa trên toàn tập và nhiều tiêu chí.

## Những điều giảng viên có thể hỏi

- **Có huấn luyện lại để vẽ hình không?** Không. Mọi điểm đều từ pipeline đã lưu; mô hình, ngưỡng và dữ liệu không đổi.
- **Bề mặt có chứng minh tiền sử gây thanh lý không?** Không. Nó mô tả phản ứng của hàm đã học. Kịch bản thử không có nhãn tương lai để biết đúng/sai.
- **Vì sao không đổi HF để điểm chạy theo?** Bộ history34 không nhận HF, nợ hoặc giá; tự cho HF làm điểm đổi sẽ sai với mô hình trong báo cáo.
- **Một ca đúng có đủ chứng minh dùng được không?** Không. Tab Kiểm chứng cho thấy cả kết quả toàn test và quy tắc đơn giản để tránh chọn một ca thuận lợi rồi suy rộng.

Bản trình diễn: https://aave-polygon-nhom12.streamlit.app/. Có thể chạy local bằng CHAY_DEMO.bat tại http://localhost:8514.
