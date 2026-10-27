# Kịch bản demo với thầy — Nhóm 12

Thời lượng gợi ý: 6–8 phút. Mở sẵn app, chọn ca đầu tiên ở thanh bên. Mọi ca mặc định là lịch sử thật, không phải dữ liệu live.

## 1. Câu hỏi nghiên cứu (45 giây)

Mở **Tổng quan**. Nói: “Nhóm kiểm tra lịch sử giao dịch có giúp ưu tiên đúng vị thế xuất hiện thanh lý trong bảy ngày tiếp theo hay không. Đây là test hồi cứu 2024, 320.974 snapshot còn nợ và 80 mẫu dương.” Một ví có nhiều tuần; không gọi 80 mẫu là 80 người.

## 2. HF và giá trị sử dụng (60 giây)

Mở **Theo dõi HF**. Chọn ca “HF đỏ nhưng nợ rất nhỏ”, nhìn HF và nợ. Chọn ca đầu “cảnh báo đúng trước ngưỡng HF”. Kéo giá trị thế chấp -10%; giải thích HF đổi theo công thức, giữ LT. Đỏ < 1; vàng 1–1,3; xanh > 1,3. Mốc 1,3 chỉ là quy ước demo, không phải bảo đảm an toàn.

Chỉ rõ HF và điểm XGBoost tách nhau: mô hình history34 không dùng HF/nợ/giá. Kéo HF không được tự làm điểm mô hình thay đổi. Ví mới ngoài tập dữ liệu cần thu đủ lịch sử, không chỉ nhập địa chỉ là có thể chấm.

## 3. Chạy pipeline và giải thích (60 giây)

Mở **Chấm điểm vị thế**, để “Ca lịch sử thật”. Ba điểm được tính trực tiếp từ pipeline đã lưu. Chỉ ngưỡng riêng của từng mô hình, không gọi điểm là xác suất. XGBoost ca đầu có điểm khoảng 0,7072, HF khoảng 1,0187 và nợ khoảng 151 USDC.e.

Đọc waterfall: thanh lý trước 30 ngày đóng góp lớn nhất trong ca này. HF không có trong hình vì không thuộc 34 feature. SHAP nói mô hình dùng cột nào, không chứng minh nguyên nhân. Mở ca bỏ sót để công khai giới hạn.

Nếu cần demo nhập liệu, chọn **Chấm CSV**, tải mẫu rồi tải lại đúng tệp đó và bấm chạy. Với chỉnh biến, phải giữ số đếm theo thứ tự 1 ≤ 7 ≤ 30 ≤ 90 ngày; tổng hành vi bằng hoạt động. Đây là giả định, không có nhãn tương lai thật.

## 4. Một vị thế qua ba mô hình (3 phút)

- **Ca thật và ba quyết định:** giữ ca đầu. Logistic cho 0,716 so ngưỡng 0,899; RF cho 0,463 so ngưỡng 0,651; XGBoost cho 0,707 so ngưỡng 0,349. Chỉ XGBoost cảnh báo. Điểm Logistic cao hơn nhưng không cảnh báo, vì ngưỡng khác nhau.
- **Giải thích cơ chế:** lần lượt chọn Logistic, RF, XGBoost. Logistic cộng tín hiệu theo trọng số; RF lấy trung bình điểm của 300 cây; XGBoost cộng dần điều chỉnh của 400 cây. Mở số liệu cơ chế khi thầy hỏi sâu; các số đang xem dùng đúng đầu vào của chấm 3D.
- **Thử đầu vào:** giữ lịch sử khoảng 637,1 ngày, chuyển thanh lý 30 ngày từ 0 sang 1. XGBoost đổi khoảng 0,0477 thành 0,6673 và vượt ngưỡng. Phần ngày 31–90 giữ 2 lần, nên 0 lần gần đây không phải chưa có tiền sử 90 ngày. Đầu vào đang thử không có nhãn tương lai.
- **Kiểm chứng:** nhấn Trở về ca thật, sang Kiểm chứng kết quả. Ca gốc có thanh lý sau khoảng 2,75 ngày. Đọc thêm toàn test và quy tắc đơn giản để tránh kết luận từ một ca thuận lợi.

Kịch bản từng mốc thời gian và cách trả lời câu hỏi nằm trong **PHUONG_AN_SHOWCASE_3D.md**. Bản đồ thứ hạng được giữ trong Phân tích sâu, chỉ mở khi cần phản biện về nhóm bị bỏ sót và sự đồng thuận mô hình.

Điểm ba mô hình khác thang. Trục xếp hạng cho biết có bao nhiêu snapshot test đạt điểm cao hơn hoặc bằng ca đang xét; thang log tách vùng top nhỏ. Mặt phẳng phân loại chính xác các điểm lịch sử. Với kịch bản mới, thứ hạng có thể đồng hạng; bảng luôn xét cảnh báo bằng điểm gốc và ngưỡng gốc.

## 5. So sánh mô hình và trả lời phản biện (90 giây)

Mở **Đối chiếu kết quả**. Ở XGBoost/ngưỡng khóa: TP 10, FP 290, FN 70, TN 320.604. Precision 3,33%, recall 12,5%, F2 khoảng 0,08065. AP khoảng 0,01605.

Bỏ chọn “Dùng ngưỡng báo cáo”, giảm ngưỡng; số cảnh báo và recall có thể tăng nhưng precision có thể giảm. AP vẫn giữ vì thứ tự điểm không đổi. Nói rõ đang khám phá hậu kiểm, không đổi kết quả báo cáo.

So quy tắc 90 ngày: 24/80 ca, 787 cảnh báo và F2 cao hơn ở ngưỡng riêng; không gọi XGBoost tốt nhất cho vận hành. Cùng top 1%: XGBoost 19 ca, HF 3; top 5%: HF 52, XGBoost 29. Ưu thế phụ thuộc nguồn lực rà soát.

## 6. Kết luận (30 giây)

“Chúng em đã xây dữ liệu, kiểm tra quy trình và demo suy luận thật. Tín hiệu hiện phụ thuộc tiền sử thanh lý; chưa chứng minh cảnh báo sớm tổng quát hoặc ngăn được tổn thất. Cần mở rộng log/vị thế, chốt chính sách trước và kiểm chứng trên thời gian mới rồi mới chạy giám sát thử.”

## Ba câu hỏi nên trả lời được

1. **Tại sao HF không vào mô hình cuối?** Đã thử các bộ trạng thái/kết hợp; history34 vẫn cao nhất trên validation trong cấu hình đã thử. HF vẫn là trạng thái, đối chứng và tiêu chí phân nhóm.
2. **Đã ngăn được 10 vụ chưa?** Chưa. Đó là 10 snapshot đúng trong backtest, thuộc 7 ví, đều có tiền sử 90 ngày. Không có nghiên cứu can thiệp.
3. **Có chạy thật không?** Có: pipeline chấm trực tiếp ca/CSV và lưới hợp lệ, TreeSHAP chạy từ booster, điểm từng cây đọc từ fitted RF. Dữ liệu là lịch sử, không phải số dư live.
4. **Tại sao chọn hai trục bề mặt đó?** Tuổi lịch sử và tiền sử thanh lý 30 ngày có mean |SHAP| lớn nhất trên các ca dương của XGBoost; không suy ra nguyên nhân thực tế.
5. **Bề mặt có là vị thế đã từng tồn tại không?** Không bảo đảm. App kiểm tra ràng buộc theo dữ liệu hiện có; số mẫu trong ô chỉ là tham chiếu trên hai trục, không xác nhận toàn bộ tổ hợp đầu vào.
