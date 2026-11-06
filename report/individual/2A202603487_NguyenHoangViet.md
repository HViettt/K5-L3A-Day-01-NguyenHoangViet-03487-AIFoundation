# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

* **MSSV:** 2A202603487
* **Họ và tên:** Nguyễn Hoàng Việt
* **SEED:** 669

---

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](https://www.google.com/search?q=day01_lab.ipynb) — notebook đã chạy, còn output | ☑ |
| [`lab1_report.json`](https://www.google.com/search?q=lab1_report.json) — do ô "Báo cáo cuối" sinh ra | ☑ |
| [`hand_calc.jpg`](https://www.google.com/search?q=hand_calc.jpg) — ảnh tính tay có các bước | ☑ |

---

## 2. Kết quả chính

*Bảng trích xuất trực tiếp từ ô tổng hợp Phần 5 của notebook:*

| Thế hệ / Mô hình | F1 (test) | F1 (SUB 40) | p95 ms/tin | USD / 1k tin |
| --- | --- | --- | --- | --- |
| **1. BoW + LogReg (tự code)** | 0.906 | 0.857 | 0.156 | $0.00000 |
| **2. BoW + MLP (deep learning)** | 0.964 | 0.889 | 0.431 | $0.00000 |
| **3a. Encoder (MiniLM) + kNN** | 0.871 | 0.947 | 70.705 | $0.00000 |
| **3b. Encoder (MiniLM) + LogReg** | 0.965 | 0.947 | 39.392 | $0.00000 |
| **3c. Encoder (MiniLM) + MLP head** | 0.959 | 0.947 | 60.634 | $0.00000 |
| **4a. LLM nhỏ zero-shot (Gemini Flash-Lite)** | 0.857 | 0.857 | 1325.859 | $0.01900 |
| **4b. LLM lớn zero-shot (Gemini Flash-Lite)** | 0.884 | 0.884 | 9716.418 | $0.01900 |
| **4c. LLM nhỏ few-shot** | 0.900 | 0.900 | 8919.572 | $0.06000 |

---

## 3. Dự đoán và thực tế

* **Dự đoán 1 — Learning rate `lr=50` (Thí nghiệm 1.B):**
* *Bạn đoán gì:* Dự đoán đường loss sẽ dao động dữ dội, bị phân kỳ và vọt lên cao không ổn định do bước nhảy quá lớn.
* *Ra gì:* Loss chỉ vọt lên ở vài bước đầu tiên, sau đó dốc thẳng xuống và hội tụ về mức cực nhỏ ($0.000$).
* *Vì sao lệch:* Gradient descent trên hàm lồi với đầu vào nhị phân thưa (sparse binary features) khi gặp bước nhảy cực lớn đã đẩy các trọng số $w$ tiến thẳng ra vùng bão hòa của hàm Sigmoid, khiến xác suất $p$ ép sát về 0 hoặc 1 làm giá trị BCE loss tiệm cận 0 thay vì phân kỳ vô hạn.


* **Dự đoán 2 — Đánh đổi Pareto và SLA 50 ms (Biểu đồ Phần 5):**
* *Bạn đoán gì:* Dự đoán toàn bộ các mô hình LLM gọi qua API (4a, 4b, 4c) sẽ nằm bên phải vạch SLA 50 ms và bị các mô hình cục bộ lấn át.
* *Ra gì:* Hoàn toàn khớp thực tế: p95 của 4a là $1325.86\text{ ms}$, 4c lên tới gần $9000\text{ ms}$; các mô hình nằm trên đường Pareto và thỏa mãn SLA chỉ gồm 1 (BoW + LogReg), 2 (BoW + MLP) và 3b (Embedding + LogReg).
* *Vì sao khớp:* Mô hình API chịu rào cản vật lý lớn về độ trễ mạng Internet và cơ chế sinh token tuần tự (autoregressive), không thể nào cạnh tranh tốc độ với suy luận vector ma trận nội bộ.


* **Dự đoán 3 — Thử thách kNN với các giá trị $k$ (Đề #4):**
* *Bạn đoán gì:* $k=1$ dễ dính nhiễu/overfit, $k=25$ bị lớp đa số lấn át (underfit), $k=5$ cho kết quả cân bằng nhất.
* *Ra gì:* Khớp chính xác với thực nghiệm: $k=5$ đạt $F_1 \approx 0.932$, vượt trội so với $k=1$ ($F_1 \approx 0.895$) và $k=25$ ($F_1 \approx 0.865$).
* *Vì sao khớp:* Không gian embedding bảo toàn cấu trúc cụm ngữ nghĩa; chọn láng giềng vừa phải giúp triệt tiêu nhiễu đơn lẻ mà không làm phẳng biên quyết định vào lớp chiếm đa số (ham).



---

## 4. Phân tích 50 case khó

* **Mô hình tụt nhiều nhất:** Mô hình Bag-of-Words / TF-IDF tụt điểm nghiêm trọng nhất, đặc biệt ở nhóm `obfuscation` (né từ khóa) và nhóm `talks_about_spam` / `ham_looks_spam`.
* **Case cụ thể 1 (Obfuscation):**
* *Nội dung:* `"F R E E  e n t r y  into our weekly draw, t x t  WIN  to 80086"` (Case 30)
* *Cơ chế khiến model sai:* Bộ tách từ (tokenizer) cắt theo khoảng trắng làm từ khóa `FREE` biến thành các chữ cái đơn lẻ `F`, `R`, `E`, `E`. Các ký tự này không nằm trong từ điển đặc trưng spam của BoW, dẫn đến vector đếm rỗng và mô hình đoán nhầm thành **ham** (False Negative).


* **Case cụ thể 2 (Talks about spam):**
* *Nội dung:* `"Did you also get that weird 'your parcel is held' message? Don't click it, it's a scam"` (Case 48)
* *Cơ chế khiến model sai:* Đây là tin nhắn người dùng cảnh báo nhau (ham thật), nhưng lại chứa hàng loạt từ khóa có trọng số spam rất lớn (`parcel`, `click`, `scam`). BoW chỉ cộng dồn trọng số từ vựng độc lập mà không hiểu cấu trúc phủ định hoặc ngữ cảnh đàm thoại, dẫn tới kết luận nhầm là **spam** (False Positive).



---

## 5. Thử thách cá nhân

* **Đề nhận được (SEED=669):** Đề #4 — *Embedding: Chạy `knn_predict` với $k = 1, 5, 25$. Báo F1. Vì sao $k$ quá nhỏ hoặc quá lớn đều có hại?*
* **Cách làm:** Lặp qua danh sách $k \in [1, 5, 25]$, gọi hàm `knn_predict(E_tr, y_tr, E_te, k=k_val)` trên không gian vector MiniLM và tính điểm thông qua `metrics(y_te, preds)`.
* **Kết quả:**
* $k = 1$: $F_1 \approx 0.895$ | Precision = 0.842 | Recall = 0.954
* $k = 5$: $F_1 \approx 0.932$ | Precision = 0.915 | Recall = 0.950
* $k = 25$: $F_1 \approx 0.865$ | Precision = 0.960 | Recall = 0.787


* **Giải thích:**
* $k=1$ quá nhỏ dẫn đến **Overfitting**: biên phân loại quá nhạy cảm với các mẫu dị biệt, nhiễu cục bộ hoặc điểm gán nhãn sai nằm sát biên.
* $k=25$ quá lớn dẫn đến **Underfitting**: do mất cân bằng lớp (ham chiếm ~87%), việc mở rộng bán kính lấy mẫu khiến lớp đa số lấn át hoàn toàn số phiếu biểu quyết, làm sụt giảm nghiêm trọng độ nhạy (Recall) của tin spam.
* $k=5$ tạo sự cân bằng tối ưu giữa việc làm mịn biên phân tách và khả năng bắt đúng cấu trúc cụm.



---

## 6. Phương án cho sản phẩm thật

* **Phương án lựa chọn:** Mô hình **3b: Sentence Transformers (MiniLM) kết hợp Logistic Regression**.
* **Số liệu dẫn chứng:**
* Độ trễ: $p95 = 39.392\text{ ms} < 50\text{ ms}$ (thỏa mãn chỉ tiêu SLA thời gian thực khắt khe của hạ tầng viễn thông).
* Chất lượng: $F_1\text{ (test)} = 0.965$, $F_1\text{ (SUB 40)} = 0.947$ (nằm trong nhóm cao nhất của toàn bộ bài lab).
* Chi phí: $0 USD chi phí API bên thứ ba. Nếu dùng LLM cho 10 triệu tin nhắn/ngày, chi phí vận hành sẽ từ $6.000 đến $18.000 USD/tháng và độ trễ $p95 > 1300\text{ ms}$ làm sập luồng gửi tin tức thời.


* **Chấp nhận đánh đổi:**
* Cần tài nguyên RAM và tính toán máy chủ lớn hơn để chạy encoder transformer 384 chiều so với mô hình BoW truyền thống (tốc độ ~32 tin/s so với ~29.000 tin/s của BoW).
* Cần xây dựng quy trình giám sát và thu thập dữ liệu định kỳ để huấn luyện lại tầng phân loại nếu phân phối ngôn ngữ tin rác thay đổi.



---

## 7. AI log và điều đã học

| Phần | Bạn hỏi AI gì (prompt) | AI trả lời/viết gì | Bạn đã sửa / kiểm chứng gì |
| --- | --- | --- | --- |
| **Phần 4** | Lỗi xác thực 401 khi dùng Anthropic SDK và xử lý hàm `llm_classify_batch` | Hướng dẫn cấu hình lại `PROVIDERS` sang API tương thích OpenAI dùng Gemini (`gemini-3.5-flash-lite`), gán model nhỏ cho cả hai nhánh và chèn sleep 4.1s tránh rate limit 429 | Đã kiểm tra Secrets trên Colab, chạy hoàn tất 40 tin của tập SUB mà không bị gián đoạn hay văng lỗi xác thực |
| **Tính tay** | Kiểm tra công thức và quy tắc tính 5 giá trị trong `HAND_CALC` | Cung cấp công thức: Baseline Accuracy (0.87), Sigmoid(0) (0.50), F1 từ ma trận nhầm lẫn (0.73), Cosine similarity (0.60) và chi phí token/ngày (10.50 USD) | Đã tự tính lại từng bước ra giấy, đối chiếu với cell kiểm tra mã hash SHA-256 trong notebook để nhận đủ 5 tích xanh và chụp ảnh `hand_calc.jpg` |
| **Phần 4 (Prompt)** | Giải thích vì sao phải đặt `temperature=0` và rủi ro nếu để `max_tokens=1` hoặc thiếu cấu hình reasoning | Phân tích cơ chế tất định của bộ sinh nhãn và rủi ro chuỗi tư duy ngầm (reasoning tokens) ngốn hết token khiến kết quả rỗng | Tự tay kiểm tra phản hồi thực tế của LLM trên notebook và viết lại câu trả lời theo cách hiểu của bản thân |

* **Điều hiểu ra hôm nay:**
1. Mô hình ngôn ngữ lớn (LLM) không phải là lựa chọn ưu tiên cho các bài toán phân loại nhãn thời gian thực quy mô lớn; các yếu tố về chi phí, thông lượng và chuẩn SLA độ trễ đóng vai trò sống còn trong môi trường kỹ thuật sản phẩm.
2. Điểm số cao trên tập dữ liệu kiểm thử cùng phân phối (IID) tạo cảm giác an toàn giả tạo; khi đối mặt với dữ liệu có tính đối kháng hoặc ngụy trang từ ngữ (stress test), những hạn chế cố hữu về mặt biểu diễn của từng thế hệ AI mới bộc lộ rõ ràng.
