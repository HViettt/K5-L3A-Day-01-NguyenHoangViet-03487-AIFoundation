# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

* **MSSV:** 2A202603487
* **Họ và tên:** Nguyễn Hoàng Việt
* **SEED:** 669

---

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | ☑ |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | ☑ |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước | ☑ |

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

* **Dự đoán 1 — Learning rate `lr=50` (Thí nghiệm 1.B, key `ml_lr`):**
  * *Bạn đoán gì:* Dự đoán đường loss sẽ dao động dữ dội, bị phân kỳ và vọt lên cao không ổn định do bước nhảy gradient quá lớn nhảy qua đáy cực tiểu.
  * *Ra gì:* Loss chỉ vọt lên ở vài bước đầu tiên, sau đó dốc thẳng xuống và hội tụ về mức cực nhỏ ($0.000$).
  * *Vì sao lệch:* Gradient descent trên hàm lồi với đầu vào nhị phân thưa (sparse binary features) khi gặp bước nhảy cực lớn đã đẩy các trọng số $w$ tiến thẳng ra vùng bão hòa của hàm Sigmoid, khiến xác suất $p$ ép sát về 0 hoặc 1 làm giá trị BCE loss tiệm cận 0 thay vì phân kỳ vô hạn.

* **Dự đoán 2 — Đánh đổi Pareto và SLA 50 ms (Biểu đồ Phần 5, key `tradeoff`):**
  * *Bạn đoán gì:* Dự đoán toàn bộ các mô hình LLM gọi qua API (4a, 4b, 4c) sẽ nằm bên phải vạch SLA 50 ms và bị các mô hình cục bộ lấn át hoàn toàn.
  * *Ra gì:* Hoàn toàn khớp thực tế: p95 của 4a là $1325.86\text{ ms}$, 4c lên tới gần $9000\text{ ms}$; các mô hình nằm trên đường Pareto và thỏa mãn SLA chỉ gồm 1 (BoW + LogReg), 2 (BoW + MLP) và 3b (Embedding + LogReg).
  * *Vì sao khớp:* Mô hình API chịu rào cản vật lý lớn về độ trễ mạng Internet và cơ chế sinh token tuần tự (autoregressive), không thể nào cạnh tranh tốc độ với suy luận vector ma trận nội bộ trên máy cục bộ.

* **Dự đoán 3 — So sánh LLM với Embedding (Thí nghiệm 4.A/4.B, key `llm_models`):**
  * *Bạn đoán gì:* Dự đoán LLM nhờ lượng tham số khổng lồ và khả năng hiểu ngữ cảnh sâu sẽ đạt F1 ngang ngửa hoặc nhỉnh hơn nhẹ so với mô hình 3b (Encoder + LogReg).
  * *Ra gì:* Thực tế LLM 4a chỉ đạt $F_1 = 0.857$, 4b đạt $0.884$, thua kém đáng kể so với mô hình 3b ($F_1 = 0.965$ trên tập test thường và $0.947$ trên SUB 40), đồng thời độ trễ p95 cao gấp hơn 30 lần.
  * *Vì sao lệch:* Phân loại spam là bài toán phân loại miền đóng (closed domain). Mô hình 3b được huấn luyện lớp phân loại trực tiếp trên tập train SMS nên học được ranh giới quyết định tối ưu cho bài toán này, trong khi LLM zero-shot là mô hình sinh ngôn ngữ tổng quát không có bước fine-tuning trên phân phối dữ liệu SMS, dễ bị phân vân trước các từ ngữ giao tiếp thông thường.

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
* **Cách làm:** Tự code vòng lặp qua danh sách $k \in [1, 5, 25]$, gọi hàm `knn_predict(E_tr, y_tr, E_te, k=k_val)` trên không gian vector MiniLM và tính điểm thông qua `metrics(y_te, preds)`. *(Phần này thực hiện độc lập 100%, không dùng AI theo đúng quy định)*.
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
| **Phần 1 (Công thức)** | Giải thích công thức đạo hàm hàm BCE loss kết hợp Sigmoid và vì sao gradient vector hóa có dạng $(1/n) X^T (p - y)$? | Trình bày chứng minh đạo hàm hàm hợp $\frac{\partial L}{\partial z} = p - y$, khi nhân với đạo hàm biến đầu vào $\frac{\partial z}{\partial w} = X$ thu được vector gradient $\frac{1}{n} X^T (p - y)$. | Chỉ tham khảo giải tích toán học; tự tay code hàm `train_logreg` bằng numpy, kiểm tra shape vector $(V,)$ và kiểm chứng qua bài test `_run_check("1.3 train_logreg", ...)`. |
| **Phần 2 (Thuộc tính & Khái niệm)** | Giải thích thuộc tính của hàm kích hoạt `nn.ReLU()`. Nếu bỏ ReLU giữa 2 lớp Linear thì mạng MLP suy biến thành mô hình nào? | Phân tích tính chất toán học: tích của hai phép biến đổi affine liên tiếp $W_2(W_1 x + b_1) + b_2$ tương đương một phép biến đổi tuyến tính duy nhất $W'x + b'$, khiến MLP suy biến về Logistic Regression và mất khả năng học quan hệ phi tuyến. | Tự tay triển khai kiến trúc PyTorch trong `build_mlp` và vòng lặp `train_mlp` thuần túy; dùng lý thuyết này để viết câu trả lời cho Câu hỏi 2.A và dự đoán `dl_relu`. |
| **Phần 3 (Công thức & Thuộc tính)** | Công thức tính Cosine Similarity giữa 2 vector và cách xử lý số học tránh lỗi ZeroDivisionError khi vector có độ dài bằng 0? | Cung cấp công thức $\cos(u, v) = \frac{u \cdot v}{\|u\|_2 \|v\|_2 + \epsilon}$, đề xuất thêm hằng số $\epsilon = 10^{-8}$ vào mẫu số để ổn định phép chia. | Tự code hàm `cosine_similarity` và `top_k_similar` bằng numpy thuần (`np.dot`, `np.linalg.norm`), vượt qua các bài kiểm tra `_run_check("3.1–3.2 cosine / top_k_similar", ...)`. |
| **Tính tay (Công thức)** | Kiểm tra lại công thức toán học của 5 bài toán trong mục Tính tay (Baseline Accuracy khi luôn đoán ham, Sigmoid(0), F1 từ Confusion Matrix 2x2, Cosine Similarity và chi phí token/ngày theo giá 1M tokens). | Cung cấp các công thức chuẩn: Baseline Accuracy = $\frac{\text{ham}}{\text{tổng}}$, $\sigma(0) = \frac{1}{1 + e^0} = 0.5$, $F_1 = \frac{2TP}{2TP + FP + FN}$, $\cos = \frac{u \cdot v}{\|u\| \|v\|}$, và công thức nhân số lượng token/ngày $\times$ đơn giá $/1\text{M}$. | Tự đặt bút tính toán chi tiết từng bước ra giấy, đối chiếu với cell kiểm tra mã hash SHA-256 trong notebook để nhận đủ 5 tích xanh và chụp ảnh `hand_calc.jpg`. |
| **Phần 4 (Code & Cấu hình - Vùng cho phép dùng AI)** | Khắc phục lỗi xác thực 401 khi gọi SDK Anthropic và chuyển đổi cấu hình sang endpoint OpenAI-compatible dùng model Gemini (`gemini-3.5-flash-lite`); xử lý lỗi rate limit 429. | Hướng dẫn cấu hình lại từ điển `PROVIDERS`, gán `base_url` của Google Gemini qua thư viện OpenAI client, đồng thời chèn `time.sleep(4.1)` giữa các request để không vượt ngưỡng TPM/RPM. | Đọc kỹ code, cấu hình an toàn bằng Colab Secrets (`userdata.get("GEMINI_API_KEY")`), chạy hoàn tất 40 tin của tập SUB mà không bị ngắt quãng, đo chính xác latency và chi phí token. |
| **Phần 4 (Thuộc tính Token & Prompt)** | Giải thích vì sao phân loại văn bản cần đặt `temperature=0`? Rủi ro khi đặt `max_tokens=1` là gì và tại sao model suy luận (reasoning) lại cần cẩn thận với ngân sách token? | Phân tích cơ chế tất định của bộ sinh nhãn khi `temperature=0`. Nếu `max_tokens=1`, chuỗi trả về dễ bị rỗng nếu token đầu tiên là khoảng trắng. Với model reasoning, các token suy nghĩ ngầm sẽ ngốn hết ngân sách nếu đặt quá nhỏ khiến câu trả lời rỗng. | Tự cấu hình `max_tokens=512` kết hợp `reasoning_effort="low"` trong dictionary của Gemini, viết hàm `parse_label` dùng regex bắt chuẩn xác nhãn spam/ham. |

> *Ghi chú tuân thủ quy tắc:* Ở các Phần 1, 2, 3 và phần Tính tay, AI chỉ được sử dụng để hỏi đáp công thức toán học và giải thích thuộc tính khái niệm, mã nguồn hoàn toàn do học viên tự viết. Riêng phần **Thử thách cá nhân (Phần 5)** được thực hiện độc lập 100%, tuyệt đối không sử dụng AI.

* **Điều hiểu ra hôm nay:**
1. Mô hình ngôn ngữ lớn (LLM) không phải là lựa chọn ưu tiên cho các bài toán phân loại nhãn thời gian thực quy mô lớn; các yếu tố về chi phí, thông lượng và chuẩn SLA độ trễ đóng vai trò sống còn trong môi trường kỹ thuật sản phẩm.
2. Điểm số cao trên tập dữ liệu kiểm thử cùng phân phối (IID) tạo cảm giác an toàn giả tạo; khi đối mặt với dữ liệu có tính đối kháng hoặc ngụy trang từ ngữ (stress test), những hạn chế cố hữu về mặt biểu diễn của từng thế hệ AI mới bộc lộ rõ ràng.
3. Không gian biểu diễn (Representation) quyết định trần hiệu năng của bài toán: từ bag-of-words rời rạc đến không gian vector ngữ nghĩa liên tục của Transformer là một bước nhảy vọt về khả năng khái quát hóa.
