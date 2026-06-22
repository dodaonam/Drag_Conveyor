# Đóng gói Drag Conveyor thành EXE cài đặt trên máy khách

**Ngày:** 2026-06-22
**Trạng thái:** Đã duyệt thiết kế, chờ review spec

## 1. Mục tiêu

Đóng gói toàn bộ app AIO (GUI + server FastAPI + inference + xuất báo cáo) thành **một bộ cài Windows** chạy được trên máy khách **chưa từng có môi trường code** (không Python, không GTK). Sau khi cài, người dùng bấm shortcut để chạy mỗi lần đi kiểm tra.

Không xây dựng/duy trì bản cloud — app chạy tự chứa trên máy khách.

## 2. Ràng buộc & quyết định đã chốt

| Vấn đề | Quyết định |
|--------|-----------|
| GPU | Không cần — chạy **onnxruntime CPU**. Model nhỏ (imgsz 640). |
| Mạng tại máy khách | **Có internet ổn định.** Giữ nguyên kiến trúc mạng: Groq (VLM) + Cloudflare R2 (ảnh) + Cloudflare tunnel (QR điện thoại). Không refactor mạng. |
| Secret/key | **Không nhúng** key của dev vào bundle. Ship config template; key mới do dev/khách điền. |
| Hình thức cài | **Installer Inno Setup + shortcut** (ưu tiên); bản portable onedir là dự phòng. |
| Engine PDF | **Giữ weasyprint + kèm GTK runtime.** Đường lui: đổi sang Chromium chỉ bằng sửa `render_pdf` nếu GTK quá khó. |

## 3. Stack runtime thực tế (đã xác minh)

- **GUI:** Tkinter (`gui/app.py`) — màn setup, spawn server + cloudflared, hiện QR.
- **Server:** FastAPI/uvicorn (`server/main.py`), worker, SQLite `jobs.db`, R2 qua boto3.
- **Inference YOLO:** `onnxruntime` (nạp `best.onnx`) + `opencv` + `numpy`. **Không dùng torch/ultralytics** — file `.pt` chỉ là artifact train, runtime không đụng.
- **VLM:** Groq (key trong `.env` qua `GROQ_API_KEYS`).
- **PDF:** weasyprint (`server/report.py:149-151`), import lazy, dùng `report.css` + ảnh data-URI. Chỉ `save_report` gọi `render_pdf` → blast radius nhỏ.

## 4. Kiến trúc đóng gói

- **PyInstaller (onedir)** đóng băng app thành thư mục tự chứa (= bản portable).
- **Inno Setup** gói thư mục đó thành `Setup.exe`: cài vào `Program Files`, tạo shortcut Desktop/Start Menu, đặt config + `cloudflared.exe` + GTK + model.
- Một pipeline → hai artifact: `Setup.exe` (chính) + thư mục portable (dự phòng).

## 5. Các thay đổi code bắt buộc

### 5.1 Entry point hợp nhất (để freeze chạy được)

**Vấn đề:** GUI hiện spawn `sys.executable -m uvicorn main:app` (`gui/app.py:150`), `update_cors.py` (`gui/app.py:188`), `cloudflared` (`gui/app.py:160`). Khi freeze, `sys.executable` là exe chứ không phải python → các lệnh này hỏng.

**Giải pháp:** một exe duy nhất có subcommand:
- Không tham số → mở GUI.
- `--serve` → chạy uvicorn **in-process** (`uvicorn.run(app)`).
- `--update-cors <url>` (hoặc gọi in-process) thay cho spawn `update_cors.py`.
- GUI spawn server bằng `[sys.executable, "--serve"]` → đúng cả khi dev lẫn freeze.
- `cloudflared.exe` gọi bằng **đường dẫn tuyệt đối** cạnh exe, không dựa PATH.

### 5.2 Đường dẫn ghi & tài nguyên

- **Thư mục ghi** (`REPORTS_DIR`, `TEMP_DIR`, `jobs.db`) → `%LOCALAPPDATA%\DragConveyor\`.
  - `REPORTS_DIR`, `TEMP_DIR` đã override được qua env.
  - `jobs.db` đang hardcode `Path(__file__).parent` (`server/db.py:10`) → **cần cho phép override**.
- **Tài nguyên đọc** (`best.onnx`, `base_profile.json`, `report.css`, `static/`) → resolve theo thư mục bundle (`sys._MEIPASS`/thư mục exe), **không theo cwd**.

### 5.3 Config & secret

- Ship **file config template**; key mới do dev/khách điền (đúng kế hoạch tạo config mới).
- GUI hiện có màn nhập R2/API/OpenAI (`gui/app.py:20-27`) nhưng **Groq keys đang ở `.env`**, chưa lên GUI → đưa Groq vào config/template để khách cấu hình được.

## 6. Tài nguyên native nhúng kèm

onnxruntime CPU (+ providers DLL), opencv, numpy, **GTK runtime** (weasyprint), `cloudflared.exe`, **chỉ `best.onnx` 640** (bỏ `.pt` và onnx 320/416 nếu không dùng) để gói gọn hơn.

## 7. Tiêu chí nghiệm thu

Test trên **một máy Windows sạch** (chưa từng có Python/GTK):
1. GUI mở.
2. Server chạy in-process (`--serve`).
3. Cloudflare tunnel mở, QR hiện.
4. **onnxruntime nạp `best.onnx` + nhận diện chạy** (không lỗi thiếu DLL).
5. **Lưu PDF qua weasyprint/GTK thành công**, layout y hệt bản dev.
6. Thư mục ghi (`reports`, `temp`, `jobs.db`) nằm ở `%LOCALAPPDATA%`, ghi được.

## 8. Rủi ro còn mở (xử lý sau, không chặn MVP)

- **SmartScreen/antivirus** cảnh báo exe chưa ký số → cân nhắc chứng chỉ ký số sau.
- **Firewall khách** chặn outbound Groq/R2/tunnel → cần whitelist.
- **Tunnel `trycloudflare.com`** đổi URL mỗi lần chạy (ephemeral) — hiện chấp nhận được vì hiện QR mỗi phiên.

## 9. Ngoài phạm vi (YAGNI)

- Bản cloud / hosting.
- GPU/CUDA.
- Refactor luồng mạng (offline mode).
- Ký số (giai đoạn sau).
- Tải PDF về máy/điện thoại người kiểm tra (luồng hiện chỉ lưu trên máy host).
