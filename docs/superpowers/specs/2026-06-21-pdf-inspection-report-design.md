# Thiết kế: Lưu kết quả & xuất phiếu kiểm tra PDF (XT-100)

- **Ngày:** 2026-06-21
- **Trạng thái:** Đã duyệt thiết kế, chờ viết plan
- **Nhánh:** feat/report

## 1. Bối cảnh & vấn đề

Sau khi chạy inspection, KTV mở kết quả trên điện thoại (web app trong `server/static/`) và **sửa lại phân loại lỗi cho đúng**. Hiện việc sửa chỉ nằm trong trình duyệt (`G.allDefects`/`G.allNormals` trong `app.js`) — **không có cơ chế lưu**. Không có endpoint lưu kết quả, không có sinh PDF.

Cần: một nút **"Lưu kết quả"** → sinh **file PDF** theo đúng format phiếu **XT-100** ("PHIẾU KIỂM TRA XÍCH TẢI") → lưu vào **thư mục trên máy tính chạy local**.

Server chạy trong **WSL** trên máy Windows, phơi ra ngoài qua cloudflared tunnel; điện thoại truy cập qua URL tunnel.

## 2. Mục tiêu / Ngoài phạm vi

**Mục tiêu**
- Nút "Lưu kết quả" trên màn kết quả của web app.
- Sinh PDF format XT-100 với 4 loại lỗi: biến dạng bên trái (`bent_left`), bên phải (`bent_right`), 2 bên (`bent_both`), gãy (`broken`).
- Lưu PDF vào thư mục Windows cấu hình qua `.env`.

**Ngoài phạm vi (YAGNI)**
- Các loại lỗi khác trong mẫu (biến dạng giữa, biến dạng màu, dị vật) — chưa cần.
- Lưu lịch sử/audit bản sửa vào DB (PDF là bản ghi cuối). Có thể bổ sung sau.
- Tải PDF về điện thoại; chọn thư mục đích từ điện thoại.

## 3. Quyết định thiết kế (chốt từ brainstorming)

| Vấn đề | Quyết định |
|---|---|
| Nơi lưu PDF | Thư mục Windows, cấu hình qua `.env` (`REPORTS_DIR`), vd `/mnt/c/Users/lebao/KetQuaKiemTra` |
| Thanh còn `other`/`_unclassified` | **Chặn lưu**, buộc KTV phân loại hết về 1 trong 4 loại (hoặc bình thường) trước |
| Render PDF | **WeasyPrint** (HTML+CSS → PDF), bám sát mẫu, tái dùng bảng màu web |
| Dữ liệu client gửi | Điện thoại gửi **trạng thái cuối của mọi thanh** (corrections); server tra ảnh từ store của nó |

## 4. Kiến trúc & luồng

```
[Điện thoại]  KTV sửa kết quả (G.allDefects)
     │  bấm "Lưu kết quả"
     │  ├─ validate cục bộ: còn 'other'/'_unclassified' → chặn + báo lỗi
     │  └─ POST /api/jobs/{id}/report { inspector_name, conveyor_name, corrections[] }
     ▼
[Server WSL]
     ├─ load summary đã lưu (nguồn sự thật cho ảnh/snapshot_key)
     ├─ áp corrections → tập defect cuối (4 loại) + tính tổng
     ├─ validate lại: không còn other/unclassified (else 400)
     ├─ tải ảnh từ R2 theo snapshot_key (validate prefix results/{job_id}/)
     ├─ render HTML+CSS → PDF (WeasyPrint)
     └─ ghi PDF vào REPORTS_DIR → trả { filename }
```

**Nguyên tắc:** điện thoại chỉ gửi *quyết định của người*; server tự tra ảnh từ summary đã lưu → an toàn, không tin dữ liệu ảnh từ client.

## 5. Endpoint & hợp đồng dữ liệu

`POST /api/jobs/{job_id}/report` — auth giống các endpoint khác (`Depends(require_auth)`).

```jsonc
// request
{
  "inspector_name": "Trần Văn Trường",
  "conveyor_name": "M515A",
  "corrections": [
    { "bar_id": "<id>", "defect_type": "bent_left" },  // 4 loại hợp lệ
    { "bar_id": "<id>", "defect_type": "normal" }       // hạ về bình thường
  ]
}
// 200
{ "filename": "XT-100_M515A_20260611_1847_<job>.pdf", "saved": true }
// 400 nếu sau khi áp corrections còn defect 'other'/'_unclassified'
// 409 nếu job chưa completed; 404 nếu không có job
```

- `bar_id` đối chiếu với defects ∪ normals trong summary đã lưu (normals đã có `bar_id`).
- `defect_type` hợp lệ: `bent_left | bent_right | bent_both | broken | normal`.
- Áp corrections: thanh `normal` → loại khỏi tập lỗi; thanh đổi loại → cập nhật; thanh không có trong corrections → giữ nguyên trạng thái trong summary.

## 6. Template PDF (bám mẫu XT-100)

- **Header**: logo C.P. (tùy chọn `server/assets/logo.png`; không có thì chỉ chữ) + "CÔNG TY CỔ PHẦN C.P. VIỆT NAM / Chi nhánh Xuân Mai – Hà Nội • Phòng Kỹ Thuật" + "PHIẾU KT / Mã: XT-100". Chuỗi công ty để hằng số trong template.
- **Tiêu đề**: "PHIẾU KIỂM TRA XÍCH TẢI / Báo cáo kiểm tra cánh gạt băng tải".
- **3 ô meta**: NHÂN VIÊN KIỂM TRA (`inspector_name`); NGÀY & GIỜ KIỂM TRA (job `created_at`, format `dd/MM/yyyy · HH:mm` giờ ICT `_TZ_ICT`); TÊN MÁY (`conveyor_name`).
- **3 thẻ thống kê**: Tổng cánh đã kiểm tra (`total_bars`) · Tổng cánh lỗi (số defect sau sửa) · Tỷ lệ lỗi (%).
- **Tối đa 4 mục lỗi** (chỉ hiện mục có ≥1 cánh), thứ tự + nhãn:
  - "Biến dạng bên trái" → `bent_left`
  - "Biến dạng bên phải" → `bent_right`
  - "Biến dạng 2 bên" → `bent_both`
  - "Gãy" → `broken`
  Mỗi mục: badge "N cánh lỗi" + lưới ảnh (~3 ảnh/hàng), tự xuống trang. Ảnh snapshot đã có sẵn overlay contour xanh + bbox đỏ.
- Nếu 0 cánh lỗi: ghi "Không phát hiện cánh lỗi".
- Ảnh nhúng base64 data-URI (tải từ R2). CSS riêng `report.css` (tái dùng bảng màu `styles.css`).

## 7. Config & lưu file

- `server/.env`: `REPORTS_DIR=/mnt/c/Users/lebao/KetQuaKiemTra`. `settings.py` đọc, tạo thư mục nếu chưa có.
- Tên file: `XT-100_{conveyor}_{YYYYMMDD_HHMM}_{job_id}.pdf`; sanitize `conveyor` (bỏ ký tự không an toàn cho tên file); trùng tên → thêm hậu tố `_2`, `_3`...
- Dependency: thêm `weasyprint` vào `pyproject.toml`. **Một lần** trong WSL:
  `sudo apt-get install -y libpango-1.0-0 libpangoft2-1.0-0 libcairo2 libgdk-pixbuf-2.0-0` (ghi vào README).

## 8. Frontend (`app.js`, `index.html`, `styles.css`)

- Nút **"Lưu kết quả"** ở màn kết quả.
- Bấm → kiểm tra `G.allDefects` còn `other`/`_unclassified` → nếu có: chặn, highlight, báo "Hãy phân loại hết các thanh chưa rõ".
- Hợp lệ → gom `corrections` (trạng thái cuối của mọi thanh defect + normal đã đổi) → POST → hiện "Đã lưu: <filename>" hoặc lỗi.
- Trạng thái nút: đang lưu (spinner) / xong / lỗi.

## 9. Xử lý lỗi & ca biên

| Tình huống | Hành vi |
|---|---|
| Còn 'other'/chưa phân loại | Chặn ở client; server cũng trả 400 (phòng thủ) |
| `REPORTS_DIR` không ghi được | 500 + thông báo rõ; tạo thư mục nếu thiếu |
| Ảnh R2 lỗi/thiếu | Bỏ ô ảnh đó, vẫn render PDF; log cảnh báo |
| `snapshot_key` không thuộc job (prefix khác `results/{job_id}/`) | Từ chối (chống đọc R2 tùy tiện) |
| Job chưa completed | 409 |

## 10. Kiểm thử

- **Pure-python (không cần WeasyPrint)**: áp corrections → tập defect đúng; tính tổng/tỷ lệ; validate chặn 'other'; sanitize tên file; build chuỗi HTML chứa đúng mục/đếm.
- **Smoke test render**: WeasyPrint sinh bytes bắt đầu `%PDF` (skip nếu thiếu lib hệ thống).
- **Endpoint**: corrections áp đúng; file ghi vào `REPORTS_DIR` tạm; trả filename; ca 400/409.

## 11. Module mới / file sửa

**Mới**
- `server/report.py` — áp corrections, tính tổng, validate, build HTML, render PDF, lưu file (logic thuần tách khỏi WeasyPrint).
- `server/templates/report.html` + `server/templates/report.css`.
- `server/assets/logo.png` (tùy chọn).

**Sửa**
- `server/main.py` — endpoint `POST /api/jobs/{job_id}/report`.
- `server/settings.py` — `REPORTS_DIR`.
- `server/.env` — `REPORTS_DIR=...`.
- `pyproject.toml` — thêm `weasyprint`.
- `server/static/app.js`, `index.html`, `styles.css` — nút Lưu + validate + gọi API.
- `README.md` — ghi bước cài lib hệ thống cho WeasyPrint.
