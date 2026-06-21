Chạy web server để upload video và chọn ROI trực tiếp từ giao diện mobile/web.

ROI hiện luôn xử lý theo hướng băng chuyền từ trên xuống dưới; giao diện chỉ cho chọn vùng kiểm tra và trigger band.

## Xuất phiếu PDF (WeasyPrint)

Tính năng lưu phiếu kiểm tra PDF dùng WeasyPrint. Cài thư viện hệ thống một lần trong WSL:

    sudo apt-get install -y libpango-1.0-0 libpangoft2-1.0-0 libcairo2 libgdk-pixbuf-2.0-0

Đặt thư mục lưu phiếu trong `server/.env`:

    REPORTS_DIR=/mnt/c/Users/<bạn>/KetQuaKiemTra
