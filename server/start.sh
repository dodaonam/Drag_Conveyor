#!/usr/bin/env bash
# Khởi động FastAPI + TryCloudflare tunnel + tự cập nhật R2 CORS
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── Dừng process cũ nếu có ─────────────────────────────────────────────────
echo "[1/4] Dừng process cũ..."
pkill -f "uvicorn main:app" 2>/dev/null || true
pkill -f "cloudflared tunnel" 2>/dev/null || true
sleep 1

# ── Khởi động FastAPI ──────────────────────────────────────────────────────
echo "[2/4] Khởi động FastAPI tại localhost:8000..."
uvicorn main:app --host 127.0.0.1 --port 8000 --log-level warning &
API_PID=$!
sleep 2

# Kiểm tra FastAPI đã sẵn sàng chưa
if ! curl -sf http://localhost:8000/api/health \
     -H "Authorization: Bearer $(grep API_AUTH_TOKEN .env | cut -d= -f2)" > /dev/null; then
  echo "Lỗi: FastAPI chưa khởi động được. Kiểm tra lại."
  kill "$API_PID" 2>/dev/null
  exit 1
fi
echo "    FastAPI OK (PID $API_PID)"

# ── Khởi động TryCloudflare tunnel ────────────────────────────────────────
echo "[3/4] Khởi động Cloudflare Tunnel..."
TMPLOG=$(mktemp)
cloudflared tunnel --url http://localhost:8000 > "$TMPLOG" 2>&1 &
CF_PID=$!

# Chờ URL xuất hiện (tối đa 30 giây)
TUNNEL_URL=""
for i in $(seq 1 30); do
  sleep 1
  TUNNEL_URL=$(grep -o 'https://[a-zA-Z0-9-]*\.trycloudflare\.com' "$TMPLOG" | head -1)
  [ -n "$TUNNEL_URL" ] && break
done
rm -f "$TMPLOG"

if [ -z "$TUNNEL_URL" ]; then
  echo "Lỗi: không lấy được URL từ cloudflared. Kiểm tra kết nối internet."
  kill "$API_PID" "$CF_PID" 2>/dev/null
  exit 1
fi
echo "    Tunnel OK (PID $CF_PID)"

# ── Cập nhật R2 CORS ──────────────────────────────────────────────────────
echo "[4/4] Cập nhật R2 CORS với URL mới..."
python update_cors.py "$TUNNEL_URL"

# ── Done ──────────────────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║  Hệ thống đã sẵn sàng!                              ║"
echo "║                                                      ║"
printf  "║  URL: %-46s ║\n" "$TUNNEL_URL"
echo "║                                                      ║"
echo "║  Gửi URL này cho nhân viên để bắt đầu kiểm tra.     ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""
echo "Nhấn Ctrl+C để dừng tất cả."

# Giữ script chạy, dừng khi Ctrl+C
trap "echo ''; echo 'Đang dừng...'; kill $API_PID $CF_PID 2>/dev/null; exit 0" INT TERM
wait
