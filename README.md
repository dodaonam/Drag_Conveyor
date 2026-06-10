```bash
uv run python -m drag_conveyor self-check --profile config/base_profile.json
uv run python -m drag_conveyor inspect --profile config/base_profile.json --source data/raw_data/vid_1.mp4
```

Hoặc chạy web server để upload video và chọn ROI trực tiếp từ giao diện mobile/web.

ROI hiện luôn xử lý theo hướng băng chuyền từ trên xuống dưới; giao diện chỉ cho chọn vùng kiểm tra và trigger band.
