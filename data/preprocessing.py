from pathlib import Path

import cv2
import numpy as np


# =========================================================
# CHỈNH 2 ĐƯỜNG DẪN NÀY
# =========================================================

VIDEO_FOLDER = Path(r"/home/lebao/projects/CP_segmentation/Drag_Conveyor/data/raw_data")
OUTPUT_FOLDER = Path(r"/home/lebao/projects/CP_segmentation/Drag_Conveyor/data/images")


# Số frame lấy từ mỗi video
FRAMES_PER_VIDEO = 17

# Mỗi frame cắt thành tối đa 3 ảnh
CROPS_PER_FRAME = 3

# Bỏ qua đầu và cuối video vì thường rung hoặc chưa ổn định
SKIP_START_SECONDS = 1
SKIP_END_SECONDS = 1

# Mỗi vị trí lấy ảnh sẽ kiểm tra các frame lân cận
SEARCH_RADIUS_FRAMES = 12

# Chỉ giữ crop có độ nét từ ngưỡng này trở lên
# Có thể thử 30, 50, 80 tùy video
MIN_SHARPNESS = 40.0

# Vùng chứa máy theo tỷ lệ ảnh:
# x trái, y trên, x phải, y dưới
#
# Với ảnh VLC anh gửi, máy nằm ở vùng trung tâm.
# Nếu video raw không có phần nền hai bên thì có thể đổi thành:
# MACHINE_ROI = (0.0, 0.0, 1.0, 1.0)
MACHINE_ROI = (0.34, 0.0, 0.66, 1.0)

# Kích thước ảnh đầu ra
OUTPUT_WIDTH = 512
OUTPUT_HEIGHT = 256

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv"}


def crop_by_ratio(image, roi):
    """Cắt ảnh theo tọa độ tỷ lệ."""
    height, width = image.shape[:2]

    x1_ratio, y1_ratio, x2_ratio, y2_ratio = roi

    x1 = int(width * x1_ratio)
    y1 = int(height * y1_ratio)
    x2 = int(width * x2_ratio)
    y2 = int(height * y2_ratio)

    return image[y1:y2, x1:x2]


def resize_with_padding(image, target_width, target_height):
    """Resize nhưng không làm méo ảnh."""
    height, width = image.shape[:2]

    scale = min(
        target_width / width,
        target_height / height
    )

    new_width = max(1, int(width * scale))
    new_height = max(1, int(height * scale))

    resized = cv2.resize(
        image,
        (new_width, new_height),
        interpolation=cv2.INTER_AREA
    )

    canvas = np.zeros(
        (target_height, target_width, 3),
        dtype=np.uint8
    )

    x_offset = (target_width - new_width) // 2
    y_offset = (target_height - new_height) // 2

    canvas[
        y_offset:y_offset + new_height,
        x_offset:x_offset + new_width
    ] = resized

    return canvas


def split_into_three(machine_image):
    """
    Chia vùng máy thành 3 vùng ngang.
    Mỗi vùng kỳ vọng chứa tối đa một thanh.
    """
    height, width = machine_image.shape[:2]

    # Có một chút overlap để tránh thanh nằm đúng ranh giới bị cắt đôi
    zones = [
        (0, int(height * 0.38)),
        (int(height * 0.28), int(height * 0.72)),
        (int(height * 0.62), height),
    ]

    crops = []

    for y1, y2 in zones:
        crop = machine_image[y1:y2, 0:width]

        if crop.size > 0:
            crops.append(crop)

    return crops


def process_video(video_path, image_counter):
    capture = cv2.VideoCapture(str(video_path))

    if not capture.isOpened():
        print(f"Không mở được: {video_path.name}")
        return image_counter

    fps = capture.get(cv2.CAP_PROP_FPS)
    total_frames = int(
        capture.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    if fps <= 0:
        fps = 30.0

    start_frame = int(
        SKIP_START_SECONDS * fps
    )

    end_frame = (
        total_frames
        - int(SKIP_END_SECONDS * fps)
        - 1
    )

    if end_frame <= start_frame:
        start_frame = 0
        end_frame = total_frames - 1

    target_frame_indices = np.linspace(
        start_frame,
        end_frame,
        FRAMES_PER_VIDEO,
        dtype=int
    )

    print(
        f"\nĐang xử lý {video_path.name}: "
        f"{len(target_frame_indices)} vị trí"
    )

    saved_from_video = 0

    for sample_index, center_frame_index in enumerate(
        target_frame_indices,
        start=1
    ):
        (
            frame,
            selected_frame_index,
            frame_sharpness
        ) = get_sharpest_frame(
            capture=capture,
            center_frame_index=int(center_frame_index),
            total_frames=total_frames,
            search_radius=SEARCH_RADIUS_FRAMES
        )

        if frame is None:
            continue

        machine_image = crop_by_ratio(
            frame,
            MACHINE_ROI
        )

        crops = split_into_three(
            machine_image
        )

        for crop_index, crop in enumerate(
            crops,
            start=1
        ):
            crop_sharpness = calculate_sharpness(
                crop
            )

            # Bỏ crop quá mờ
            if crop_sharpness < MIN_SHARPNESS:
                print(
                    f"  Bỏ frame {selected_frame_index}, "
                    f"crop {crop_index}: "
                    f"sharpness={crop_sharpness:.1f}"
                )
                continue

            output_image = resize_with_padding(
                crop,
                OUTPUT_WIDTH,
                OUTPUT_HEIGHT
            )

            image_counter += 1
            saved_from_video += 1

            output_name = (
                f"{video_path.stem}"
                f"_sample_{sample_index:03d}"
                f"_frame_{selected_frame_index:06d}"
                f"_bar_{crop_index}"
                f"_sharp_{crop_sharpness:.0f}.jpg"
            )

            output_path = (
                OUTPUT_FOLDER / output_name
            )

            cv2.imwrite(
                str(output_path),
                output_image,
                [cv2.IMWRITE_JPEG_QUALITY, 95]
            )

    capture.release()

    print(
        f"Đã lưu {saved_from_video} ảnh "
        f"từ {video_path.name}"
    )

    return image_counter
def calculate_sharpness(image):
    """
    Đo độ nét bằng phương sai Laplacian.
    Giá trị càng cao thì ảnh thường càng rõ.
    """
    if image is None or image.size == 0:
        return 0.0

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    return cv2.Laplacian(
        gray,
        cv2.CV_64F
    ).var()


def get_sharpest_frame(
    capture,
    center_frame_index,
    total_frames,
    search_radius=12
):
    """
    Tìm frame rõ nhất trong khoảng:
    center_frame_index ± search_radius.
    """
    start_index = max(
        0,
        center_frame_index - search_radius
    )

    end_index = min(
        total_frames - 1,
        center_frame_index + search_radius
    )

    best_frame = None
    best_frame_index = None
    best_sharpness = -1.0

    for frame_index in range(start_index, end_index + 1):
        capture.set(
            cv2.CAP_PROP_POS_FRAMES,
            frame_index
        )

        success, frame = capture.read()

        if not success or frame is None:
            continue

        # Chỉ đánh giá độ nét trong vùng máy,
        # tránh nền bên ngoài ảnh hưởng kết quả
        machine_image = crop_by_ratio(
            frame,
            MACHINE_ROI
        )

        sharpness = calculate_sharpness(
            machine_image
        )

        if sharpness > best_sharpness:
            best_sharpness = sharpness
            best_frame = frame.copy()
            best_frame_index = frame_index

    return (
        best_frame,
        best_frame_index,
        best_sharpness
    )


def main():
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

    videos = sorted([
        path
        for path in VIDEO_FOLDER.iterdir()
        if path.suffix.lower() in VIDEO_EXTENSIONS
    ])

    print(f"Tìm thấy {len(videos)} video:")

    for video in videos:
        print(f" - {video.name}")

    image_counter = 0

    for video in videos:
        image_counter = process_video(
            video,
            image_counter
        )

    print(f"\nHoàn thành: đã lưu {image_counter} ảnh")
    print(f"Folder output: {OUTPUT_FOLDER}")


if __name__ == "__main__":
    main()