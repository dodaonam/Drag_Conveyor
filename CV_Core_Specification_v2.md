# ĐẶC TẢ KỸ THUẬT CHÍNH THỨC

> Amendment implementation — 25/07/2026: Trong release placeholder hiện tại,
> quyết định deformation ưu tiên rule image-plane đơn giản: fit trục cánh theo
> `(s,q)`, so với transverse axis `h`, và lỗi khi `abs(angle) > 15°`.
> `h` là phương ngang vuông góc centerline; với centerline dọc, đây tương đương
> `abs(angle_from_vertical) < 75°`. Không dùng `minimum_side_pixels`, projected
> span pixel, hoặc guard band làm gate angle trong rule này; hai điểm hữu hạn là
> điều kiện toán học tối thiểu để fit line. Không fit được line là `uncertain`.
> Một side duy nhất vượt ngưỡng phải trả `uncertain` với
> `single_side_angle_exceeds_threshold`, không ép `bent_left/right`. Nhãn
> definitive `normal`/`bent_*` vẫn chỉ dùng sample cùng original frame có đủ hai
> side. Amendment này thay các bootstrap angle thresholds/gates mâu thuẫn ở Mục
> 22 và 27.2; các rule breakage, provenance, deterministic association và legacy
> compatibility vẫn giữ nguyên.

## Geometry V2 — Hệ thống kiểm tra cánh gạt xích tải dựa trên bằng chứng đa frame

| Thuộc tính | Giá trị |
|---|---|
| Mã tài liệu | `DC-CV-GEOMETRY-V2` |
| Phiên bản | `2.0.0` |
| Trạng thái | Implementation-ready; production capabilities chưa được enable |
| Ngày chốt | 24/07/2026 |
| Baseline source | Nhánh `v2`, commit `fec1e6b587547e4bd111973039e041c487e05c43` |
| Core runtime | Python 3.12, OpenCV, NumPy, ONNX Runtime |
| Model khởi đầu | `weights/model_imgsz_640/best.onnx` |
| SHA-256 model khởi đầu | `ef05955f43c8db6d2ff76b72fb65806e69afe525e85d8486eeb2dfb7566dcd65` |
| Result schema | `geometry_v2_result/2.0` |
| Rule version khởi đầu | `geometry_v2_rules/2.0.0` |

---

# 0. Cách đọc và tính ràng buộc của tài liệu

Các từ khóa sau mang nghĩa bắt buộc:

- **MUST / PHẢI**: yêu cầu bắt buộc để implementation được coi là phù hợp spec.
- **MUST NOT / KHÔNG ĐƯỢC**: hành vi bị cấm.
- **SHOULD / NÊN**: yêu cầu mặc định; chỉ được khác khi có bằng chứng benchmark và quyết định được ghi lại.
- **MAY / CÓ THỂ**: lựa chọn implementation không làm thay đổi hợp đồng bên ngoài.

Thứ tự nguồn sự thật khi có mâu thuẫn:

1. Các invariant và decision table trong tài liệu này.
2. Hành vi đã kiểm chứng của source tại baseline commit.
3. Hành vi thực nghiệm của model được pin bằng SHA-256.
4. Tài liệu dự án cũ.

Không được suy diễn capability của model chỉ từ tên file, input size hoặc nhãn training. Mỗi artifact model phải có capability manifest và benchmark riêng.

Các giá trị threshold trong tài liệu này là **bootstrap defaults** để implementation chạy được và test được. Chúng chưa phải bằng chứng accuracy production. Trước khi phát hành chính thức, threshold phải được hiệu chỉnh trên validation set có ground truth và được khóa trong một `vision_rule_version` mới.

## 0.1 Numeric và raster primitives canonical

Mọi implementation/fixture dùng cùng conventions sau:

- pixel `(x, y)` đại diện điểm ở tâm `(x + 0.5, y + 0.5)` khi project sang
  `(s, q)`;
- image arrays dùng row-major `[y, x]`;
- integer bbox/raster slice dùng half-open `[x1:x2, y1:y2]`;
- bbox crop hiện tại được rasterize sau clip bằng `floor` trên bốn tọa độ
  non-negative, đúng với `int()` của source baseline; `x2/y2` là exclusive;
- với ROI closed rectangle `[0, roi_width] × [0, roi_height]`, đặt
  `I = {s ∈ R | P0 + s*d thuộc ROI}`; `s_min = inf(I)`, `s_max = sup(I)`;
  không lấy projection của bốn ROI corners hoặc xấp xỉ từ image height;
- `H = s_max - s_min`;
- `chain_band_half_width = 0.5 * chain_band_width_ratio * roi_width`;
- bins là half-open `[edge_i, edge_{i+1})`, riêng bin cuối nhận cả right endpoint;
- quantile `P5/P10/P50/P90/P95` dùng Hyndman–Fan type 7
  (`numpy.quantile(..., method="linear")`);
- `median = P50`; `MAD = median(abs(x - median(x)))` theo cùng convention;
- sums/covariance/geometry reductions dùng `float64`; mask/area/count dùng integer;
- trước compare DP cost, quantize thành integer
  `round_half_even(cost / cost_quantization)`; tie-break không so raw float.

Nếu một dependency thay convention, adapter phải normalize về conventions trên.
Các golden tests phải kiểm cả tọa độ nằm đúng boundary/bin/quantile interpolation.

---

# 1. Quyết định kiến trúc đã chốt

## 1.1 Mode mới

Hệ thống bổ sung inspection mode:

```text
geometry_v2
```

Hai mode cũ:

```text
auto_baseline
average_ratio
```

được giữ nguyên trong giai đoạn chuyển đổi và không được refactor để dùng chung tracker, trigger hoặc rule với `geometry_v2`.

`geometry_v2` được triển khai thành pipeline song song, dùng chung hạ tầng inference và job processing nhưng có:

- mask/component analysis riêng;
- observation builder riêng;
- tracker riêng;
- trigger lifecycle riêng;
- evidence aggregation riêng;
- breakage analyzer riêng;
- angle analyzer riêng;
- decision engine riêng;
- renderer riêng.

## 1.2 Phần được tái sử dụng

Các thành phần sau được tái sử dụng:

- `OnnxRuntimeEngine.load/infer/close`;
- letterbox preprocessing và coordinate restore;
- YOLO detection decode;
- confidence/class filtering;
- NMS hiện có trong adapter model hiện tại;
- binary `Detection.mask_roi`;
- ROI crop;
- video/container infrastructure, qua geometry `FrameSource` contract riêng;
- model artifact loading;
- worker, SQLite queue và R2;
- public entry point `run_batch_inspection`;
- `BatchInspectionResult` ở biên tương thích;
- layout snapshot trên R2;
- presigned result URLs;
- desktop launcher, tunnel và packaging.

## 1.3 Phần không được tái sử dụng trong mode mới

`geometry_v2` MUST NOT dùng các thành phần sau để ra quyết định:

- `CentroidTracker` cũ;
- `TriggerEngine` cũ;
- `measure_contour`;
- detection filter dựa trên aspect ratio và `length * width`;
- auto-baseline percentile;
- average-ratio classifier;
- `RuleEngine` cũ;
- VLM;
- `Detection.contour_frame` làm nguồn hình học;
- `Detection.centroid_frame_xy` làm identity của cánh;
- `Detection.bbox_*` làm chiều dài cánh.

## 1.4 Frame nội suy

Trong `geometry_v2`, optical-flow slow motion mặc định bị tắt.

Pipeline MUST chạy trên frame gốc của video.

Nếu một implementation tương lai dùng frame nội suy để hỗ trợ motion prediction:

- frame phải có `is_original=false`;
- phải lưu source frames tạo ra nó;
- MUST NOT vote presence;
- MUST NOT vote center connection;
- MUST NOT vote breakage;
- MUST NOT đóng góp angle sample;
- MUST NOT được chọn làm snapshot;
- MUST NOT làm tăng evidence count.

## 1.5 VLM

VLM bị hard-disable trong `geometry_v2`.

```text
vlm_request_count = 0
classification_source = "geometry_v2"
```

Nhãn cuối phải deterministic từ video, geometry config, model artifact và rule config.

---

# 2. Sự thật đã kiểm chứng về hệ thống và model hiện tại

## 2.1 `largest` là post-processing mode, không phải model mode

Luồng post-processing hiện tại:

```text
model outputs
→ confidence/class filter
→ NMS
→ reconstruct probability mask
→ threshold binary mask
→ crop theo predicted bbox
→ find all external contours
→ chọn representative contour theo contour_mode
```

Với:

```text
contour_mode = "largest"
```

postprocessor chỉ làm cho:

- `contour_frame`;
- bbox trả ra;
- centroid trả ra;
- measurement cũ;

đại diện cho connected component lớn nhất.

`Detection.mask_roi` vẫn chứa toàn bộ binary components còn nằm trong predicted bbox.

Vì vậy spec MUST mô tả chính xác:

> Model và postprocessor hiện tại có thể phát một bên, hai detection riêng, hoặc một detection có nhiều component. `largest` làm cho tracker/measurement cũ quan sát representative component, nhưng không đảm bảo toàn bộ `mask_roi` chỉ có một component.

## 2.2 Giới hạn của `mask_roi`

Profile hiện tại có:

```text
crop_mask_to_bbox = true
```

Do đó `mask_roi` là:

```text
binary instance mask sau threshold và sau predicted-bbox crop
```

Nó không phải toàn bộ proto mask không giới hạn.

Current postprocessor chỉ tạo `Detection` khi cropped binary mask có ít nhất một
external contour với `contourArea > min_contour_area`. Vì vậy `mask_roi` giữ mọi
thresholded pixels trong raw model crop của một detection đã qua gate đó; nó không
đại diện các raw rows đã bị filter/reject trước khi `Detection` được tạo.

Pipeline MUST giữ `crop_mask_to_bbox=true` cho adapter model hiện tại trong cấu hình khởi đầu.

Không được tắt bbox crop chỉ để cố “cứu” fragment bị miss, vì thử nghiệm cho thấy việc này có thể đưa nhiều blob ngoài bbox vào mask.

Một controlled bbox padding chỉ được bổ sung sau A/B benchmark và phải được version hóa.

## 2.3 Các hình thái emission phải hỗ trợ

Pipeline MUST hỗ trợ tất cả hình thái sau:

```text
A. Một detection chứa một connected component.

B. Một detection chứa nhiều connected component đáng kể.

C. Hai hoặc nhiều detection trong cùng frame đại diện trùng lặp
   hoặc bổ sung cho cùng một physical paddle.

D. Frame N chỉ thấy trái, frame N+1 chỉ thấy phải.

E. Frame N chỉ thấy phải, frame N+1 chỉ thấy trái.

F. Một phía xuất hiện ổn định, phía kia bị miss trong toàn bộ event.

G. Không có detection trong một số frame.

H. Model/NMS suppress một complementary fragment.
```

Không hình thái nào trong số trên tự nó là một nhãn lỗi.

## 2.4 Kết quả kiểm chứng model khởi đầu

Artifact:

```text
path:
weights/model_imgsz_640/best.onnx

sha256:
ef05955f43c8db6d2ff76b72fb65806e69afe525e85d8486eeb2dfb7566dcd65

input:
[1, 3, 640, 640]

outputs:
[1, 300, 38]
[1, 32, 160, 160]
```

Trên các ảnh mẫu `broken` có sẵn:

- có ảnh trả hai detections trong cùng frame;
- có ảnh trả một detection;
- có detection chứa hai component lớn;
- có detection chỉ chứa một component;
- có hai detections cùng visual candidate với predicted-box IoU khoảng `0.296`;
- full detection-mask IoU khoảng `0.439`, nhưng overlap trên mask nhỏ hơn khoảng
  `0.909`;
- sau component split, cặp component trùng tương ứng có IoU khoảng `0.857` và
  IoS khoảng `0.938`.

Kết luận implementation:

- pre-split instance NMS/IoU đơn thuần là không đủ;
- phải tách component trước;
- phải dùng containment hoặc `intersection / min(area_a, area_b)`;
- phải track theo chain coordinate, không track theo wing centroid;
- phải có temporal fusion.

## 2.5 Model zoo không được coi là tương đương

Ba artifact 320/416/640 có output behavior khác nhau trên ảnh mẫu.

Mỗi production configuration MUST pin:

- model path;
- SHA-256;
- input size;
- thresholds;
- capability manifest.

Không được tự đổi artifact trong cùng một job.

Cấu hình bootstrap của `geometry_v2` pin model 640 nói trên.

320 và 416 chỉ được dùng cho `geometry_v2` sau benchmark riêng.

## 2.6 Giới hạn kiểm chứng

Repository không có raw video `vid4`/`vid5` kèm frame-level ground truth, nên chưa thể đo:

- tần suất đổi trái ↔ phải;
- recall fragment;
- precision broken-center;
- uncertainty rate production;
- threshold production tối ưu.

Các kết luận về temporal behavior trong spec là thiết kế bắt buộc để xử lý worst case do người dùng xác nhận, không phải metric đã chứng minh.

---

# 3. Mục tiêu nghiệp vụ

Pipeline nhận video xích tải chuyển động chủ yếu từ trên xuống dưới và phải:

1. Phát hiện các instance mask do model trả ra trong ROI.
2. Khôi phục và giữ thông tin connected components cần thiết.
3. Chuẩn hóa detection thành fragment observations.
4. Deduplicate component trùng lặp từ nhiều detection.
5. Ghép fragment trái/phải khi có bằng chứng đủ mạnh.
6. Theo dõi physical paddle theo vị trí dọc trục xích.
7. Cho phép observation chuyển đổi:

   ```text
   whole → left-only → right-only → whole
   ```

8. Không đếm hai fragment của một paddle thành hai paddle.
9. Xác nhận paddle qua trigger đúng một lần.
10. Thu thập frame thật trước và sau trigger.
11. Fusion các tracklet thuộc cùng một physical event.
12. Tổng hợp bằng chứng độc lập đa frame.
13. Phân tích breakage trước angle.
14. Chỉ đo góc khi hai phía và center integrity đủ tin cậy.
15. Trả đúng một canonical status cho mỗi physical paddle.
16. Trả `uncertain` thay vì ép nhãn khi thông tin không đủ.
17. Lưu một single-frame snapshot hoặc deterministic composite.
18. Ghi đủ diagnostics để tái lập quyết định.

---

# 4. Nhãn cuối và giới hạn nhận dạng

## 4.1 Canonical statuses

```text
normal
bent_left
bent_right
bent_both
broken_left
broken_right
broken_center
uncertain
```

Không có:

```text
broken_both
```

Semantic:

| Status | Định nghĩa |
| --- | --- |
| `normal` | Center và hai side nguyên vẹn; toàn bộ angle metrics hợp lệ và trong ngưỡng |
| `bent_left` | Side trái nguyên vẹn về topology nhưng trục hình học trái vượt ngưỡng |
| `bent_right` | Đối xứng với trái |
| `bent_both` | Cả hai side vượt ngưỡng, hoặc global tilt/center kink rule hợp lệ |
| `broken_left` | Có positive localized break/shortness/gap ở side trái ngoài center corridor |
| `broken_right` | Đối xứng với trái |
| `broken_center` | Kết nối topology qua center bị đứt, với bằng chứng hai phía đủ tin cậy |
| `uncertain` | Không đủ thông tin, identity/FOV/model conflict, hoặc damage không biểu diễn được bằng đúng một nhãn còn lại |

`bent_*` và `broken_*` là mutually exclusive trong một result. Breakage có ưu tiên;
khi break đã definitive thì bend không được đo. Nếu có nhiều break locations không
biểu diễn được bằng taxonomy, trả `uncertain`.

## 4.2 Ưu tiên quyết định

```text
Breakage
    ↓
Angle deformation
    ↓
Normal
```

Khi đã trả một nhãn broken:

```text
angle_left_deg = null
angle_right_deg = null
global_tilt_deg = null
center_kink_deg = null
```

## 4.3 Giới hạn quan sát bắt buộc

Nếu toàn bộ event chỉ quan sát được một phía, ví dụ:

```text
LEFT_ONLY, LEFT_ONLY, LEFT_ONLY
```

thì các ground truth sau có thể tạo cùng một observation:

```text
broken_right
broken_center + model luôn miss right fragment
segmentation dropout
ROI/FOV không đủ
```

Không có thuật toán deterministic nào phân biệt chắc chắn các trường hợp đó chỉ từ observation nói trên.

Với model khởi đầu, single-side localization là unavailable.

Do đó:

- một phía duy nhất MUST NOT tạo `broken_center`;
- một phía duy nhất MUST NOT tự động tạo `broken_left/right`;
- kết quả mặc định MUST là `uncertain`;
- output phải có `suspected_breakage=true`;
- output phải liệt kê `possible_breakage_statuses`;
- reason phải là `single_side_only_location_unidentifiable` hoặc reason cụ thể tương đương.

Đây không phải feature chỉ cần flip một boolean trong tương lai. Kể cả absence đã
được benchmark, `LEFT_ONLY` vẫn không tự phân biệt được `broken_right` với
`broken_center` có right fragment bị miss. Một model/adaptor tương lai chỉ có thể
cho single-side verdict khi cung cấp **observable mới được định nghĩa rõ**, ví dụ
attachment-to-chain/topology semantic hoặc explicit side-completeness class, và
toàn bộ decision rule mới được benchmark end-to-end theo capability record mới.

`absence_as_negative_evidence` nếu có chỉ chứng minh “side kỳ vọng không được
quan sát”; nó không tự cấp location `broken_left/right/center`. Cho tới khi có
contract/rule mới như trên, single-side localization là ngoài phạm vi và vẫn
`uncertain`.

## 4.4 Ngoài phạm vi

Pipeline không bắt buộc phát hiện:

- micro-crack không đổi silhouette;
- surface crack khi mask vẫn liền;
- biến dạng 3D không biểu hiện trong ảnh top-down;
- paddle mất hoàn toàn và không còn detection nào;
- paddle bị che hoàn toàn trong mọi frame;
- local curvature không làm thay đổi side axis, global tilt hoặc center kink;
- kích thước vật lý mm/cm;
- camera perspective 3D nghiêng lớn;
- lens calibration/homography;
- khoảng trống chu kỳ để suy ra một paddle mất hoàn toàn.

Nếu paddle mất hoàn toàn và model không trả detection:

```text
không tạo paddle result giả
không tạo broken_both
```

---

# 5. Invariant bắt buộc

Implementation MUST bảo đảm:

1. Một finalized event hypothesis có `count_certified=true` chỉ sinh một
   `paddle_id`.
2. Một `paddle_id` chỉ có một canonical status.
3. Một observation thuộc tối đa một tracklet.
4. Một tracklet thuộc tối đa một physical event sau fusion.
5. Final `paddle_id` chỉ được cấp sau event fusion.
6. Mọi association phải one-to-one và deterministic.
7. Association phải bảo toàn thứ tự paddle theo trục xích.
8. Association mơ hồ không được ép merge.
9. Trigger không đồng nghĩa evaluate ngay.
10. Track chưa trigger không tạo paddle result.
11. Track đã evaluated không được evaluate lại.
12. Chỉ frame gốc được vote evidence.
13. Một source frame tối đa một vote cho mỗi evidence type của một event.
14. Evidence phải được de-correlate theo thời gian.
15. `mask_roi`, không phải `contour_frame`, là nguồn mask của adapter hiện tại.
16. Geometry V2 không dùng centroid ngang của wing làm identity.
17. Không dùng bbox width làm side length.
18. Không dùng aspect ratio filter cũ để loại fragment.
19. Không dùng morphological closing lớn để tạo bridge.
20. Không tạo center connection nhân tạo.
21. Broken-center bắt buộc có bằng chứng cho cả trái và phải.
22. Một phía duy nhất không đủ kết luận broken-center.
23. Với capability hiện tại, một phía duy nhất không đủ định vị broken-left/right.
24. Negative evidence chỉ hợp lệ khi FOV opportunity hợp lệ.
25. Strong connected evidence và strong disconnected evidence xung đột phải trả `uncertain`.
26. Breakage luôn chạy trước angle.
27. Broken result có mọi final angle bằng `null`.
28. Global tilt không được tính bằng `abs(theta_left) + abs(theta_right)`.
29. Global tilt phải dùng hai outer endpoints trong cùng frame.
30. Center kink phải đo độ không thẳng hàng giữa hai fitted side axes.
31. Không được ghép left endpoint của frame A với right endpoint của frame B để tính angle.
32. VLM không tham gia canonical status.
33. Tie-break phải ổn định và được mô tả.
34. Cùng input/config/model phải tạo cùng decision JSON, trừ runtime timing fields được khai báo non-deterministic.
35. Model hash, config hash, rule version và evidence IDs phải được lưu.

---

# 6. Thuật ngữ và hệ tọa độ

## 6.1 Full frame

Frame gốc decode từ video.

Mỗi frame có:

```text
source_frame_id: integer, bắt đầu từ 1
source_timestamp_sec: decoder PTS đã chuẩn hóa; xem Mục 10.1
timestamp_source: decoder_pts | cfr_index_fallback
is_original: true
```

## 6.2 Inspection ROI

Hình chữ nhật do người dùng chọn trong full-frame coordinates:

```json
{
  "x": 100,
  "y": 80,
  "w": 1000,
  "h": 540,
  "frame_width": 1280,
  "frame_height": 720
}
```

ROI là vùng:

- inference;
- fragment analysis;
- tracking;
- trigger;
- evidence collection;
- snapshot.

## 6.3 Chain centerline

Đường tâm xích được người dùng xác định bằng hai điểm ROI-local:

```json
{
  "top": {"x": 500.0, "y": 0.0},
  "bottom": {"x": 515.0, "y": 540.0}
}
```

Hai điểm được canonicalize sao cho:

```text
top.y < bottom.y
```

Với trường hợp bằng nhau (sẽ không qua roll/span validation), tie-break bằng
`x` chỉ để serialization deterministic. Motion direction không làm đảo hai điểm;
direction được xử lý riêng ở tracker.

## 6.4 Chain coordinate

Cho:

```text
P0 = top point
P1 = bottom point
```

Vector dọc theo xích:

```text
d = (P1 - P0) / ||P1 - P0||
```

Vector ngang sang phải:

```text
h = (d_y, -d_x)
```

Với pixel ROI-local `P`:

```text
s(P) = dot(P - P0, d)
q(P) = dot(P - P0, h)
```

Ý nghĩa:

```text
s tăng: chuyển động xuống theo xích
q < 0: bên trái xích
q > 0: bên phải xích
```

Mọi tracking, pairing, left/right và angle MUST dùng `(s, q)`.

## 6.5 Chain band

Dải:

```text
abs(q) <= chain_band_half_width
```

Nó đại diện cho vùng:

- xích;
- liên kết paddle–chain;
- center topology;
- vùng loại bỏ khi fit side axes.

## 6.6 Trigger strip

Trigger canonical là một khoảng theo `s`, không phải centroid band theo trục Y tuyệt đối.

Cho phạm vi hợp lệ của centerline trong ROI:

```text
s_min
s_max
```

Hai giá trị này dùng exact projection/line–rectangle convention Mục 0.1, không mặc
định `s_min=0`, `s_max=roi_height`.

thì:

```text
trigger_center_s =
    s_min + trigger_center_ratio * (s_max - s_min)

trigger_height_s =
    trigger_height_ratio * (s_max - s_min)

trigger_top_s =
    trigger_center_s - trigger_height_s / 2

trigger_bottom_s =
    trigger_center_s + trigger_height_s / 2
```

Khi centerline thẳng đứng, strip này tương đương trigger rectangle ngang cũ.

Frontend SHOULD render polygon của trigger strip sau khi clip với ROI.

## 6.7 Fragment anchor

Fragment anchor là vị trí dọc `s` của fragment tại vùng gần chain band.

Nó không phải:

- mask centroid;
- bbox center;
- contour centroid.

## 6.8 Physical event, tracklet và paddle

- `observation`: bằng chứng trong một source frame.
- `tracklet`: chuỗi observation được online tracker ghép.
- `physical_event`: một hoặc nhiều tracklet sau offline fusion.
- `paddle_id`: sequence ID cấp cho physical event sau fusion.
- `track_id`: primary tracklet ID dùng cho compatibility.
- `track_ids`: toàn bộ alias tracklet IDs đã merge.

---

# 7. Input contract

## 7.1 Trusted core/local invocation

```json
{
  "video_source": "<path-or-stream>",
  "inspection_mode": "geometry_v2",

  "roi": {
    "x": 100,
    "y": 80,
    "w": 1000,
    "h": 540,
    "frame_width": 1280,
    "frame_height": 720
  },

  "geometry": {
    "schema_version": "geometry_input/2.0",
    "chain_centerline": {
      "top": {"x": 500.0, "y": 0.0},
      "bottom": {"x": 515.0, "y": 540.0}
    },
    "chain_band_width_ratio": 0.05
  },

  "vision_config_path": "config/geometry_v2.json"
}
```

`video_source` và `vision_config_path` chỉ thuộc trusted local/core invocation.
Remote `POST /api/jobs` không nhận hai field này; nó dùng presigned object upload
và server-controlled config path.

Remote request shape được định nghĩa ở Mục 28.1.

## 7.2 ROI validation

PHẢI thỏa:

```text
w > 0
h > 0
x >= 0
y >= 0
x + w <= frame_width
y + h <= frame_height
```

Ngoài ra:

```text
w >= minimum_roi_width_px
h >= minimum_roi_height_px
```

Bootstrap:

```text
minimum_roi_width_px = 160
minimum_roi_height_px = 160
```

Nếu invalid:

```text
core/worker re-validation:
    job failure
    failure_reason = "geometry_invalid_roi"

remote create-job validation:
    HTTP 422
    no DB job row created
```

## 7.3 Centerline validation

PHẢI thỏa:

- hai điểm hữu hạn;
- hai điểm khác nhau;
- line length tối thiểu `0.70 * roi_height`;
- line đi qua ROI;
- roll khỏi phương dọc không vượt `maximum_allowed_roll_deg`;
- centerline tại trigger không nằm quá sát biên ngang;
- chain band nằm trong ROI đủ để phân tích hai phía.

Centerline là continuous ROI geometry, nên endpoints hợp lệ trong closed bounds:

```text
0 <= x <= roi_width
0 <= y <= roi_height
```

Vì vậy `bottom.y == roi_height` trong sample là hợp lệ; đây không phải pixel-index
contract.

Roll:

```text
roll_deg = degrees(atan2(abs(d_x), abs(d_y)))
```

Bootstrap:

```text
maximum_allowed_roll_deg = 15.0
minimum_centerline_span_ratio = 0.70
```

Nếu người dùng nhập ngược top/bottom, implementation tự swap.

Nếu invalid:

```text
core/worker re-validation:
    job failure
    failure_reason = "geometry_invalid_centerline"

remote create-job validation:
    HTTP 422
    no DB job row created
```

## 7.4 Chain-band validation

Canonical configuration dùng ratio:

```text
chain_band_width_px = chain_band_width_ratio * roi_width
```

PHẢI thỏa:

```text
0.02 <= chain_band_width_ratio <= 0.20
```

Bootstrap:

```text
chain_band_width_ratio = 0.05
```

Schema 2.0 không nhận pixel override. Job MAY override ratio; precedence:

```text
job chain_band_width_ratio
    > server default_chain_band_width_ratio
```

Resolved result luôn lưu ratio và pixel width.

## 7.5 FOV validation

Tại mỗi anchor `s`, implementation phải tính giao tuyến từ centerline theo `-h` và `+h` tới biên ROI:

```text
available_left_extent(s)
available_right_extent(s)
```

Geometry job-level chỉ hợp lệ khi tại trigger:

```text
available_left_extent >= minimum_side_field_of_view_ratio * roi_width
available_right_extent >= minimum_side_field_of_view_ratio * roi_width
```

Bootstrap:

```text
minimum_side_field_of_view_ratio = 0.25
```

FOV hợp lệ ở job-level không đồng nghĩa negative evidence hợp lệ cho từng paddle. Per-frame FOV opportunity vẫn phải được tính.

Invalid remote geometry/ROI/FOV bị reject HTTP 422 trước khi tạo job khi có thể
validate từ request. Worker vẫn re-validate với decoded frame metadata; mismatch
phát hiện muộn fail job bằng stable `geometry_invalid_*` reason.

---

# 8. Model contract và capability manifest

## 8.1 Segmentation adapter

Mọi model hiện tại hoặc tương lai phải được bọc sau interface:

```python
class SegmentationAdapter(Protocol):
    def infer(
        self,
        roi_image: np.ndarray,
        *,
        source_frame_id: int,
        source_timestamp_sec: float,
    ) -> list[SegmentationInstance]:
        ...
```

`SegmentationInstance`:

```json
{
  "instance_id": 12,
  "source_frame_id": 105,
  "source_timestamp_sec": 3.5,
  "model_output_row_index": 41,
  "class_id": 0,
  "confidence": 0.93,
  "model_bbox_roi_xyxy": [120.0, 210.0, 850.0, 280.0],
  "model_bbox_crop_roi_xyxy": [120, 210, 850, 280],
  "binary_mask_roi": "<uint8 HxW>",
  "mask_probability_roi": null,
  "adapter_diagnostics": {}
}
```

`model_bbox_roi_xyxy` phải là bbox model sau coordinate restore nhưng trước khi
representative contour ghi đè.

Adapter current-model phải expose field này bằng một extension tương thích:

```python
Detection.model_bbox_roi_xyxy: tuple[float, float, float, float] | None = None
Detection.model_bbox_crop_roi_xyxy: tuple[int, int, int, int] | None = None
Detection.model_output_row_index: int | None = None
```

Legacy fields/bbox semantics giữ nguyên. Geometry-v2 startup phải fail
`model_contract_mismatch` nếu một trong ba field là `None`.

`model_bbox_crop_roi_xyxy` là exact clipped integer half-open slice
`[x1:x2, y1:y2]` đã dùng để zero mask ngoài bbox. Union bbox của foreground mask
không thể chứng minh vùng zero là model-background hay bị bbox crop và chỉ được
lưu như metric khác.

`model_output_row_index` phải được giữ từ raw decoded output qua confidence/class
filters và NMS; nó là stable tie-break/provenance, không phải index sau sorting.

## 8.2 Artifact manifest và system capability record

Artifact manifest chỉ mô tả sự thật intrinsic đọc được từ bytes của model. Nó
không chứa hành vi của adapter, postprocessor hoặc kết quả benchmark:

```json
{
  "schema_version": "model_artifact/1.0",
  "artifact_manifest_id": "<sha256-of-canonical-manifest-without-this-field>",
  "sha256": "ef05955f43c8db6d2ff76b72fb65806e69afe525e85d8486eeb2dfb7566dcd65",
  "format": "onnx",
  "onnx_ir_version": 7,
  "opset_imports": {"": 13},
  "input": {
    "name": "images",
    "dtype": "float32",
    "shape": [1, 3, 640, 640]
  },
  "outputs": [
    {"name": "output0", "dtype": "float32", "shape": [1, 300, 38]},
    {"name": "output1", "dtype": "float32", "shape": [1, 32, 160, 160]}
  ],
  "class_names": {"0": "white_bar"},
  "export_metadata_verbatim": {
    "producer": "Ultralytics 8.4.67",
    "task": "segment",
    "stride": 32,
    "imgsz": [640, 640],
    "end2end": true,
    "export_date": "2026-06-14"
  },
  "license_metadata_verbatim": {
    "license": "AGPL-3.0",
    "license_url": "https://ultralytics.com/license"
  }
}
```

Các giá trị metadata phải được copy verbatim từ artifact thực; ví dụ trên là record
đã kiểm chứng cho artifact 640 hiện tại. `artifact_manifest_id` được tính trên
canonical JSON sau khi bỏ chính field ID, nên không tự tham chiếu.

Capability quyết định production không được gắn với model SHA đơn độc. Nó gắn với
full evaluated system signature và operating domain:

```json
{
  "schema_version": "geometry_capabilities/1.0",
  "capability_record_hash": "<sha256-of-canonical-record-without-this-field>",
  "system_signature_hash": "<sha256-of-system_signature-object>",
  "system_signature": {
    "artifact_manifest_id": "<immutable-id>",
    "preprocess_fingerprint": "<canonical-hash>",
    "postprocess_fingerprint": "<conf-iou-mask-crop-class-hash>",
    "adapter_version": "current_yolo_seg_adapter/2.0",
    "adapter_build_commit": "fec1e6b587547e4bd111973039e041c487e05c43",
    "geometry_rule_version": "geometry_v2_rules/2.0.0",
    "algorithm_config_hash": "<canonical-hash>"
  },
  "runtime": {
    "fingerprint": "<canonical-hash>",
    "python": "3.12.3",
    "onnxruntime": "1.27.0",
    "execution_provider_order": ["CPUExecutionProvider"],
    "execution_provider_options": [{}],
    "session_options": {
      "intra_op_num_threads": "<resolved>",
      "inter_op_num_threads": "<resolved>",
      "execution_mode": "<resolved>",
      "graph_optimization_level": "<resolved>"
    },
    "opencv": "4.13.0",
    "numpy": "2.4.6",
    "os": "<production-os-version>",
    "architecture": "<production-architecture>"
  },
  "adapter_evaluation": {
    "mask_source": "detection.mask_roi_after_bbox_crop",
    "observed_emission_traits": {
      "single_fragment": true,
      "multi_component_mask": true,
      "multiple_instances_per_visual_candidate": true
    }
  },
  "validated_operating_domain": {
    "deployment_profile_id": "<immutable-id>",
    "camera_ids": [],
    "conveyor_line_ids": [],
    "calibration_ids": [],
    "frame_resolution_ranges": [],
    "roi_size_ranges": [],
    "side_fov_ratio_range": [],
    "centerline_roll_deg_range": [],
    "fps_range": [],
    "chain_speed_range": [],
    "lighting_profile_ids": [],
    "dataset_version": "<held-out-dataset-version>",
    "acceptance_report_id": "<immutable-id>"
  },
  "pipeline_support": {
    "temporal_fragment_pairing_implemented": true,
    "original_frame_only_voting": true
  },
  "validation": {
    "same_frame_center_topology": "provisional",
    "temporal_complementary_emission": "provisional",
    "side_geometry_validity": "provisional",
    "localized_side_break": "provisional",
    "angle_classification": "provisional",
    "single_side_localization": "unvalidated",
    "absence_as_negative_evidence": "unvalidated"
  },
  "production_enabled": {
    "same_frame_center_topology": false,
    "temporal_center_break": false,
    "side_geometry_validity": false,
    "localized_side_break": false,
    "angle_classification": false,
    "single_side_localization": false,
    "absence_as_negative_evidence": false
  }
}
```

Các observed emission traits chỉ chứng minh pipeline phải chịu được hình thái đó
trên fixtures/runtime đã chạy; chúng không cam kết recall/correctness.

Bootstrap phải tính lần lượt, không tạo vòng hash:

1. verify bytes model và tạo/lookup artifact manifest;
2. parse canonical geometry config rồi tính `algorithm_config_hash`;
3. resolve runtime/provider/session options và tính runtime fingerprint;
4. tạo `system_signature_hash` từ artifact + preprocess + postprocess + adapter +
   rule/config;
5. lookup immutable capability record có đúng system signature, runtime fingerprint
   và operating domain.

`capability_record_hash` tính sau cùng trên canonical record khi bỏ chính hash
field; `system_signature_hash` tính riêng trên object `system_signature`. Không
hash filename/path local hay wall-clock vào identity.

Geometry config không chứa hash của capability record; capability record mới là
record tham chiếu `algorithm_config_hash`. Runtime result lưu riêng cả hai hash.
`algorithm_config_hash` là SHA-256 của **projection thuật toán** từ
`geometry_v2.json` sau resolve default. Projection loại `model.artifact_path`
(locator local) và mọi deployment/capability binding; identity artifact nằm ở
artifact manifest. Không được đưa `capability_record_hash` vào config rồi lại đưa
config hash vào capability, vì đó là circular identity không canonicalize được.

Nếu không tìm thấy exact capability record, runtime/config/camera nằm ngoài
validated operating domain, hoặc provider options khác record:

- shadow/replay MAY chạy và lưu candidate diagnostics;
- production không được phát definitive physical status;
- nếu policy `outside_domain_policy="uncertain"`, mọi reportable event là
  `uncertain` với `operating_domain_not_validated`;
- nếu input geometry/runtime bản thân invalid, fail job theo reason code tương
  ứng, không dùng `uncertain` để che lỗi hệ thống.

Các boolean capability chỉ là resolved fields của record này, không phải
intrinsic fact của model. Không được dùng `absence_as_negative_evidence` như alias
cho khả năng định vị single-side.

## 8.3 Model tương lai

Model mới SHOULD ưu tiên một trong hai emission contract:

### Contract A — physical paddle union mask

Một physical paddle là một instance, kể cả mask gồm hai polygon/component rời.

### Contract B — fragment instances

Model phát fragment trái/phải riêng, có class hoặc side-neutral mask; adapter/observation builder pair chúng.

Model mới chỉ được thay vào production khi:

- có artifact hash;
- artifact manifest và full-system capability record;
- benchmark riêng;
- threshold riêng;
- regression test với downstream pipeline;
- không làm thay đổi canonical `PaddleResult`.

---

# 9. Pipeline tổng thể

```text
Original video frame
    ↓
Frame provenance + ROI crop
    ↓
Pinned segmentation adapter
    ↓
Binary mask normalization
    ↓
Connected-component extraction
    ↓
Cross-detection component deduplication
    ↓
Fragment feature extraction in (s, q)
    ↓
Per-frame PaddleObservation building
    ↓
Order-preserving 1D tracklet association
    ↓
Trigger event collection
    ↓
Bounded pre/post evidence buffers
    ↓
Tracklet finalization
    ↓
Offline physical-event fusion
    ↓
Independent evidence binning
    ↓
Center topology analysis
    ↓
Side integrity analysis
    ↓
Breakage decision
    ↓
Angle analysis if intact
    ↓
Canonical decision
    ↓
Evidence rendering
    ↓
Legacy-compatible result adapter
```

Pipeline có hai phần:

### Streaming bounded phase

- decode frame;
- inference;
- observations;
- tracklets;
- top-K evidence;
- không giữ toàn bộ video.

### Offline event-fusion phase

- dùng tracklet metadata và bounded evidence;
- merge complementary tracklets;
- cấp final paddle IDs;
- classify;
- render.

Offline phase không được yêu cầu load lại toàn bộ video vào RAM.

---

# 10. Frame provenance và timestamp

## 10.1 Source frame identity

Mỗi decoded original frame:

```json
{
  "source_frame_id": 105,
  "source_timestamp_sec": 3.466666667,
  "timestamp_source": "decoder_pts",
  "is_original": true
}
```

Bootstrap backend canonical là PyAV/FFmpeg, pin dependency `av` trong
`pyproject.toml` và `uv.lock`. Mỗi decoded video frame dùng:

```text
decoder_pts_sec = float(frame.pts * frame.time_base)

source_timestamp_sec =
    decoder_pts_sec - first_valid_decoder_pts_sec
```

Nếu `frame.pts is None`, backend MAY dùng timestamp best-effort do FFmpeg/PyAV
expose nếu giá trị đó có provenance rõ và qua cùng monotonic validation; không
dùng `CAP_PROP_PTS`, `CAP_PROP_POS_MSEC`, frame index hoặc average FPS để giả làm
decoder PTS.

PTS phải:

- finite;
- strictly increasing với
  `source_timestamp_sec[n] - source_timestamp_sec[n-1] >
  timestamp_epsilon_sec`;
- gắn với decoded frame vừa đọc;
- không dùng wall-clock processing time.

Equal/reversed PTS làm stream PTS unusable; không đưa `dt=0` vào Kalman và không
cho second vote. Backend phải chọn một timestamp source cho toàn job trong
preflight, không switch giữa PTS/index giữa chừng.

Nếu decoder không cung cấp usable PTS, chỉ được fallback:

```text
source_timestamp_sec =
    (source_frame_id - 1) / validated_fps

timestamp_source = "cfr_index_fallback"
```

khi container/decoder contract xác nhận constant-frame-rate và FPS finite/positive.
Average FPS metadata một mình không đủ chứng minh CFR.

Nếu không có usable PTS và không xác nhận được CFR:

```text
job failure
failure_reason = "invalid_video_timestamps"
```

Nếu dùng CFR fallback nhưng FPS invalid:

```text
failure_reason = "invalid_video_fps"
```

Result phải lưu `timestamp_source`. Bootstrap:

```text
timestamp_epsilon_sec = 1e-9
```

Chỉ non-finite/equal/reversed/wrapped timestamp mới invalid. Một positive gap lớn
hơn nominal frame period là hợp lệ với VFR hoặc dropped presentation frame:
tracker phải advance bằng exact `dt`, không fail, clamp hoặc chia nhỏ ngầm. Một
deployment MAY có separate maximum-gap policy để kết thúc track, nhưng không được
gọi positive timestamp gap là timestamp corruption.

Geometry-v2 dùng explicit decoder contract:

```python
class FrameSource(Protocol):
    def read(self) -> FrameRead:
        """Return exactly one tagged result: FRAME, EOF, or DECODE_ERROR."""


class FrameRecord:
    source_frame_id: int
    decoder_pts_sec: float | None
    source_timestamp_sec: float
    timestamp_source: str
    image: np.ndarray
```

PyAV implementation phải:

- demux/decode theo presentation order;
- trả `EOF` chỉ khi iterator kết thúc sạch;
- map exception giải mã/container I/O thành `DECODE_ERROR` kèm frame/packet
  provenance;
- không biến exception thành EOF;
- đóng container ở mọi exit path.

`cv2.VideoCapture` chỉ được dùng làm backend thay thế nếu conformance test chứng
minh:

- usable timestamps cho supported containers;
- CFR fallback confirmation;
- expected EOF phân biệt được premature decode failure.

Legacy `open_video_source` không tự thỏa contract này chỉ vì `read()` trả false.

Regression bắt buộc cho `data/raw_data/vid_1.mp4`:

- canonical PTS strictly increasing;
- frame 33 → 34 có gap khoảng `0.066667 sec` và vẫn hợp lệ;
- frame 468/469 có hai canonical timestamps khác nhau;
- không tái tạo timestamp bằng `frame_id / 29.96798`;
- OpenCV `CAP_PROP_PTS` duplicate quan sát ở vùng này không được dùng làm
  canonical source.

## 10.2 Frame-size stability

Mọi frame phải khớp `frame_width`, `frame_height` đã khai báo.

Nếu một frame lệch kích thước:

- không silently resize;
- ghi diagnostics;
- fail job với `video_geometry_changed`, trừ khi policy rõ ràng cho phép reject riêng frame;
- bootstrap policy là fail job.

## 10.3 Evidence independence

Frame evidence được de-correlate theo thời gian rồi mới cấp logical time-bin IDs.

Bootstrap:

```text
minimum_evidence_spacing_frames = 2
minimum_evidence_spacing_sec = 0.05
```

Hai votes cùng evidence type phải thỏa đồng thời cả frame-ID delta và PTS delta.
Candidate selection chi tiết ở Mục 16.4.

Một frame có thể đóng góp nhiều evidence type khác nhau, nhưng mỗi type chỉ một vote.

---

# 11. Chuẩn hóa mask và connected components

## 11.1 Không dùng representative contour làm nguồn hình học

Trong `geometry_v2`, các trường sau từ pipeline cũ chỉ được dùng để chẩn đoán hoặc
compatibility:

- `Detection.contour_frame`;
- `Detection.centroid_frame_xy`;
- `Detection.bbox_roi_xyxy` đã được tính lại từ representative contour.

Nguồn hình học canonical là:

```text
Detection.mask_roi
```

Lý do: với `contour_mode = "largest"`, ba trường representative ở trên có thể chỉ
đại diện cho một cánh, trong khi `mask_roi` vẫn có thể chứa nhiều connected
components bên trong bounding box dự đoán.

Không được suy ra rằng phần không có trong `mask_roi` chắc chắn không tồn tại ngoài
đời thật. Với model hiện tại, `mask_roi` đã bị crop theo predicted bounding box.

## 11.2 Hai mask phục vụ hai mục đích

Mỗi detection tạo hai view logic:

```text
topology_mask
geometry_mask
```

`topology_mask`:

- threshold đúng theo model profile;
- loại connected component quá nhỏ;
- không closing;
- không opening;
- không dilation;
- không erosion;
- không fill khe nối giữa hai component;
- dùng để xác định connected/disconnected.

`geometry_mask`:

- bắt đầu từ component đã được chấp nhận trong `topology_mask`;
- MAY fill lỗ nhỏ nằm hoàn toàn bên trong cùng một component;
- MAY smooth riêng biên component với kernel nhỏ;
- MUST NOT nối hai component vốn rời nhau;
- dùng để fit đường, endpoint và angle.

Mọi phép morphology trên `geometry_mask` phải log:

```json
{
  "geometry_mask_modified": true,
  "operation": "fill_small_internal_holes",
  "affected_area_px": 12
}
```

Bootstrap không dùng closing/opening kể cả trên `geometry_mask`.

## 11.3 Connected-component extraction

Dùng 8-connectivity trên từng `mask_roi`.

Cho:

```text
A_roi      = roi_width * roi_height
A_instance = tổng số foreground pixel của mask instance
```

Ngưỡng component khởi đầu:

```text
minimum_component_area_px =
    max(
        16,
        0.00002 * A_roi,
        0.005 * A_instance
    )
```

Component nhỏ hơn ngưỡng:

- không tham gia geometry;
- không tham gia topology;
- được đếm vào diagnostics;
- không bị xóa khỏi raw evidence nếu debug artifact được bật.

Ngưỡng này là bootstrap, không phải ground truth. Việc hiệu chỉnh production phải
dựa trên validation set có nhãn.

## 11.4 Component record

Mỗi component hợp lệ tạo record:

```json
{
  "component_id": "f000000105-d02-c01",
  "source_frame_id": 105,
  "source_detection_id": "f000000105-d02",
  "source_detection_score": 0.8731,
  "class_id": 0,
  "area_px": 2118,
  "bbox_roi_xyxy": [331, 192, 454, 226],
  "centroid_sq": [382.4, 209.1],
  "s_anchor": 211.7,
  "s_anchor_sigma": 1.8,
  "q_median": -62.4,
  "side_hint": "left",
  "touches_roi_boundary": false,
  "duplicate_of": null
}
```

`centroid_sq` chỉ là diagnostic; tracker không dùng nó làm measurement chính.

ID phải deterministic:

1. detections được sort theo `score desc`, rồi raw model bbox lexicographic, rồi
   `model_output_row_index`;
2. components trong detection được sort theo `s_anchor`, `q_median`,
   `area desc`;
3. ID được cấp sau khi sort.

## 11.5 Fragment anchor

Với mỗi component có tập pixel `C`, đặt:

```text
distance_to_chain(P) = max(0, abs(q(P)) - chain_band_half_width)
K = max(5, ceil(0.10 * |C|))
D_K = K-th smallest value of distance_to_chain
N0 = {P in C | distance_to_chain(P) <= D_K}
```

Phải lấy toàn bộ ties ở `D_K`; không cắt tie bằng thứ tự `s`, vì một whole/center
component có thể có nhiều pixels `distance=0` và bị kéo anchor về phía trên.

```text
anchor_histogram_bin_px = max(1.0, 0.002 * H)
```

Tạo histogram có origin cố định tại `s_min` và trọng số đều của `s(N0)`. Tính
tổng trên sliding window ba bins.
Chọn window có tổng lớn nhất; tie-break:

1. center gần `median(s(C))` hơn;
2. center nhỏ hơn.

Gọi tập pixels trong winning window là `N`.

```text
s_anchor = median({s(P) | P in N})
```

Nếu hai local-peak windows không overlap, cách nhau hơn
`maximum_anchor_spread_ratio * H`, và peak thứ hai đạt ít nhất:

```text
secondary_anchor_peak_ratio = 0.80
```

so với peak đầu, component có nhiều anchor khả dĩ và phải gắn
`anchor_quality="ambiguous"`/`MULTI_PADDLE_MERGED`.

Độ bất định:

```text
s_anchor_sigma =
    max(
        1.0 px,
        1.4826 * MAD({s(P) | P in N})
    )
```

Nếu winning `N` vẫn trải rộng:

```text
P95(s(N)) - P5(s(N)) > maximum_anchor_spread_ratio * H
```

thì component được gắn:

```text
anchor_quality = "ambiguous"
```

Bootstrap:

```text
maximum_anchor_spread_ratio = 0.05
secondary_anchor_peak_ratio = 0.80
H = s_max - s_min
```

## 11.6 Side hint

Cho:

```text
B = chain_band_half_width
q50 = median(q(component pixels))
q05 = P5(q(component pixels))
q95 = P95(q(component pixels))
```

Phân loại sơ bộ:

```text
if q95 < -B: LEFT
else if q05 > +B: RIGHT
else if q05 <= -B and q95 >= +B: SPANS_BOTH
else: CENTER
```

`side_hint` không phải kết luận tình trạng cánh. Nó chỉ mô tả vị trí component.

## 11.7 Boundary flags

Mỗi component phải lưu việc nó chạm:

- ROI left;
- ROI right;
- ROI top;
- ROI bottom;
- predicted detection bbox boundary;
- chain band.

Predicted-bbox boundary flag bắt buộc dùng
`SegmentationInstance.model_bbox_roi_xyxy` raw. Không dùng representative contour
bbox hoặc mask-union bbox.

Bootstrap:

```text
boundary_margin_px = max(2, round(0.003 * max(roi_width, roi_height)))
```

Component chạm biên ROI không được dùng làm negative evidence về phần bị thiếu ở
phía biên đó.

---

# 12. Khử trùng lặp component giữa các detections

## 12.1 Vì sao IoU đơn thuần không đủ

Trên dữ liệu kiểm chứng, hai detections cùng visual candidate có:

```text
predicted-box IoU       ≈ 0.29587
full detection-mask IoU ≈ 0.439
full-mask IoS           ≈ 0.909
```

Sau connected-component split, cặp component trùng tương ứng có:

```text
component IoU ≈ 0.85728
component IoS ≈ 0.93766
```

Vì vậy predicted-box NMS `0.5` và pre-split full-mask IoU đều có thể không loại
đúng duplicate/complementary instance. `geometry_v2` phải split trước rồi dedup ở
component level; ở sample này cả component-IoU và IoS đều bắt được cặp trùng.

## 12.2 Metrics

Với hai component masks `A`, `B`:

```text
intersection = |A ∩ B|
union        = |A ∪ B|
IoU          = intersection / union
IoS          = intersection / min(|A|, |B|)
```

Hai component chỉ được xét duplicate khi:

- cùng source frame;
- cùng class hợp lệ;
- side hints giống nhau trong `{LEFT, RIGHT, CENTER, SPANS_BOTH}`;
- `abs(s_anchor_A - s_anchor_B) <= duplicate_anchor_gate_ratio * H`.

Bootstrap:

```text
duplicate_anchor_gate_ratio = 0.03
```

Duplicate nếu một trong hai điều kiện:

```text
IoS >= 0.85
```

hoặc:

```text
IoU >= 0.70
```

Không dedup hai component trái/phải chỉ vì anchor gần nhau.

Cross-type containment là ngoại lệ một chiều:

- nếu một `SPANS_BOTH` component đã qua bridge `PRESENT`;
- và nó chứa ít nhất `0.85` area của một auxiliary LEFT/RIGHT component;

thì auxiliary component được alias vào whole observation. Không bao giờ để
single-side component loại `SPANS_BOTH` component theo score.

Ngoại lệ này được resolve ở observation consolidation sau bridge test, không áp
dụng trong initial component-dedup pass.

## 12.3 Canonical suppression và representative

Không tạo transitive duplicate group bằng connected-component closure: `A`
duplicate `B`, `B` duplicate `C` không chứng minh `A` duplicate `C`.

Canonical NMS:

1. sort component candidates theo:
   - không chạm ROI/predicted-crop boundary trước;
   - area lớn hơn trước;
   - source detection score cao hơn;
   - anchor uncertainty thấp hơn;
   - component ID lexicographic;
2. duyệt theo thứ tự;
3. candidate chỉ bị suppress/alias nếu đạt duplicate predicate trực tiếp với
   chính một accepted representative;
4. nếu đạt nhiều representatives, chọn representative có IoS lớn nhất, rồi IoU
   lớn nhất, rồi representative order ở bước 1.

Area/truncation đứng trước score vì high-score subset không được loại một mask đầy
đủ hơn rồi tạo false shortness. Score chỉ là tie-break sau completeness.

Không union masks trong bootstrap, vì union có thể đưa false-positive pixels vào
topology. Representative giữ:

```json
{
  "duplicate_aliases": ["f000000105-d03-c01"],
  "duplicate_support_count": 2
}
```

Nếu duplicate chỉ pass IoS containment nhưng `IoU < 0.70`, hai masks là
`containment_variants`, không được giả định geometry tương đương:

- representative fuller/untruncated được dùng cho identity/render;
- chạy frame analyzer trên representative và contained alternative;
- nếu center/side state khác nhau, frame geometry là `UNKNOWN` với
  `duplicate_mask_geometry_disagreement`;
- aliases không tạo observation/vote riêng;
- score cao hơn không được tự giải quyết disagreement.

Duplicate aliases:

- không tạo observation riêng;
- không tạo vote riêng;
- MAY làm tăng diagnostic support;
- MUST NOT làm hai source detections trong cùng frame trở thành hai independent
  evidence frames.

Dedup không được làm mất raw provenance. Canonical component phải giữ cho từng
alias:

```text
source component ID
source detection ID
model output row index
raw float model bbox
rasterized crop bbox
source mask/component membership
pairwise IoU/IoS
```

Observation/topology builder có thể nhờ provenance này biết canonical left và
right từng cùng xuất hiện trong một source instance, dù left winner pixels đến từ
detection khác. Pixel mask winner không được thay thế provenance graph.

---

# 13. Xây dựng observation trong một frame

## 13.1 Observation types

Canonical enum:

```text
CONNECTED_WHOLE
DISCONNECTED_BOTH
LEFT_ONLY
RIGHT_ONLY
CENTER_ONLY
MULTI_PADDLE_MERGED
AMBIGUOUS
```

Ý nghĩa:

- `CONNECTED_WHOLE`: một physical candidate có foreground liên tục từ cánh trái,
  qua chain corridor, tới cánh phải.
- `DISCONNECTED_BOTH`: có component trái và phải phù hợp cùng một paddle nhưng
  không có bridge component.
- `LEFT_ONLY`: chỉ có fragment trái đủ điều kiện.
- `RIGHT_ONLY`: chỉ có fragment phải đủ điều kiện.
- `CENTER_ONLY`: chỉ có bằng chứng trong/giáp chain band, không đủ cánh.
- `MULTI_PADDLE_MERGED`: một instance/component có nhiều anchor dọc tách biệt,
  có khả năng chứa nhiều paddle.
- `AMBIGUOUS`: pairing không duy nhất hoặc hình học không đủ an toàn.

## 13.2 Candidate grouping theo anchor

Components trong frame được sort theo `s_anchor`.

Hai component có thể thuộc cùng observation khi:

```text
abs(s_anchor_A - s_anchor_B)
    <= same_frame_anchor_gate_ratio * H
```

Bootstrap:

```text
same_frame_anchor_gate_ratio = 0.03
```

Pair hợp lệ ưu tiên:

- một LEFT và một RIGHT;
- một SPANS_BOTH;
- LEFT/RIGHT cùng một CENTER component nếu topology thực sự kết nối;
- source detections khác nhau được phép sau dedup.

Không được ghép khi:

- một component đang là candidate tốt tương đương cho hai paddle lân cận;
- anchor của một bên có quality `ambiguous`;
- interval theo `s` của candidate groups overlap theo cách không có nghiệm duy
  nhất;
- implied paddle width vượt FOV hoặc giới hạn cơ khí đã cấu hình.

## 13.3 Ghép theo thứ tự, không ghép tham lam

LEFT candidates và RIGHT candidates được sort theo `s_anchor`. Pairing dùng dynamic
programming order-preserving, với các operation:

```text
MATCH(left, right)
LEFT_UNMATCHED(left)
RIGHT_UNMATCHED(right)
```

Match cost:

```text
C_pair =
    abs(s_left - s_right) / H
  + pairing_uncertainty_weight
    * sqrt(sigma_left^2 + sigma_right^2) / H
```

Bootstrap:

```text
unmatched_cost = 0.05
```

Shape/FOV/mechanical plausibility là hard eligibility gates ở Mục 13.2, không đưa
một `shape_incompatibility` chưa chuẩn hóa vào cost.

Chỉ match nếu qua anchor gate. Order-preserving pairing đảm bảo hai paddle không
chéo thứ tự nhau trong cùng frame.

Recurrence canonical với `L[0:i]`, `R[0:j]`:

```text
DP[0,0] = {empty path, cost 0}

DP[i,j] = best_two_distinct_paths(
    DP[i-1,j-1] + MATCH(L[i-1], R[j-1])   # chỉ khi eligible
    DP[i-1,j]   + LEFT_UNMATCHED(L[i-1])
    DP[i,j-1]   + RIGHT_UNMATCHED(R[j-1])
)
```

Mỗi cell giữ hai full operation paths khác nhau, không chỉ hai scalar costs.
Mỗi operation cost được đổi sang integer theo Mục 0.1 trước khi cộng. Rank path:

1. total integer cost nhỏ hơn;
2. nhiều `MATCH` hơn;
3. ít unmatched hơn;
4. full ordered operation tuple
   `(opcode, left_component_id, right_component_id)` lexicographic.

Không collapse hai paths có cùng cost. Đặt `best_int`, `second_int` là total
quantized integer costs của hai full paths. Nếu final runner-up gần best:

```text
(second_int - best_int) * cost_quantization
    < pairing_ambiguity_margin
```

thì “candidates liên quan” chính xác là symmetric difference giữa hai full
operation paths. Chỉ các component/links trong difference bị taint
`AMBIGUOUS`; common matches giữ nguyên. Không ép một link khác biệt thành
definitive.

Bootstrap:

```text
pairing_ambiguity_margin = 0.01
```

## 13.4 Xác định connected trong một frame

`CONNECTED_WHOLE` chỉ được tạo khi chính `topology_mask` cung cấp đường foreground
liên tục thỏa toàn bộ:

- có pixels ngoài chain band ở cả `q < -B` và `q > +B`;
- cùng một 8-connected component đi qua corridor trung tâm;
- component giao left inner gate, chain band và right inner gate;
- coverage ngang qua corridor đạt ngưỡng tại anchor tương ứng;
- không dùng morphology để tạo bridge.

Một detection có một connected component lớn chưa mặc nhiên là
  `CONNECTED_WHOLE`; nó vẫn phải qua bridge test ở Mục 19.

## 13.5 Xác định disconnected trong một frame

`DISCONNECTED_BOTH` chỉ được tạo khi:

- có ít nhất một LEFT và một RIGHT component;
- pair là duy nhất;
- anchor gate đạt;
- cả hai component có support/quality hợp lệ;
- bridge test trả `ABSENT`, không phải `UNKNOWN`;
- FOV cho hai phía hợp lệ.

Nếu left/right pairing là duy nhất nhưng bridge test `UNKNOWN`, tạo một
`AMBIGUOUS` composite observation chứa cả hai component IDs và một anchor chung.
Không tách nó thành hai tracks trong cùng frame. Observation này hỗ trợ motion
tracking nhưng không vote `DISCONNECTED_BOTH`.

Nếu chính pairing cũng ambiguous, không tạo composite definitive và không dùng
frame đó cho classification evidence.

## 13.6 Một instance chứa nhiều paddle

Nếu component có nhiều cụm pixels gần chain band với các center theo `s` cách nhau:

```text
> multi_anchor_separation_ratio * H
```

Bootstrap:

```text
multi_anchor_separation_ratio = 0.08
```

thì:

- đánh dấu `MULTI_PADDLE_MERGED`;
- không tự cắt mask bằng watershed trong release bootstrap;
- không dùng làm evidence quyết định;
- MAY dùng predicted anchor clusters làm tracking hints với quality thấp;
- log `merged_instance_requires_review`.

Việc tự split instance chỉ được bật sau khi có validation riêng.

## 13.7 Observation schema

```json
{
  "observation_id": "f000000105-o02",
  "source_frame_id": 105,
  "source_timestamp_sec": 3.466666667,
  "type": "DISCONNECTED_BOTH",
  "s_anchor": 211.9,
  "s_anchor_sigma": 2.1,
  "component_ids": [
    "f000000105-d01-c01",
    "f000000105-d01-c02"
  ],
  "left_component_id": "f000000105-d01-c01",
  "right_component_id": "f000000105-d01-c02",
  "center_component_id": null,
  "model_scores": [0.8731],
  "fov": {
    "left_opportunity": true,
    "right_opportunity": true
  },
  "evidence_gate_margin": 0.14,
  "reject_reasons": []
}
```

Một `observation_id` chỉ xuất hiện trong tối đa một online track tại một thời
điểm.

---

# 14. Online tracking theo chain coordinate

## 14.1 Mục tiêu

Tracker phải giữ cùng identity khi model luân phiên:

```text
frame n:     LEFT_ONLY
frame n + 1: RIGHT_ONLY
frame n + 2: LEFT_ONLY
```

Tracker không được dựa vào centroid của cánh, vì centroid trái/phải có thể lệch
nhau hơn giới hạn jump của tracker cũ.

Measurement canonical:

```text
z = observation.s_anchor
```

## 14.2 State model

Mỗi track có Kalman state 1D constant velocity:

```text
x = [s, v]^T
```

Transition với `dt`:

```text
F = [[1, dt],
     [0,  1]]
```

Measurement:

```text
Hk = [1, 0]
z  = s_anchor
```

```text
R =
    max(
        s_anchor_sigma^2,
        minimum_measurement_sigma_px^2
    )
```

Process noise dùng white acceleration model:

```text
Q = sigma_a^2 *
    [[dt^4 / 4, dt^3 / 2],
     [dt^3 / 2, dt^2]]
```

Bootstrap:

```text
sigma_acceleration_ratio_per_sec2 = 0.15
sigma_a = sigma_acceleration_ratio_per_sec2 * H  # px/sec^2
minimum_measurement_sigma_px = 1.0
```

Mọi state/covariance/reduction dùng `float64`. Với một initialized track trên mỗi
original frame timestamp, predict đúng một lần:

```text
x_minus = F x_plus_previous
P_minus = F P_plus_previous F^T + Q

innovation = z - Hk x_minus
S = Hk P_minus Hk^T + R
NIS = innovation^2 / S
K = P_minus Hk^T / S

x_plus = x_minus + K * innovation
I_KH = I - K Hk
P_plus = I_KH P_minus I_KH^T + K R K^T  # Joseph form
P_plus = 0.5 * (P_plus + P_plus^T)
```

`S` phải finite và `> 0`; covariance non-finite/non-PSD vượt numeric tolerance làm
track geometry invalid, không silently reset. Tất cả association candidates của
cùng track/frame dùng chung `x_minus/P_minus`. Sau DP:

- MATCH commit đúng một update;
- MISS commit `x_plus=x_minus`, `P_plus=P_minus`;
- không predict lại cho từng candidate;
- gap nhiều frame vì vậy tích lũy process noise đúng theo canonical PTS `dt`.

Mọi parameter resolved theo pixel phải được lưu trong result diagnostics.

### Deterministic initialization

Observation đầu tiên chỉ tạo tentative seed:

```text
seed = (s1, sigma1, t1)
kalman_initialized = false
```

Không giả vờ biết velocity và không dùng NIS cho first-to-second match.

Observation thứ hai phải qua:

```text
0 < dt = t2 - t1 <= maximum_track_gap_sec
abs(s2 - s1) <= 0.08 * H
reverse_displacement <= 0.015 * H
```

Sau đó:

```text
v2 = (s2 - s1) / dt
x2 = [s2, v2]^T

var_s1 = max(sigma1^2, minimum_measurement_sigma_px^2)
var_s2 = max(sigma2^2, minimum_measurement_sigma_px^2)

P2 = [
  [var_s2,                 var_s2 / dt],
  [var_s2 / dt, (var_s1 + var_s2) / dt^2]
]
```

Để tránh covariance suy biến:

```text
P2[1,1] =
    max(
        P2[1,1],
        (minimum_velocity_sigma_ratio_per_sec * H)^2
    )
```

Bootstrap:

```text
minimum_velocity_sigma_ratio_per_sec = 0.02
```

Từ observation thứ ba trở đi mới dùng Kalman predict/NIS canonical.

## 14.3 Motion direction

Bootstrap hỗ trợ:

```text
motion_direction = "positive_s"
```

Tức paddle đi theo chiều `top -> bottom`.

Nếu dây chuyền chạy ngược, job phải cung cấp:

```text
motion_direction = "negative_s"
```

Khi negative, implementation chuẩn hóa:

```text
s_motion = -s
v_motion = -v
```

để toàn bộ gate phía sau vẫn dùng chuyển động dương.

Không tự đoán direction từ vài frame đầu nếu không có mode calibration rõ ràng.

## 14.4 Hard gates

Track–observation match chỉ hợp lệ khi toàn bộ đạt:

```text
abs(innovation) <= 0.08 * H
reverse_displacement <= 0.015 * H
time_since_last_match <= 0.35 sec
```

Trong đó:

```text
reverse_displacement =
    max(0, s_motion_last_matched_observation - s_motion_observation)
```

Không tính reverse displacement từ current predicted state vì prediction noise có
thể che một observation thật sự chạy ngược.

Với Kalman đã initialized, thêm:

```text
NIS = innovation^2 / innovation_variance <= 9.0
```

`innovation_variance` chính là `S` trong Joseph/Kalman contract Mục 14.2.

First-to-second seed match dùng initialization gates Mục 14.2, không tính NIS.

Ngoài ra:

- observation không phải duplicate đã tiêu thụ;
- type composition không mâu thuẫn vật lý;
- source frame tăng nghiêm ngặt;
- một track không nhận hai observations có cùng source frame.

Composition tương thích bao gồm:

- LEFT_ONLY ↔ RIGHT_ONLY;
- LEFT_ONLY/RIGHT_ONLY ↔ DISCONNECTED_BOTH;
- single-side ↔ CONNECTED_WHOLE;
- các type hợp lệ khác nếu anchor và order gates đạt.

Không áp hard gate theo bbox area giữa LEFT_ONLY và RIGHT_ONLY.

## 14.5 Association cost

Với pair đã qua hard gates:

```text
C_match =
    w_nis       * min(NIS / 9, 1)
  + w_anchor    * abs(innovation) / (0.08 * H)
  + w_type      * type_transition_cost
```

Bootstrap:

```text
w_nis = 0.55
w_anchor = 0.30
w_type = 0.15
```

First-to-second seed cost:

```text
C_seed =
    0.80 * abs(s2 - s1) / (0.08 * H)
  + 0.20 * type_transition_cost
```

`type_transition_cost`:

```text
same type                         0.00
LEFT_ONLY <-> RIGHT_ONLY          0.05
single-side <-> both/whole        0.03
both/whole <-> both/whole         0.00
ambiguous                         0.50
incompatible                      INF
```

## 14.6 Order-preserving sequence alignment

Trong mỗi frame:

1. predicted active tracks được sort theo predicted `s_motion`;
2. observations được sort theo measured `s_motion`;
3. giải dynamic programming với operations:

```text
MATCH(track_i, observation_j)
MISS_TRACK(track_i)
NEW_TRACK(observation_j)
```

Không dùng unconstrained Hungarian làm canonical bootstrap, vì paddle trên cùng
một xích không vượt qua nhau. Order-preserving DP loại nhiều identity swaps.

Áp đúng recurrence/top-2-path contract Mục 13.3 sau khi thay:

```text
L -> predicted tracks
R -> current observations
MATCH -> C_match/C_seed
LEFT_UNMATCHED -> MISS_TRACK
RIGHT_UNMATCHED -> NEW_TRACK
```

Tracks sort theo `(predicted_s_motion, track_id)`; observations sort theo
`(measured_s_motion, observation_id)`. Full operation tuple dùng
`(opcode, track_id, observation_id)`. Ineligible match không được đưa vào
recurrence.

Bootstrap:

```text
miss_track_cost = 0.65
new_track_cost = 0.65
```

Tie-break deterministic:

1. integer total cost theo `cost_quantization`;
2. nhiều MATCH hơn;
3. ít NEW_TRACK hơn;
4. full ordered operation tuple lexicographic nhỏ hơn.

Đặt `best_int`, `second_int` là total quantized integer costs của hai association
paths. Bootstrap `association_ambiguity_margin = 0.03`. Nếu:

```text
(second_int - best_int) * cost_quantization
    < association_ambiguity_margin
```

thì:

- vẫn có thể cập nhật tracking bằng best path;
- tracks/observations trong symmetric difference của hai full paths nhận
  `association_ambiguous = true`;
- evidence của observation đó không được dùng để kết luận definitive;
- ambiguity được lưu vào diagnostics.

Bootstrap ambiguity taint là sticky:

```text
track.identity_ambiguity_tainted = true
```

nếu ambiguity xảy ra từ first event-lifetime observation tới final crossing/fusion.
Taint truyền qua mọi merged tracklet và làm final event `uncertain`.

Không tự clear chỉ vì các frame sau “trông hợp lý”. Một version tương lai chỉ được
clear khi có offline multi-hypothesis solver chứng minh một nghiệm duy nhất với
configured margin và lưu proof; bootstrap chưa có solver đó.

## 14.7 Track lifecycle

States:

```text
TENTATIVE
CONFIRMED
LOST
FINALIZABLE
FINALIZED
REJECTED
```

Bootstrap:

```text
minimum_track_hits = 2
maximum_track_gap_sec = 0.35
minimum_track_duration_sec = 0.04
```

Rules:

- observation đầu tiên tạo `TENTATIVE`;
- đủ hits ở independent source frames và đủ minimum duration thì `CONFIRMED`;
- không match nhưng chưa quá gap thì `LOST`;
- quá gap thì `FINALIZABLE`; evidence retention không quyết lifecycle;
- tentative không đủ hits trở thành `REJECTED`;
- finalization không cấp `paddle_id` ngay; phải qua offline event fusion.

## 14.8 Track schema

```json
{
  "track_id": 17,
  "state": "CONFIRMED",
  "first_source_frame_id": 92,
  "last_source_frame_id": 131,
  "first_timestamp_sec": 3.033333333,
  "last_timestamp_sec": 4.333333333,
  "kalman": {
    "s": 287.2,
    "v": 142.7,
    "covariance": [[3.1, 0.2], [0.2, 6.8]]
  },
  "observation_ids": [
    "f000000092-o01",
    "f000000094-o01"
  ],
  "composition_counts": {
    "LEFT_ONLY": 4,
    "RIGHT_ONLY": 3,
    "DISCONNECTED_BOTH": 1,
    "CONNECTED_WHOLE": 0
  },
  "crossing_time_sec": 3.72,
  "crossing_time_sigma_sec": 0.018
}
```

---

# 15. Trigger, crossing time và physical-event candidate

## 15.1 Trigger semantics

Tracklet giữ trigger history nhưng chưa tự trở thành physical event. Nó là
provisional fusion candidate khi observation/predicted trajectory hợp lệ giao
trigger strip theo đúng motion direction.

Canonical crossing point:

```text
s_trigger = trigger_center_s
```

Nguồn crossing canonical ưu tiên hai accepted original-frame measurements kề nhau
`(t0,s0,R0)`, `(t1,s1,R1)` bao quanh `s_trigger`, không trộn filtered state vào
phép interpolation:

```text
alpha = (s_trigger - s0) / (s1 - s0)
tau = t0 + alpha * (t1 - t0)

var_tau =
    ((t1 - t0) / (s1 - s0))^2
    * ((1 - alpha)^2 * R0 + alpha^2 * R1)

sigma_tau = sqrt(max(0, var_tau))
```

Yêu cầu `0 <= alpha <= 1`, `s1 != s0`, hai measurements đã qua identity gates và
đúng motion order. Nếu không có bracket, chỉ dùng posterior Kalman tại `(t,s,v,P)`
để extrapolate:

```text
tau = t + (s_trigger - s) / v
J = [-1 / v, -(s_trigger - s) / v^2]
var_tau = max(0, J P J^T)
sigma_tau = sqrt(var_tau)
```

với điều kiện:

```text
abs(v) >= minimum_velocity_ratio_per_sec * H
abs(tau - t) <= maximum_crossing_extrapolation_sec
sigma_tau <= maximum_crossing_sigma_sec
```

Bootstrap:

```text
minimum_velocity_ratio_per_sec = 0.05
maximum_crossing_extrapolation_sec = 0.10
maximum_crossing_sigma_sec = 0.05
```

Không đủ điều kiện:

```text
crossing_time = null
reason = "crossing_time_unresolved"
```

Tracklet vẫn được đưa vào fusion nếu trigger histories của một candidate cluster
có thể cùng tạo exact bracket; sau mỗi merge, crossing time phải được recompute từ
sorted union original measurements theo công thức interpolation trên. Extrapolation
không biến một cluster thiếu BEFORE hoặc AFTER observation thành reportable.

## 15.2 Trigger state

Mỗi tracklet lưu state của từng accepted observation:

```text
BEFORE_TRIGGER
IN_TRIGGER
AFTER_TRIGGER
```

Một tracklet chỉ được chuyển theo direction hợp lệ. Reverse jitter trong tolerance
không làm tạo event lần hai. Tuy nhiên full transition được kiểm **sau fusion**
trên union histories:

```text
BEFORE_TRIGGER -> IN_TRIGGER -> AFTER_TRIGGER
```

Không yêu cầu từng tracklet đủ ba state. Ví dụ hợp lệ:

```text
tracklet A: BEFORE -> IN
tracklet B: IN -> AFTER
fused event A+B: BEFORE -> IN -> AFTER
```

Direct `BEFORE -> AFTER` giữa hai accepted original measurements cũng chứng minh
traverse nếu segment trong `(t,s)` cắt cả hai strip boundaries, crossing
interpolation/sigma pass và không có identity competitor.

Tại clean EOF, worker phải flush mọi `CONFIRMED`/`LOST` tracklet vào fusion, không
drop vì chưa `FINALIZABLE`. Sau fusion, physical-event candidate chỉ reportable
khi:

- union original observations có chronological, motion-monotonic evidence trước
  và sau strip;
- trajectory collectively traverse strip;
- crossing time/sigma hợp lệ;
- fusion/identity và event cardinality là unique.

Nếu video bắt đầu khi union event đã ở `IN_TRIGGER/AFTER_TRIGGER`, hoặc kết thúc
trước khi union có `AFTER_TRIGGER`:

- đánh dấu `partial_start_event` hoặc `partial_end_event`;
- không cấp reportable `paddle_id`;
- không tính vào total/normal/defect;
- giữ diagnostics để audit.

Việc back-extrapolate crossing time không biến partial event thành reportable ở
bootstrap.

`provisional_event_id` của tracklet:

```text
pe-{primary_track_id:06d}
```

Nó chỉ là internal ID. `paddle_id` cuối cùng chỉ cấp sau fusion + collective
trigger validation.

## 15.3 Evidence window

Mọi original frame observation đã:

- nằm trong lifetime từ first accepted tracklet observation đến final collective
  AFTER/track expiry;
- uniquely associated với event;
- qua FOV/geometry/analyzer-specific hard gates;

MAY vote classification. Không giới hạn eligibility chỉ ở trigger strip hoặc
`±0.20H`; cần giữ evidence trước/trong/sau trigger để chịu được alternating
left/right emission.

Vùng ưu tiên chỉ dùng cho heavy-artifact retention/ranking:

```text
preferred_evidence_window_half_height_ratio = 0.20
preferred iff
    abs(s_anchor - s_trigger)
        <= preferred_evidence_window_half_height_ratio * H
```

Track giữ bounded pre-trigger ring metadata, post-trigger metadata và top-K
artifacts theo Mục 16. Frame ngoài preferred window vẫn vote nếu qua hard gates;
distance-to-trigger chỉ là deterministic rank/tie-break, không là evidence gate.

Mọi evidence phải lưu khoảng cách normalized tới trigger để chọn best snapshot.

## 15.4 Không đếm paddle từ detection count

Các giá trị sau không phải paddle count:

- số detection trong frame;
- số connected components;
- số online tracklets trước fusion;
- số snapshot.

Paddle count canonical:

```text
number of finalized event hypotheses sau offline fusion và trigger validation,
nhưng chỉ được công bố `total_bars` exact khi count_certified=true
```

Nếu có `identity_conflict_group` với `possible_event_count_min != max`, result
phải lưu count bounds thay vì suy diễn paddle count; export report bị block theo
Mục 17.5/24.6.

---

# 16. Bounded evidence store

## 16.1 Mục tiêu

Không giữ toàn bộ frame/mask của video trong RAM. Mỗi track giữ:

- scalar metadata cho observations trong event lifetime/candidate tracklet;
- top-K evidence items theo loại;
- compact masks/crops chỉ cho top-K;
- running robust statistics.

Scalar evidence metrics được tính ngay trên frame và có thể vote sau temporal
de-correlation dù crop/mask của item đó không nằm trong top-K. `top-K` giới hạn
heavy image artifacts, không giới hạn logical vote candidates.

Canonical internal binary-mask artifact `bbox_local_rle_v1`:

- bbox integer half-open ROI coordinates;
- crop flattened row-major/C order;
- run counts alternate `0,1,0,1...`, bắt đầu bằng zero-run (được phép count `0`);
- counts là unsigned 32-bit little-endian;
- tổng counts đúng bằng crop width × height;
- header canonical JSON chứa bbox, shape, order, start value và encoding version;
- SHA-256 tính trên `header_utf8 + 0x00 + packed_counts`.

Round-trip phải bit-exact. Source crops dùng deterministic configured JPEG encoder
và lưu encoder/runtime fingerprint.

Heavy evidence dùng content-addressed store trong phạm vi job:

```text
artifact_id = sha256(canonical artifact bytes)
```

- mỗi unique RLE/JPEG blob chỉ được spool một lần;
- evidence item chỉ giữ `artifact_id`, type, byte length và provenance;
- nhiều track/event cùng tham chiếu một blob không nhân bản RAM/disk;
- in-memory/spool quota đếm unique resident/stored bytes, không đếm số reference;
- reference count dùng để xóa blob khi job cleanup; không xóa khi còn event/result
  tham chiếu;
- hash collision hoặc bytes không khớp stored length/hash làm job fail
  `evidence_artifact_integrity_error`.

Store là job-local, không deduplicate xuyên tenant/job và không biến evidence thành
public object. Snapshot cuối vẫn theo R2 layout compatibility ở Mục 28.13.

Bootstrap:

```text
top_k_per_evidence_type = 8
maximum_metadata_observations_per_track = 256
```

`maximum_metadata_observations_per_track` là RAM cap, **không** là logical
evidence cap. Mỗi scalar candidate eligible được append lossless, theo canonical
`(tracklet_id, timestamp, frame_id, observation_id)` vào job-local scalar spool
trước khi rời active RAM. RAM chỉ giữ ring/index ≤256 items/track; Mục 16.4 đọc
toàn bộ scalar candidates từ spool để chọn maximum-cardinality logical bins.

Không downsample, không giữ chỉ first/last, và không bỏ scalar vote candidate vì
top-K/artifact cap. Nếu scalar spool + heavy artifact spool vượt configured unique
byte quota, fail job `geometry_resource_limit_exceeded`; không silently giảm
logical evidence. Heavy RLE/JPEG vẫn chỉ giữ top-K theo evidence type.

## 16.2 Evidence types

```text
CONNECTED_BRIDGE
DISCONNECTED_SAME_FRAME
LEFT_PRESENT
RIGHT_PRESENT
LEFT_GEOMETRY_VALID
RIGHT_GEOMETRY_VALID
LEFT_LOCALIZED_BREAK
RIGHT_LOCALIZED_BREAK
ANGLE_PAIR_VALID
FOV_LEFT_OPPORTUNITY
FOV_RIGHT_OPPORTUNITY
AMBIGUOUS_ASSOCIATION
BOUNDARY_TRUNCATED
MODEL_DROPOUT
```

## 16.3 Deterministic evidence ranking

Không dùng learned/arbitrary scalar quality trong decision hoặc tracking.

Mỗi analyzer trả các numeric hard-gate margins. Với gate:

```text
value >= minimum:
    normalized_margin =
        (value - minimum) / max(abs(minimum), 1e-12)

value <= maximum:
    normalized_margin =
        (maximum - value) / max(abs(maximum), 1e-12)
```

Boolean gate fail thì reject; boolean gate pass không tham gia phép `min`
(tương đương margin `+infinity`). Item:

```text
evidence_gate_margin =
    min(all applicable normalized numeric gate margins)
```

Nếu không có numeric gate áp dụng, diagnostic margin là `0`. Item fail một hard
gate bị reject và không vote. Quy ước này tránh việc một boolean pass có margin
`0` làm mọi evidence score hòa nhau.

Để chọn temporal candidate/top-K/snapshot trong cùng evidence type, sort:

1. `evidence_gate_margin` giảm dần;
2. max source model score giảm dần;
3. `s_anchor_sigma` tăng dần;
4. `abs(s_anchor - s_trigger)` tăng dần;
5. source frame ID tăng dần;
6. observation ID lexicographic.

Ranking chỉ chọn evidence giữa các item đã eligible; nó không thay decision table.

## 16.4 Time bins

Cho:

```text
minimum_spacing_frames = 2
minimum_spacing_sec = 0.05
```

Hai candidates độc lập khi đồng thời:

```text
abs(frame_id_A - frame_id_B) >= minimum_spacing_frames
abs(timestamp_A - timestamp_B) >= minimum_spacing_sec
```

Duplicate/equal/reversed PTS đã bị frame provenance validation xử lý; `dt=0` không
bao giờ tạo hai votes.

Không dùng quality-first greedy vì một middle candidate có thể loại hai endpoint
candidates. Với từng `(physical_event, evidence_type)`:

1. sort candidates theo `(timestamp, frame_id, observation_id)`;
2. giải maximum-cardinality independent selection trên interval-conflict graph
   bằng dynamic programming;
3. objective/tie-break theo thứ tự:

   - nhiều accepted items hơn;
   - tổng `evidence_gate_margin` lớn hơn;
   - tổng max-model-score lớn hơn;
   - tổng anchor sigma nhỏ hơn;
   - tổng trigger distance nhỏ hơn;
   - tuple accepted frame IDs lexicographic nhỏ hơn.

Sau đó accepted items được sort theo timestamp và cấp:

```text
logical_bin_id = 0, 1, 2, ...
```

Quy trình này tránh lỗi fixed-bin boundary, nơi hai frames chỉ cách nhau rất ít
nhưng nằm ở hai bins kề nhau.

Duplicates trong cùng frame không tăng vote.

Các count trong decision logic luôn là count của accepted logical bins, không phải
raw frame count.

## 16.5 Opportunity không đồng nghĩa absence

`FOV_LEFT_OPPORTUNITY` nghĩa là vùng mà cánh trái kỳ vọng nằm trong ảnh và frame có
chất lượng đủ để quan sát.

Nó không tự động tạo:

```text
LEFT_ABSENT
```

Với model hiện tại, failure to emit một bên có thể do model dropout. Vì vậy:

- presence là positive evidence;
- localized break geometry là positive evidence;
- absence đơn thuần là insufficient/negative weak evidence;
- absence không được định vị gãy trong geometry_v2 hiện tại.

---

# 17. Offline fusion giữa các tracklets

## 17.1 Vì sao fusion là bắt buộc ở release đầu

Model hiện tại có thể:

- đổi từ cánh trái sang cánh phải giữa hai frame;
- chỉ detect một bên;
- phát nhiều detections cho cùng paddle;
- mất detection trong một số frame.

Online tracker được thiết kế để giữ identity, nhưng fusion vẫn phải xử lý track bị
split do gap, ambiguity hoặc model switch.

## 17.2 Candidate pair

Hai tracklets `A`, `B` chỉ có thể merge nếu:

- cùng job và cùng ROI/geometry version;
- temporal windows overlap hoặc khoảng cách giữa hai boundary timestamps không
  vượt `maximum_fusion_extrapolation_sec`;
- không từng có observations khác nhau trong cùng frame mà chứng minh hai paddle
  riêng biệt;
- mỗi tracklet có crossing estimate hợp lệ, hoặc sorted union measurements của
  candidate merge tạo được interpolation bracket Mục 15.1;
- velocity directions giống nhau;
- composition bổ sung hoặc không mâu thuẫn;
- merge không phá order với events lân cận.

## 17.3 Crossing-time gate

Cho:

```text
tau_A, sigma_tau_A
tau_B, sigma_tau_B
T_paddle = robust local paddle interval, nếu đã biết
```

Gate:

```text
abs(tau_A - tau_B) <= G_tau
```

Trong đó:

```text
G_tau =
    min(
        maximum_crossing_delta_sec,
        interval_term,                 # bỏ nếu T_paddle unavailable
        max(
            minimum_uncertainty_gate_sec,
            uncertainty_sigma_multiplier
              * sqrt(sigma_tau_A^2 + sigma_tau_B^2)
        )
    )
```

Trong đó:

```text
interval_term =
    expected_paddle_interval_tolerance_ratio * T_paddle
        nếu interval source == "configured"

    maximum_crossing_interval_ratio * T_paddle
        nếu interval source == "auto_estimated"
```

Nếu `T_paddle` chưa đủ tin cậy, bỏ `interval_term`.

Các gate bổ sung:

```text
relative_velocity_delta =
    abs(v_A - v_B)
    / max(abs(v_A), abs(v_B), velocity_epsilon_ratio_per_sec * H)

relative_velocity_delta <= maximum_relative_velocity_delta
trajectory_residual <= maximum_trajectory_residual_ratio * H
```

`v_A/v_B` là posterior velocity tại crossing time. `trajectory_residual` là:

```text
max_t abs(s_hat_A(t) - s_hat_B(t))
```

trên sorted union original timestamps nằm trong overlap hai tracklet windows. Nếu
không có overlap timestamp, chỉ dùng midpoint của hai closest temporal boundaries
khi mỗi trajectory extrapolate tới midpoint không quá
`maximum_fusion_extrapolation_sec`; nếu không, pair ineligible. Không sample theo
wall-clock grid tùy implementation.

## 17.4 Ước lượng paddle interval

`T_paddle` chỉ được học sau khi có ít nhất:

```text
minimum_unambiguous_events_for_interval = 5
```

Unambiguous event để học interval phải:

- có crossing time tốt;
- không phải merge cạnh tranh;
- có ít nhất một observation `CONNECTED_WHOLE` hoặc
  `DISCONNECTED_BOTH`, hoặc track hai phía được association rõ;
- không bị boundary truncation;
- không phải `uncertain` do identity.

Ước lượng:

```text
d_i = tau[i+1] - tau[i]  # tau đã sort tăng theo thời gian
m = median(d_i)
scaled_mad = 1.4826 * MAD(d_i)

keep d_i iff:
    abs(d_i - m) <= max(3 * scaled_mad, timestamp_epsilon_sec)

T_paddle = median(kept d_i)
```

Nếu `MAD=0`, chỉ các delta bằng median trong `timestamp_epsilon_sec` được giữ.
Không dùng mean đơn thuần.

Để tránh circularity, fusion chạy hai pass:

1. base pass dùng absolute crossing/uncertainty/velocity/trajectory gates, bỏ term
   `T_paddle`;
2. học interval chỉ từ provisional events unambiguous của base pass;
3. nếu đủ năm events, optional refinement pass dùng interval gate;
4. refinement không được đảo hoặc gộp hai base events đã có competing identities.

Nếu khoảng paddle vật lý đã biết, config MAY cung cấp:

```json
{
  "expected_paddle_interval_sec": 0.42,
  "expected_paddle_interval_tolerance_ratio": 0.20
}
```

Giá trị cấu hình có ưu tiên hơn auto-estimate và phải được ghi nguồn
`"configured"`.

## 17.5 Constrained multi-tracklet fusion

Một physical event có thể chứa hơn hai tracklets nếu tracker split nhiều lần.
Invariant one-to-one áp ở membership:

```text
one observation/tracklet belongs to at most one finalized event hypothesis
```

không phải “mỗi event chỉ là một pair”. Cho tới khi cardinality được certify,
cluster chỉ là event hypothesis, chưa được gọi là physical paddle đã biết chắc.

Khởi tạo mỗi tracklet là một cluster. Trong crossing-time window, lặp:

1. tính candidate merge cost giữa các clusters kề order;
2. chỉ cho merge nếu mọi cross-cluster tracklet pair qua hard gates
   (complete-link compatibility);
3. chọn lowest-cost unique merge;
4. merge clusters;
5. recompute aggregate crossing/velocity/trajectory và candidate costs.

Không dùng single-link transitivity: `A` compatible `B` và `B` compatible `C`
không đủ merge `A+B+C` nếu `A` không compatible `C`.

Một cluster merge chỉ definitive khi:

- best candidate duy nhất cho cả hai clusters;
- không có competitor mà
  `(competitor_int - best_int) * cost_quantization < fusion_ambiguity_margin`;
- merge không làm hai physical events đảo thứ tự;
- mọi member tracklet chỉ thuộc cluster đó.

Bootstrap:

```text
fusion_ambiguity_margin = 0.10
```

Nếu cạnh tranh:

- không merge mù;
- tạo stable `identity_conflict_group_id`;
- giữ candidate-compatibility graph và bounded ordered-DP backpointers, không
  enumerate mọi feasible partition;
- DP tính **exact** `possible_event_count_min/max` trong resource bound;
- chỉ materialize tối đa `maximum_identity_hypotheses_per_group` k-best
  hypotheses cho diagnostics; nếu bị cap, set `hypotheses_truncated=true` nhưng
  không làm approximate min/max;
- các hypothesis liên quan gắn `identity_ambiguous`;
- final status tối đa là `uncertain`;
- lưu toàn bộ candidate IDs và costs.

Nếu `possible_event_count_min != possible_event_count_max`, event cardinality
không xác định:

```text
count_certified = false
job returns failure after preserving diagnostics
report export is blocked
reason = "event_cardinality_unresolved"
```

Không cấp `paddle_id`, không đưa các competing tracklets thành hai “physical
events” rồi tuyên bố `total_bars` exact. UI phải hiển thị conflict group cho human
review/reprocessing. Nếu min=max nhưng membership còn ambiguous, số hypothesis đó
chỉ được cấp IDs theo canonical rank-1 partition bên dưới, canonical status
`uncertain`, và `count_certified=true`; alternative partitions không tạo thêm
result/track ownership. Identity taint vẫn không được xóa.

Nếu conflict group vượt `maximum_identity_conflict_tracklets`, hoặc exact ordered
DP không thể hoàn thành trong explicit resource limit, fail job
`geometry_resource_limit_exceeded`; không silently truncate graph rồi certify
count.

Canonical conflict-group solver dùng thứ tự tracklet
`a_1..a_n = (crossing_time_sec, primary_track_id)` tăng dần. Một feasible event
block `B[j,i]` là contiguous block `a_j..a_i` khi mọi cross-tracklet pair trong
block qua complete-link hard gates Mục 17.2–17.3 và collective trigger/crossing
contract Mục 15. Không skip tracklet giữa block; đây là order-preserving partition.

Đặt singleton `B[i,i]=true`. Exact count DP:

```text
min_count[0] = max_count[0] = 0

min_count[i] = min(
    min_count[j-1] + 1
    for j in 1..i if B[j,i]
)

max_count[i] = max(
    max_count[j-1] + 1
    for j in 1..i if B[j,i]
)
```

Vì singleton luôn feasible, recurrence không có empty state. Kết quả cuối:

```text
possible_event_count_min = min_count[n]
possible_event_count_max = max_count[n]
```

Khi min=max, membership alternatives được rank deterministic. Block cost là `0`
cho singleton, ngược lại `max(C_fusion(a,b))` cho mọi pair trong block. Partition
rank theo:

1. tổng integer block-cost nhỏ hơn;
2. nhiều complementary LEFT/RIGHT histories hơn;
3. full tuple ordered blocks `(first_track_id,...,last_track_id)` lexicographic.

DP giữ parent pointers cho rank-1 và tối đa
`maximum_identity_hypotheses_per_group` alternatives. Rank-1 partition là **duy
nhất** partition cấp per-paddle result/track membership; các alternative chỉ là
diagnostic và mọi result của group giữ `identity_ambiguous=true`, final
`uncertain`. Nếu min≠max, không chọn rank-1 để phát result mà fail theo contract
ở trên. K-best materialization bị cap không được ảnh hưởng min/max hay rank-1.

Tie-break cluster:

1. quantized cost;
2. nhiều complementary observations hơn;
3. sorted tuple member track IDs lexicographic nhỏ hơn.

## 17.6 Fusion cost

```text
C_fusion =
    0.45 * n_tau
  + 0.25 * n_v
  + 0.20 * n_traj
  + 0.10 * composition_penalty
```

Canonical normalized terms:

```text
n_tau = clamp(abs(tau_A - tau_B) / G_tau, 0, 1)

n_v = clamp(
    relative_velocity_delta / maximum_relative_velocity_delta,
    0, 1
)

n_traj = clamp(
    trajectory_residual / (maximum_trajectory_residual_ratio * H),
    0, 1
)

composition_penalty:
    complementary LEFT_ONLY/RIGHT_ONLY histories = 0.0
    same-side split histories                    = 0.5
    other eligible composition                   = 1.0
```

Với two clusters, canonical cluster cost là:

```text
max(C_fusion(a, b) for every a in cluster_A, b in cluster_B)
```

Tức complete-link cost. Sau merge, cluster crossing time được recompute ưu tiên
exact bracket trên union measurements. Nếu chỉ có member estimates:

```text
tau_cluster = median(member_tau)
sigma_cluster = max(
    median(member_sigma_tau),
    1.4826 * MAD(member_tau),
    timestamp_epsilon_sec
)
```

Cluster crossing/velocity chỉ là diagnostics/order keys; hard gate và merge order
dùng all-pair metrics ở trên.

Ưu tiên merge complementary:

```text
LEFT_ONLY track + RIGHT_ONLY track
```

nhưng composition không được vượt hard gates.

## 17.7 Final paddle ID

Sau fusion:

1. finalized event hypotheses được sort theo crossing time tăng dần;
2. tie-break primary track ID;
3. cấp:

```text
paddle_id = 1, 2, 3, ...
```

Primary `track_id`:

1. track có nhiều independent evidence bins nhất;
2. nếu hòa, track duration dài hơn;
3. nếu hòa, track ID nhỏ hơn.

Tất cả IDs giữ trong:

```json
{
  "paddle_id": 12,
  "track_id": 17,
  "track_ids": [17, 23]
}
```

## 17.8 Không merge chỉ vì “gần”

Không được merge nếu chỉ có một trong các dấu hiệu:

- bbox gần;
- centroid gần;
- màu giống;
- cùng detection score;
- cùng snapshot;
- xảy ra trong trigger strip.

Crossing-time, trajectory, order và uniqueness là bắt buộc.

---

# 18. Hợp nhất evidence theo physical event

## 18.1 Sau fusion

Evidence của mọi alias tracklets được gom vào physical event rồi dedup lại theo:

```text
(source_frame_id, evidence_type, canonical component group)
```

Sau đó mới áp time bins.

Một source frame không được nhân đôi vote chỉ vì hai tracklets đã merge.

## 18.2 Event evidence summary

Mỗi event tạo:

```json
{
  "independent_bins": {
    "connected_bridge": 0,
    "disconnected_same_frame": 2,
    "left_present": 4,
    "right_present": 3,
    "left_geometry_valid": 3,
    "right_geometry_valid": 2,
    "angle_pair_valid": 0
  },
  "opportunity_bins": {
    "left": 5,
    "right": 5
  },
  "conflict_bins": 0,
  "ambiguous_bins": 1
}
```

## 18.3 Không dùng hard presence ratio

Không được yêu cầu:

```text
left_presence_ratio >= 0.6
right_presence_ratio >= 0.6
```

để công nhận hai bên, vì model có thể luân phiên trái/phải và mỗi bên chỉ đạt gần
`0.5`.

Canonical bootstrap dùng independent positive evidence:

```text
minimum_left_presence_bins = 2
minimum_right_presence_bins = 2
```

và association/fusion phải unique.

Presence ratio MAY được report làm diagnostic, không phải hard gate cho temporal
two-side evidence.

## 18.4 Event observability grade

```text
GRADE_A:
    >= 2 strong same-frame observations supporting the same topology

GRADE_B:
    complementary left/right temporal evidence, unique association,
    sufficient opportunities, no strong conflict

GRADE_C:
    only one side or insufficient independent bins

GRADE_D:
    identity/FOV/model/geometry ambiguity
```

Grade là diagnostic/triage, không phải decision gate thứ hai thay Mục 21/23.
Mục 21/23 là authority duy nhất cho definitive label. Bootstrap vẫn bắt buộc:

```text
GRADE_D -> uncertain
single-side GRADE_C -> uncertain theo invariant Mục 21.7
```

GRADE_A/B/C tự chúng không cấp một final status; chúng chỉ giải thích mức quan sát
trong audit/UI.

---

# 19. Center topology analyzer

## 19.1 Trạng thái đầu ra

Per-frame bridge test:

```text
PRESENT
ABSENT
UNKNOWN
```

Per-event center state:

```text
INTACT
BROKEN_TOPOLOGICAL
BROKEN_TEMPORAL
UNKNOWN
CONFLICT
```

Không dùng boolean duy nhất vì `UNKNOWN` và `ABSENT` có ý nghĩa hoàn toàn khác.

## 19.2 Center corridor

Với event anchor `s0`, đặt:

```text
B = chain_band_half_width
T_raw = median(final same-frame thickness của các side fit hợp lệ)
```

Thickness mỗi side dùng công thức Mục 20.3. Plausible range:

```text
minimum_plausible_side_thickness_px =
    max(2, 0.002 * roi_width)

maximum_plausible_side_thickness_px =
    0.08 * roi_width
```

Nếu không có side fit hợp lệ hoặc `T_raw` ngoài range, bridge test `UNKNOWN`;
không lấy thickness của arbitrary component/background. Nếu hợp lệ:

```text
T = clamp(
    T_raw,
    minimum_plausible_side_thickness_px,
    maximum_plausible_side_thickness_px
)
```

Bootstrap corridor:

```text
s in [s0 - max(0.75*T, 0.012*H),
      s0 + max(0.75*T, 0.012*H)]

q in [-Q_inner, +Q_inner]

Q_inner =
    min(
        3 * B,
        0.20 * minimum(
            available_left_extent(s0),
            available_right_extent(s0)
        )
)
```

Nếu `Q_inner <= B`, bridge test là `UNKNOWN`.

## 19.3 Inner gates

```text
left_inner_gate:
    q in [-Q_inner, -B]

chain_gate:
    q in [-B, +B]

right_inner_gate:
    q in [+B, +Q_inner]
```

Một bridge candidate phải thuộc cùng một connected component của
mask đã clip vào center corridor:

```text
bridge_test_mask = topology_mask AND center_corridor_mask
```

Chạy lại 8-connected components trên `bridge_test_mask`. Cùng một clipped
component phải giao cả ba gates.

`topology_mask` ở phép test này thuộc **một canonical source
SegmentationInstance/component**. Không OR masks từ independent instances để tạo
bridge nhân tạo. Duplicate containment variants phải qua agreement rule Mục 12.3.

Không đủ nếu pixels thuộc cùng global component nhưng đường nối vòng ra ngoài
corridor rồi quay lại.

Việc component chỉ chạm một vài pixel ở gate không đủ.

## 19.4 Q-coverage

Chia `[-Q_inner, +Q_inner]` thành:

```text
center_bridge_q_bins = 20
```

Một q-bin được covered nếu có foreground của cùng bridge component trong center
corridor.

```text
q_coverage =
    covered_q_bins / valid_q_bins
```

Bootstrap:

```text
minimum_center_q_coverage = 0.90
```

Các q-bin bị cắt bởi ROI không được tính là valid, nhưng nếu một trong hai inner
gates bị cắt thì kết quả là `UNKNOWN`, không phải `PRESENT`.

## 19.5 Bridge cross-section

Với mỗi covered q-bin, rasterize foreground pixel centers lên unit `s` bins có
origin cố định `s_min`. Tách các runs occupied liên tiếp và lấy length của
**longest contiguous run**; không dùng `max(s)-min(s)` vì khoảng trống ở giữa
không phải foreground. Dùng type-7 P10 của các run lengths:

```text
bridge_cross_section =
    P10(foreground_s_span_per_q_bin)
```

Yêu cầu:

```text
bridge_cross_section >= 0.20 * T
```

Mục tiêu là loại một đường nối chỉ dày 1–2 pixel do segmentation noise.

## 19.6 PRESENT

Per-frame bridge là `PRESENT` khi toàn bộ:

- observation có left và right geometry;
- cùng một clipped bridge-test component giao cả ba inner gates;
- `q_coverage >= 0.90`;
- bridge cross-section đạt;
- không có morphology nối component;
- component không bị cắt ở vùng cần kiểm tra;
- bridge đến từ một canonical source component, không từ union instances;
- association không ambiguous;
- source frame là original.

## 19.7 ABSENT

Per-frame bridge là `ABSENT` chỉ khi toàn bộ:

- có left và right components được pair duy nhất;
- cả hai anchor hợp lệ;
- left/right FOV opportunities hợp lệ;
- center corridor nằm trọn trong ROI;
- không có bridge component qua ba gates;
- việc thiếu bridge không do predicted bbox crop làm mất vùng mà model chưa bao
  phủ;
- observation không ambiguous.

Điều kiện crop coverage được kiểm trên raster, không suy từ việc bbox “chạm chain”:

```text
required_cells =
    mọi ROI pixel-center cell thuộc center corridor và ba inner gates

crop_coverage_valid =
    every required_cell nằm trong union exact half-open
    model_bbox_crop_roi_xyxy của các source instances thuộc observation
```

Hoặc adapter tương lai cung cấp uncropped probability mask đã validation cho toàn
`required_cells`. Union crop rectangles chỉ chứng minh vùng đã được model
reconstruct/evaluate; nó không được OR foreground masks để tạo PRESENT.

Nếu không chứng minh được điều này, trả `UNKNOWN`. Không được gọi `ABSENT` chỉ từ
hai mask đã bị crop xa nhau.

## 19.8 Strong evidence flags

```text
strong_connected =
    connected_bridge_bins >= 2

strong_same_frame_disconnected =
    disconnected_same_frame_bins >= 2
```

Một bin `DISCONNECTED_SAME_FRAME` chỉ được tạo từ per-frame `ABSENT`.

## 19.9 Temporal broken-center evidence

Temporal evidence nhằm xử lý emission:

```text
LEFT_ONLY -> RIGHT_ONLY -> LEFT_ONLY
```

Strong temporal evidence yêu cầu toàn bộ:

```text
connected_bridge_bins == 0
left_present_bins >= 2
right_present_bins >= 2
joint_two_side_opportunity_bins >= 4
association_or_fusion_is_unique == true
identity_ambiguous == false
fov_sufficient == true
current_event_not_boundary_truncated == true
```

Ngoài ra:

- left/right crossing-time residual nằm trong fusion gates;
- không có competing paddle trong cùng temporal window;
- không có same-frame observation chứng minh hai fragment thuộc hai paddles khác
  nhau;
- capability/policy cho phép temporal classification.

Opportunity phải lưu riêng:

```text
left_opportunity_bins
right_opportunity_bins
joint_two_side_opportunity_bins
```

`joint_two_side_opportunity_bins` chỉ đếm original-frame bins mà expected left
extent, expected right extent và center corridor đều nằm trong physical ROI/FOV.
Nó không phụ thuộc detection/model bbox hoặc side có được model emit hay không và
không phải tổng raw detections.

Không có generic phrase “frame/mask quality đủ” trong canonical hard gate. Nếu
blur/exposure quality được thêm, phải định nghĩa metric, region, threshold,
calibration và capability version riêng; implementation không được tự chọn.

## 19.10 Capability gate cho temporal classification

Full-system capability record phải khai báo:

```json
{
  "validation": {
    "temporal_complementary_emission": "provisional"
  },
  "production_enabled": {
    "temporal_center_break": false
  }
}
```

Policy:

```text
validated:
    strong temporal evidence MAY yield BROKEN_TEMPORAL

provisional:
    shadow/evaluation MAY calculate BROKEN_TEMPORAL,
    production final status MUST remain uncertain

unsupported:
    temporal evidence is diagnostic only
```

Behavior “model gãy giữa có thể luân phiên cánh trái/phải” là input domain do chủ
dự án cung cấp. Source code và các ảnh crop hiện có chứng minh pipeline có thể phát
một hoặc nhiều components, nhưng chưa có raw labeled event set đủ để đo false
positive của temporal rule. Vì vậy bootstrap an toàn là `provisional` cho tới khi
đạt acceptance ở Mục 32.11.

Sau khi validation pass:

- tạo immutable capability record mới với validation path `"validated"` và
  production flag tương ứng `true`;
- cập nhật deployment binding tới record mới; algorithm config hash chỉ đổi nếu
  threshold/rule config đổi;
- bump rule version tối thiểu patch vì production decision behavior đổi;
- chạy shadow/approval/rollback checks;
- không cần đổi downstream result schema.

## 19.11 Event center-state truth table

Mọi shorthand:

```text
enabled(feature) =
    exact system signature matches
    AND exact runtime fingerprint matches
    AND job nằm trong validated operating domain
    AND corresponding validation status == "validated"
    AND corresponding production_enabled flag == true
```

Một production flag `true` khi validation chưa `validated` là invalid capability
record và geometry-v2 unavailable; không coi boolean đơn lẻ là authority.

Áp theo thứ tự:

```text
if strong_connected
   and (strong_same_frame_disconnected or strong_temporal):
    center_state = CONFLICT

elif connected_bridge_bins == 1
     and (strong_same_frame_disconnected or strong_temporal):
    center_state = CONFLICT

elif strong_connected
     and enabled(same_frame_center_topology):
    center_state = INTACT

elif strong_same_frame_disconnected
     and enabled(same_frame_center_topology):
    center_state = BROKEN_TOPOLOGICAL

elif strong_temporal
     and enabled(temporal_center_break):
    center_state = BROKEN_TEMPORAL

else:
    center_state = UNKNOWN
```

Khi strong candidate có nhưng corresponding production flag còn false:

- giữ `shadow_center_candidate` (`INTACT`, `BROKEN_TOPOLOGICAL` hoặc
  `BROKEN_TEMPORAL`) và evidence trong diagnostics;
- `center_state = UNKNOWN` cho canonical production decision;
- `suspected_breakage = true` chỉ khi shadow candidate là
  `BROKEN_TOPOLOGICAL` hoặc `BROKEN_TEMPORAL`; shadow `INTACT` không tự tạo
  safety alert;
- reason `model_capability_not_validated`.

Một connected frame duy nhất không “thắng tuyệt đối”. Nó có thể là false bridge,
motion artifact hoặc segmentation merge; nếu đối đầu bằng chứng gãy mạnh thì kết
quả là `CONFLICT`.

Ngược lại, một disconnected frame duy nhất không đủ kết luận gãy.

---

# 20. Side geometry và side integrity

## 20.1 Trạng thái mỗi phía

```text
VALID
BROKEN_LOCALIZED
UNKNOWN
CONFLICT
```

Không có `ABSENT_IS_BROKEN` trong model profile hiện tại.

## 20.2 Tập pixel mỗi phía

Với anchor `s0`:

```text
left pixels:
    q < -(B + side_exclusion_margin)

right pixels:
    q > +(B + side_exclusion_margin)
```

Bootstrap:

```text
side_exclusion_margin_px = max(2, 0.005 * roi_width)
```

Chỉ dùng accepted `geometry_mask` components thuộc physical event.

Pixels của:

- chain band;
- duplicate aliases;
- component quá nhỏ;
- ambiguous merged instance;
- ROI overlay/text;

không được dùng.

## 20.3 Robust line fit

Fit từng phía độc lập bằng canonical deterministic iterative MAD + total least
squares:

1. initial TLS trên toàn bộ accepted side pixels;
2. tính initial projected span `L0 = P95 - P5`;
3. estimate `T0 = area_px / max(L0, 1)`;
4. tính orthogonal residuals;
5. mỗi iteration:

   ```text
   robust_sigma = max(1.0, 1.4826 * MAD(residuals))
   statistical_cut = median(residuals) + 3 * robust_sigma
   physical_cut = side_outlier_residual_thickness_ratio * T0
   cut = max(1.0, min(statistical_cut, physical_cut))
   inliers = residual <= cut
   ```

6. refit TLS trên inliers;
7. dừng khi inlier IDs không đổi hoặc sau năm iterations;
8. hướng vector từ inner ra outer;
9. reject nếu fit suy biến/thiếu pixels.

RANSAC không thuộc bootstrap; fit luôn là deterministic iterative MAD/TLS và
không phụ thuộc random sampling.

Bootstrap:

```text
minimum_side_pixels = 80
minimum_side_projected_span_px = max(20, 0.04 * roi_width)
side_outlier_residual_thickness_ratio = 0.35
maximum_side_fit_iterations = 5
minimum_linearity_ratio = 0.90
```

Linearity:

```text
linearity_ratio = lambda_major / (lambda_major + lambda_minor)
```

trong đó `lambda` là eigenvalues covariance của inlier pixels.

Residual:

```text
P_reference = mean(final inlier pixel-center coordinates)
residual(P) = abs(dot(P - P_reference, unit_normal_to_fitted_axis))
median_orthogonal_residual = median(residual(P))
residual_mad

final_projected_span = P95(r(inliers)) - P5(r(inliers))
estimated_side_thickness =
    final_inlier_count / max(final_projected_span, 1)
```

Fit hợp lệ khi:

```text
linearity_ratio >= 0.90
median_orthogonal_residual <= 0.20 * estimated_side_thickness
```

Nếu không đạt, frame-side state là `UNKNOWN`, không tự coi là bent hay broken.

## 20.4 Outward coordinate và endpoints

Với fitted unit vector `u` hướng từ inner ra outer:

```text
r(P) = dot(P - P_reference, u)
```

Robust projected range:

```text
r_inner = P5(r(inlier pixels))
r_outer = P95(r(inlier pixels))
projected_length = r_outer - r_inner
```

Endpoint outer:

1. lấy 5% inlier pixels có `r` lớn nhất;
2. lấy median coordinate của nhóm;
3. project median đó lên fitted line.

Endpoint inner làm đối xứng với 5% `r` nhỏ nhất.

Không dùng một pixel cực trị duy nhất.

## 20.5 Coverage profile

Có hai profile khác nhau, không dùng lẫn.

**Expected profile** chia extent từ inner gate tới expected outer endpoint thành:

```text
side_coverage_bins = 20
```

Chỉ bins nằm trong physical FOV là `valid`. Một valid bin `covered` khi:

```text
inlier_pixel_count >= minimum_coverage_pixels_per_bin
and
bin_thickness >= minimum_coverage_thickness_ratio
                 * estimated_side_thickness
```

Với final side-fit `P_reference` và unit normal `n_axis`, `bin_thickness` là:

```text
n(P) = dot(P - P_reference, n_axis)
bin_thickness = P95(n(P) for final inlier pixels thuộc bin)
                - P5(n(P) for final inlier pixels thuộc bin)
```

`P5/P95` dùng Mục 0.1. Nếu bin không đủ `minimum_coverage_pixels_per_bin`, nó
uncovered và không tính thickness. Cùng công thức/threshold dùng cho expected và
intrinsic profile; không được dùng bbox width, contour width hoặc a heuristic khác.

Bootstrap:

```text
minimum_coverage_pixels_per_bin = 5
minimum_coverage_thickness_ratio = 0.25
```

Bins dùng half-open convention Mục 0.1. Metrics expected:

```text
coverage_ratio = covered_valid_bins / valid_bins

outer_length_ratio =
    observed_projected_length / expected_projected_length
```

Nếu không có expected extent hoặc không đủ valid bins tới expected endpoint, hai
metrics expected là unavailable, không phải zero.

**Intrinsic profile** dùng cùng 20 bins nhưng trải trên observed robust
`[r_inner, r_outer]`; nó không cần expected extent và chỉ dùng phát hiện internal
gap. `intrinsic_coverage` là covered bins / all intrinsic bins.

```text
largest_internal_gap_ratio =
    longest consecutive uncovered-bin run
    strictly between first and last covered intrinsic bin
    / intrinsic_valid_bin_count
```

Nếu first/last covered không tồn tại, crop/ROI truncation chạm vùng profile, hoặc
fewer than `minimum_intrinsic_profile_bins=10` valid bins, internal profile là
unobservable và không tạo positive gap evidence.

## 20.6 Expected side extent

Nguồn ưu tiên:

1. giá trị cơ khí/camera calibration cấu hình;
2. perspective reference curve được calibration;
3. robust online reference từ các event `CONNECTED_WHOLE` rõ ràng;
4. unavailable.

Config có thể cung cấp:

```json
{
  "expected_left_extent_px_at_trigger": 212.0,
  "expected_right_extent_px_at_trigger": 208.0,
  "expected_extent_tolerance_ratio": 0.12
}
```

Resolved:

```text
valid_minimum_length_ratio = 1 - expected_extent_tolerance_ratio
```

Không cấu hình đồng thời hai giá trị độc lập có thể mâu thuẫn.

Online leave-one-out reference cho một target chỉ được tạo khi có ít nhất:

```text
minimum_reference_other_paddles = 5
```

**events khác** thỏa:

- center `INTACT`;
- cả hai side geometry tốt;
- không boundary truncation;
- không positive break evidence;
- anchor gần vị trí so sánh.

Reference:

```text
expected_extent = P90(valid observed extents)
```

P90 giảm ảnh hưởng của paddle bị ngắn/gãy lọt vào calibration.

Khi so sánh hai observations/reference không có perspective curve:

```text
abs(s_anchor_A - s_anchor_B) <= 0.03 * H
```

Nếu không, extent comparison là `UNKNOWN`.

Không được tự bootstrap reference từ chính event đang cần phân loại.

Reference bootstrap chạy ở offline phase theo hai pass:

1. chọn candidate reference chỉ bằng center connectivity, line-fit quality,
   `intrinsic_coverage` và FOV; chưa dùng expected profile/outer-length verdict;
2. tạo leave-one-out P90 reference từ ít nhất năm candidate events khác ở anchor
   tương đương;
3. chạy side-integrity classification cho toàn bộ events bằng reference đã khóa.

Một event không được làm reference cho chính nó. Vì vậy cần tối thiểu sáu
candidates tổng cộng để mọi candidate có năm references khác.

Nếu không có giá trị cơ khí/calibration và không thể bảo đảm population có đủ
intact paddles, population reference chỉ là `provisional`; shortness không được
trở thành definitive `BROKEN_LOCALIZED` cho tới khi reference source được
validation. Internal-gap evidence độc lập vẫn có thể dùng.

`expected_*_extent_px_at_trigger` chỉ hợp lệ khi:

```text
abs(s_anchor - s_trigger) <= reference_anchor_gate_ratio * H
```

Ngoài khoảng đó phải có validated perspective reference curve tại `s_anchor`;
nếu không expected extent là unavailable. Config không được tự nâng một population
reference từ provisional lên validated; authority nằm trong exact capability
record/deployment profile.

## 20.7 FOV opportunity cho side extent

Negative shortness evidence hợp lệ chỉ khi:

```text
available_side_extent(s0)
    >= expected_projected_length *
       (1 + minimum_fov_margin_ratio)
```

Bootstrap:

```text
minimum_fov_margin_ratio = 0.05
```

Nếu expected endpoint nằm ngoài hoặc sát ROI, side state là `UNKNOWN`.

Predicted detection bbox boundary không được coi là physical FOV boundary; việc
mask chạm bbox crop làm giảm evidence quality và có thể biến absence thành
`UNKNOWN`.

## 20.8 Frame-level side decision

Frame-side `VALID` khi:

```text
fit_valid
expected extent available and validated
FOV valid through expected endpoint
coverage_ratio >= 0.85
largest_internal_gap_ratio <= 0.08
outer_length_ratio >= valid_minimum_length_ratio
not crop_or_roi_truncated
```

Frame-side `BROKEN_LOCALIZED` khi có positive localized evidence:

```text
fit_valid
(
    (
        internal_profile_observable
        and not crop_or_roi_truncated
        and largest_internal_gap_ratio >= 0.15
    )
    OR
    (
        FOV valid through expected endpoint
        and validated reference available
        and outer_length_ratio <= 0.72
    )
)
```

Khoảng giữa:

```text
0.08 < largest_internal_gap_ratio < 0.15
0.72 < outer_length_ratio < 0.88
```

là gray zone và trả `UNKNOWN`, không nội suy thành broken/valid.

Nếu expected profile nói valid nhưng intrinsic-gap path nói broken, hoặc ngược
lại, hay reference instability làm hai paths khác verdict:

```text
CONFLICT
```

## 20.9 Event-level side decision

Mỗi phía yêu cầu:

```text
minimum_side_evidence_bins = 2
minimum_side_support_ratio = 0.60
```

Cho:

```text
valid_count
broken_count
eligible_count = valid_count + broken_count
```

Rules:

```text
side_candidate_state = null

if broken_count >= 2
   and broken_count / eligible_count >= 0.60
   and valid_count < 2:
    side_candidate_state = BROKEN_LOCALIZED

elif valid_count >= 2
     and valid_count / eligible_count >= 0.60
     and broken_count < 2:
    side_candidate_state = VALID

elif valid_count >= 2 and broken_count >= 2:
    side_candidate_state = CONFLICT

else:
    side_candidate_state = UNKNOWN
```

Raw absence frames không nằm trong `eligible_count`.

Resolve:

```text
if side_candidate_state == CONFLICT:
    side_state = CONFLICT

elif side_candidate_state == VALID
     and enabled(side_geometry_validity):
    side_state = VALID

elif side_candidate_state == BROKEN_LOCALIZED
     and enabled(localized_side_break):
    side_state = BROKEN_LOCALIZED

elif side_candidate_state in {VALID, BROKEN_LOCALIZED}:
    side_state = UNKNOWN
    shadow_side_candidate = side_candidate_state
    reason = "model_capability_not_validated"

else:
    side_state = UNKNOWN
```

`VALID` cũng là một production-relevant claim vì nó mở đường cho
`normal/bent/broken_center`; do đó không được bypass capability gate.

## 20.10 Full-side missing

Với current capability:

```text
absence_as_negative_evidence_production_enabled = false
```

Do đó một bên hoàn toàn không emit trong toàn event:

- không được tự kết luận bên đó gãy;
- không được tự kết luận phía đối diện gãy;
- event observability là `GRADE_C` hoặc `GRADE_D`;
- final status thông thường là `uncertain`.

Model tương lai MAY record absence đáng tin như diagnostic sau validation, nhưng
không có “full-side-missing rule” canonical nào chỉ từ absence. Muốn emit location
definitive phải thêm observable/model contract và decision rule mới theo Mục 4.3,
phiên bản schema/capability riêng, và held-out acceptance; không dùng flag absence
để lách invariant single-side.

---

# 21. Breakage analyzer

## 21.1 Mục tiêu

Breakage analyzer kết hợp:

- center topology;
- left integrity;
- right integrity;
- identity/FOV/model quality;
- capability manifest.

Nó không đo angle.

## 21.2 Positive và negative evidence

Positive break evidence:

- `BROKEN_TOPOLOGICAL`;
- `BROKEN_TEMPORAL` khi policy validated;
- `BROKEN_LOCALIZED` ở một side.

Không phải positive break evidence:

- không có detection;
- một side không emit;
- low model confidence;
- bbox ngắn;
- một disconnected frame đơn lẻ;
- một observation ambiguous;
- side chạm ROI.

## 21.3 Breakage result

Internal result:

```json
{
  "center_state": "BROKEN_TOPOLOGICAL",
  "left_state": "VALID",
  "right_state": "VALID",
  "definitive_status": "broken_center",
  "suspected_breakage": true,
  "possible_breakage_statuses": ["broken_center"],
  "reason_codes": ["center_disconnected_same_frame_multi_bin"]
}
```

Nếu không định vị được:

```json
{
  "center_state": "UNKNOWN",
  "left_state": "UNKNOWN",
  "right_state": "VALID",
  "definitive_status": null,
  "suspected_breakage": true,
  "possible_breakage_statuses": [
    "broken_left",
    "broken_center"
  ],
  "reason_codes": [
    "single_side_only_location_unidentifiable"
  ]
}
```

`possible_breakage_statuses` là danh sách non-exhaustive các damage hypotheses có
bằng chứng/hợp lý; nó không tuyên bố loại trừ `normal`, `bent_*`, dropout hoặc FOV
causes.

`suspected_breakage` là cờ safety-triage: breakage chưa thể bị loại trừ hoặc có
positive/temporal suspicion cần review. Nó **không** đồng nghĩa với positive
localized/topological evidence và không tự cho phép final `broken_*`.

## 21.4 Center broken

Definitive `broken_center` khi:

```text
center_state in {BROKEN_TOPOLOGICAL, BROKEN_TEMPORAL}
left_state == VALID
right_state == VALID
identity_ambiguous == false
```

Nếu center broken nhưng một side `UNKNOWN`, `BROKEN_LOCALIZED` hoặc `CONFLICT`:

```text
status = uncertain
suspected_breakage = true
reason = "center_break_with_side_state_unresolved"
```

Lý do: taxonomy không có `broken_both` hoặc combined damage label.

## 21.5 Center intact

Nếu `center_state == INTACT`:

```text
left BROKEN_LOCALIZED + right VALID
    -> broken_left

left VALID + right BROKEN_LOCALIZED
    -> broken_right

left BROKEN_LOCALIZED + right BROKEN_LOCALIZED
    -> uncertain
       reason = "both_sides_broken_no_canonical_label"

any side CONFLICT
    -> uncertain

any side UNKNOWN
    -> uncertain

left VALID + right VALID
    -> eligible for angle analysis
```

## 21.6 Center unknown

Khi `center_state == UNKNOWN`, chỉ cho phép definitive side break nếu toàn bộ:

- một side có `BROKEN_LOCALIZED` bằng positive geometry;
- side đối diện `VALID`;
- `left_reliably_present` và `right_reliably_present` đều true;
- không phải single-side-only event;
- không có temporal center-break evidence mạnh nhưng chưa validated;
- không có identity/FOV conflict;
- localized break nằm ngoài center corridor.

Khi đó:

```text
left localized  -> broken_left
right localized -> broken_right
```

Mọi trường hợp khác:

```text
uncertain
```

## 21.7 Single-side-only invariant

Nếu sau fusion:

```text
left_reliably_present =
    left_present_bins >= minimum_left_presence_bins

right_reliably_present =
    right_present_bins >= minimum_right_presence_bins

is_single_side_only =
    (left_reliably_present and not right_reliably_present)
    OR
    (right_reliably_present and not left_reliably_present)
```

Khi `is_single_side_only`:

```text
final status = uncertain
reason = "single_side_only_location_unidentifiable"
angle fields = null
```

Invariant này có ưu tiên hơn heuristic side-length.

---

# 22. Angle analyzer

## 22.1 Điều kiện được phép đo

Angle chỉ được đo khi event:

```text
center_state == INTACT
left_state == VALID
right_state == VALID
identity_ambiguous == false
```

Analyzer MAY compute shadow angle metrics khi
`enabled(angle_classification)=false`, nhưng canonical final status phải
`uncertain` với `model_capability_not_validated`. Definitive `normal/bent_*` chỉ
được trả khi flag true.

Mỗi angle sample phải đến từ:

- cùng một original source frame;
- cùng một `CONNECTED_WHOLE` observation;
- cả hai side fit hợp lệ trong frame đó;
- không ghép angle trái ở frame này với angle phải ở frame khác;
- không synthetic/interpolated frame;
- không boundary truncation;
- không morphology làm thay đổi topology.

Nếu event broken hoặc break location uncertain:

```text
all angle outputs = null
```

## 22.2 Unit vectors

Right vector:

```text
u_R = fitted right-side unit vector
if dot(u_R, h) < 0:
    u_R = -u_R
```

Left vector:

```text
u_L = fitted left-side unit vector
if dot(u_L, -h) < 0:
    u_L = -u_L
```

Cả hai đều hướng từ inner ra outer.

Reject angle sample nếu:

```text
abs(dot(u_R, h)) < axis_orientation_epsilon
or
abs(dot(u_L, -h)) < axis_orientation_epsilon
```

vì side axis gần song song chain làm phép “outer” không ổn định.

## 22.3 Per-side angles

```text
theta_right_deg =
    degrees(atan2(dot(u_R, d), dot(u_R, h)))

theta_left_deg =
    degrees(atan2(dot(u_L, d), dot(u_L, -h)))
```

Sign convention:

```text
positive:
    outer endpoint thấp hơn inner endpoint theo +d

negative:
    outer endpoint cao hơn inner endpoint theo +d

near zero:
    side gần vuông góc với chain
```

## 22.4 Global tilt

Cho robust outer endpoints:

```text
P_outer_left
P_outer_right
v_G = P_outer_right - P_outer_left
```

Reject sample nếu:

```text
norm(v_G) < minimum_outer_endpoint_separation_px
or
dot(v_G, h) <= 0
```

Điều này chặn endpoint suy biến hoặc left/right ordering đảo.

```text
global_tilt_deg =
    degrees(atan2(dot(v_G, d), dot(v_G, h)))
```

Phải dùng endpoint formula. Approximation:

```text
(theta_right - theta_left) / 2
```

chỉ được log để consistency-check, không làm canonical value.

## 22.5 Center kink

```text
center_kink_deg =
    degrees(
        acos(
            clamp(
                dot(u_R, -u_L),
                -1,
                1
            )
        )
    )
```

Với góc nhỏ:

```text
center_kink_deg ≈ abs(theta_left + theta_right)
```

Approximation chỉ là diagnostic.

## 22.6 Frame-level angle quality

Một frame được vote nếu mỗi side đạt:

```text
linearity_ratio >= 0.90
median_orthogonal_residual <= 0.20 * side_thickness
projected_span >= minimum_side_projected_span_px
pixel_count >= minimum_side_pixels
coverage_ratio >= 0.85
```

Và:

```text
bridge PRESENT
both robust outer endpoints available
abs(s_anchor - s_trigger)
    <= angle_window_half_height_ratio * H
```

Bootstrap:

```text
angle_window_half_height_ratio = 0.08
```

## 22.7 Multi-frame aggregation

Yêu cầu:

```text
minimum_angle_frames = 3 independent bins
```

Aggregate từng metric bằng median:

```text
theta_left_final   = median(theta_left_i)
theta_right_final  = median(theta_right_i)
global_tilt_final  = median(global_tilt_i)
center_kink_final  = median(center_kink_i)
```

Stability:

```text
MAD(theta_left_i)  <= 1.5 deg
MAD(theta_right_i) <= 1.5 deg
MAD(global_tilt_i) <= 1.5 deg
MAD(center_kink_i) <= 1.5 deg
```

Bootstrap:

```text
maximum_angle_mad_deg = 1.5
angle_decision_guard_deg = 0.5
axis_orientation_epsilon = 0.10
minimum_outer_endpoint_separation_px = max(2, 0.02 * roi_width)
```

Nếu thiếu sample hoặc unstable:

```text
status = uncertain
reason = "insufficient_angle_frames"
```

hoặc:

```text
reason = "unstable_angle_measurement"
```

Không được trả `normal` khi chưa đủ điều kiện đo.

## 22.8 Bootstrap angle thresholds

```text
side_angle_threshold_deg = 8.0
global_tilt_threshold_deg = 5.0
center_kink_threshold_deg = 10.0
sign_deadband_deg = 1.0
```

Các ngưỡng là starting values cần validation; không được mô tả là dung sai cơ khí
nếu chưa được chủ thiết bị phê duyệt.

MAD ổn định không đủ để khẳng định median cách ngưỡng an toàn. Với mỗi magnitude
threshold `T`, dùng guard `g = angle_decision_guard_deg`:

```text
definitively_above(value, T): value > T + g
definitively_within(value, T): value <= T - g
gray_zone: còn lại
```

Bất kỳ metric cần quyết định mà nằm gray zone làm final `uncertain` với
`angle_threshold_guard_band`; không ép thành normal/bent. Điều này thay thế rule
“exact threshold không bent” đơn giản hơn nhưng không an toàn.

## 22.9 Angle business rules

```python
left_bent = definitively_above(abs(theta_left_final), side_angle_threshold_deg)
right_bent = definitively_above(abs(theta_right_final), side_angle_threshold_deg)
left_safe = definitively_within(abs(theta_left_final), side_angle_threshold_deg)
right_safe = definitively_within(abs(theta_right_final), side_angle_threshold_deg)

opposite_direction = (
    (
        theta_left_final < -sign_deadband_deg
        and theta_right_final > sign_deadband_deg
    )
    or
    (
        theta_left_final > sign_deadband_deg
        and theta_right_final < -sign_deadband_deg
    )
)

same_direction = (
    (
        theta_left_final > sign_deadband_deg
        and theta_right_final > sign_deadband_deg
    )
    or
    (
        theta_left_final < -sign_deadband_deg
        and theta_right_final < -sign_deadband_deg
    )
)

global_tilt_bent = (
    left_safe
    and right_safe
    and opposite_direction
    and definitively_above(abs(global_tilt_final), global_tilt_threshold_deg)
)

center_kink_bent = (
    left_safe
    and right_safe
    and same_direction
    and definitively_above(center_kink_final, center_kink_threshold_deg)
)
```

Final:

```python
if left_bent and right_bent:
    status = "bent_both"
elif left_bent and right_safe:
    status = "bent_left"
elif right_bent and left_safe:
    status = "bent_right"
elif global_tilt_bent:
    status = "bent_both"
elif center_kink_bent:
    status = "bent_both"
elif (
    left_safe
    and right_safe
    and definitively_within(abs(global_tilt_final), global_tilt_threshold_deg)
    and definitively_within(center_kink_final, center_kink_threshold_deg)
):
    status = "normal"
else:
    status = "uncertain"
    reason = "angle_threshold_guard_band"
```

## 22.10 Giới hạn angle

Các góc trên là image-plane angles. Chúng chỉ gần physical angles khi:

- camera cố định;
- camera gần vuông góc mặt phẳng chuyển động;
- lens distortion/perspective ổn định;
- ROI/centerline được cấu hình đúng.

Nếu cần độ chính xác cơ khí theo độ trong không gian 3D, phải bổ sung camera
calibration/homography. Release bootstrap không tự tuyên bố metric 3D.

`bent_*` mô tả lệch trục vĩ mô của side. Local curvature không làm thay đổi fitted
axis đủ lớn là ngoài phạm vi release này.

---

# 23. Final decision engine

## 23.1 Precedence

Thứ tự canonical:

1. hard identity/data/geometry invalidity;
2. single-side observability invariant;
3. center break;
4. localized side break;
5. angle eligibility;
6. bent classification;
7. normal.

Không đo angle trước khi loại breakage.

## 23.2 Pseudocode canonical

```python
def classify_event(evidence, capabilities, config):
    if evidence.has_hard_identity_or_geometry_conflict:
        return uncertain(
            reason=evidence.primary_conflict_reason,
            suspected_breakage=evidence.has_positive_break_evidence,
        )

    if evidence.is_single_side_only:
        return uncertain(
            reason="single_side_only_location_unidentifiable",
            suspected_breakage=True,
            possible_breakage_statuses=evidence.possible_breakage_hypotheses,
        )

    center = analyze_center(evidence, capabilities, config)
    left = analyze_side("left", evidence, config)
    right = analyze_side("right", evidence, config)

    if center == "CONFLICT" or left == "CONFLICT" or right == "CONFLICT":
        return uncertain(
            reason="conflicting_geometry_evidence",
            suspected_breakage=evidence.has_positive_break_evidence,
        )

    if center in {"BROKEN_TOPOLOGICAL", "BROKEN_TEMPORAL"}:
        if left == "VALID" and right == "VALID":
            return broken("broken_center", center=center)
        return uncertain(
            reason="center_break_with_side_state_unresolved",
            suspected_breakage=True,
        )

    if center == "INTACT":
        if left == "BROKEN_LOCALIZED" and right == "VALID":
            return broken("broken_left")
        if left == "VALID" and right == "BROKEN_LOCALIZED":
            return broken("broken_right")
        if left == "BROKEN_LOCALIZED" and right == "BROKEN_LOCALIZED":
            return uncertain(
                reason="both_sides_broken_no_canonical_label",
                suspected_breakage=True,
            )
        if left != "VALID" or right != "VALID":
            return uncertain(
                reason="side_integrity_unresolved",
                suspected_breakage=evidence.has_positive_break_evidence,
            )
        if not enabled(angle_classification):
            return uncertain(
                reason="model_capability_not_validated",
                suspected_breakage=False,
            )
        return classify_angles(evidence, config)

    # center == UNKNOWN
    if evidence.has_definitive_localized_left_break_with_both_sides_observed:
        return broken("broken_left")
    if evidence.has_definitive_localized_right_break_with_both_sides_observed:
        return broken("broken_right")

    return uncertain(
        reason="center_topology_unresolved",
        suspected_breakage=evidence.has_positive_or_temporal_break_suspicion,
    )
```

## 23.3 Truth table rút gọn

| Center | Left | Right | Điều kiện thêm | Final |
| --- | --- | --- | --- | --- |
| broken | valid | valid | identity rõ | `broken_center` |
| broken | khác valid | bất kỳ | — | `uncertain` |
| intact | broken | valid | localized positive | `broken_left` |
| intact | valid | broken | localized positive | `broken_right` |
| intact | broken | broken | không có combined label | `uncertain` |
| intact | valid | valid | angle đủ và ổn định | angle result/`normal` |
| intact | valid | valid | angle thiếu/không ổn định | `uncertain` |
| unknown | broken | valid | cả hai phía đã quan sát, center không có conflict | `broken_left` |
| unknown | valid | broken | đối xứng | `broken_right` |
| unknown/conflict | bất kỳ | bất kỳ | còn lại | `uncertain` |
| bất kỳ | chỉ một phía trong toàn event | current model | — | `uncertain` |

## 23.4 Không có fallback đoán nhãn

Không được:

- đổi `uncertain` thành `normal` để giữ count;
- chọn nhãn có confidence cao nhất nếu chưa qua hard rules;
- dùng VLM để override;
- dùng previous/next paddle label để điền;
- mặc định một side là intact khi nó không emit;
- tạo `broken_both`.

## 23.5 Confidence và diagnostic support

Bootstrap không có calibrated probability model. Vì vậy:

```text
decision_confidence = null
confidence_semantics = "unavailable_until_calibrated"
```

`decision_confidence` không tham gia verdict.

Để UI cũ vẫn có finite `score`, tính diagnostic:

```text
evidence_support_score =
    min(1.0, definitive_support_bins / 3.0)
```

Trong đó:

```text
normal/bent:
    definitive_support_bins = valid angle bins

broken_center same-frame:
    definitive_support_bins = disconnected same-frame bins

broken_center temporal:
    definitive_support_bins = min(left_present_bins, right_present_bins)

broken_left/right:
    definitive_support_bins = localized-break bins của side đó

uncertain:
    definitive_support_bins = 0
```

Compatibility `BarResult.score = evidence_support_score`.

Score này chỉ mô tả số support bins đã cap; không phải probability, không override
hard decision rules, và không được gọi là confidence trong UI/report.

## 23.6 Threshold equality

Để deterministic, operator tại từng rule là canonical:

- angle/bent: `> threshold + configured_guard` để broken/bent và
  `<= threshold - configured_guard` để normal; khoảng giữa là `uncertain`;
- side internal-gap broken threshold: `>=`;
- side short-length broken threshold: `<=`;
- minimum quality/support/count: `>=`;
- maximum quality/stability/gap: `<=`.

Không áp một blanket operator cho mọi “lỗi”. Không dùng epsilon ẩn; nếu cần epsilon
số học phải cấu hình/log.

---

# 24. Canonical result contract

## 24.1 Schema và status

```text
result_schema_version = "geometry_v2_result/2.0"
```

`vision_status` là verdict canonical:

```text
normal
bent_left
bent_right
bent_both
broken_left
broken_right
broken_center
uncertain
```

Consumer mới phải đọc `vision_status`, không suy verdict từ `vlm_called`.

`vision_status` là machine verdict immutable. Sau human review:

```text
effective_status =
    final_reviewed_status
    if final_reviewed_status is not null
    else vision_status
    if vision_status is not null
    else legacy_effective_status
```

UI/report hiển thị `effective_status`; không ghi đè `vision_status`.

## 24.2 Per-paddle result

JSON sau chỉ minh họa shape/type, không phải một observation hoặc benchmark record
đã kiểm chứng.

```json
{
  "schema_version": "geometry_v2_result/2.0",
  "rule_version": "geometry_v2_rules/2.0.0",
  "algorithm_config_hash": "<sha256>",
  "capability_record_hash": "<sha256>",
  "resolved_deployment_hash": "<sha256>",
  "system_signature_hash": "<sha256>",

  "bar_id": "job123_track_000017",
  "paddle_id": 12,
  "track_id": 17,
  "track_ids": [17, 23],

  "frame_id": 105,
  "source_frame_ids": [101, 105, 109],
  "source_timestamp_sec": 3.466666667,

  "vision_status": "broken_center",
  "final_reviewed_status": null,
  "classification_source": "geometry_v2",
  "decision_confidence": null,
  "confidence_semantics": "unavailable_until_calibrated",
  "evidence_support_score": 0.666666667,

  "suspected_breakage": true,
  "possible_breakage_statuses": ["broken_center"],
  "review_required": false,

  "result": "suspected_defect",
  "defect_type": "broken",
  "vlm_called": false,
  "rule_result": "suspected_defect",
  "score": 0.666666667,

  "reason_codes": [
    "center_disconnected_same_frame_multi_bin"
  ],
  "reasons": [
    "center_disconnected_same_frame_multi_bin"
  ],
  "reason_details": {
    "connected_bridge_bins": 0,
    "disconnected_same_frame_bins": 2
  },

  "geometry_states": {
    "center": "BROKEN_TOPOLOGICAL",
    "left": "VALID",
    "right": "VALID",
    "observability_grade": "GRADE_A"
  },

  "angles": {
    "left_deg": null,
    "right_deg": null,
    "global_tilt_deg": null,
    "center_kink_deg": null,
    "left_mad_deg": null,
    "right_mad_deg": null,
    "global_tilt_mad_deg": null,
    "center_kink_mad_deg": null,
    "sample_count": 0
  },

  "side_metrics": {
    "left": {
      "projected_length_px": 211.3,
      "expected_length_px": 214.0,
      "coverage_ratio": 0.93,
      "largest_internal_gap_ratio": 0.03,
      "linearity_ratio": 0.96
    },
    "right": {
      "projected_length_px": 208.7,
      "expected_length_px": 210.0,
      "coverage_ratio": 0.91,
      "largest_internal_gap_ratio": 0.04,
      "linearity_ratio": 0.95
    }
  },

  "evidence_summary": {
    "connected_bridge_bins": 0,
    "disconnected_same_frame_bins": 2,
    "left_present_bins": 4,
    "right_present_bins": 3,
    "left_opportunity_bins": 5,
    "right_opportunity_bins": 5,
    "joint_two_side_opportunity_bins": 5,
    "ambiguous_bins": 0
  },

  "snapshot": {
    "snapshot_type": "single_frame_geometry",
    "filename": "track_000017_frame_000000105.jpg",
    "primary_source_frame_id": 105,
    "evidence_source_frame_ids": [101, 105]
  },

  "snapshot_key": "results/job123/snapshots/defects/track_000017_frame_000000105.jpg",
  "length": 419.4,
  "width": 22.1,
  "thresholds": {},
  "margins": {},

  "diagnostics": {
    "association_ambiguous": false,
    "boundary_truncated": false,
    "model_hash_verified": true
  }
}
```

Trong core `BarResult`, `snapshot_key` chưa có; server enrich flat field này sau
khi upload R2. Filename/frame IDs nằm trong `snapshot` metadata trước upload.

## 24.3 Nullability

Phải dùng JSON `null` khi metric không hợp lệ.

Không được serialize:

```text
NaN
+Infinity
-Infinity
```

Angle fields bắt buộc `null` cho:

- `broken_*`;
- `uncertain`;
- insufficient angle frames;
- unstable measurement.

Không dùng `0.0` thay cho unknown angle.

## 24.4 Compatibility mapping

| `vision_status` | legacy `result` | legacy `defect_type` |
| --- | --- | --- |
| `normal` | `normal` | `normal` |
| `bent_left` | `suspected_defect` | `bent_left` |
| `bent_right` | `suspected_defect` | `bent_right` |
| `bent_both` | `suspected_defect` | `bent_both` |
| `broken_left` | `suspected_defect` | `broken` |
| `broken_right` | `suspected_defect` | `broken` |
| `broken_center` | `suspected_defect` | `broken` |
| `uncertain` | `suspected_defect` | `null` |

Mọi geometry-v2 result:

```text
classification_source = "geometry_v2"
vlm_called = false
```

`vlm_called` được giữ chỉ để compatibility; nó không còn là cờ “đã phân loại”.

## 24.5 Compatibility measurements

`measurements` cũ được giữ:

```json
{
  "length": 419.4,
  "width": 22.1
}
```

Trong geometry-v2:

- `length` là khoảng robust từ outer-left tới outer-right khi cả hai endpoint hợp
  lệ;
- `width` là median side thickness;
- nếu không hợp lệ, giá trị compatibility là `0.0`;
- canonical field `legacy_measurements_available` phải cho biết có được phép hiển
  thị hay không.

Frontend geometry-v2 phải hiển thị `—` nếu:

```text
legacy_measurements_available == false
```

Không được trình bày `0.0` như phép đo vật lý.

## 24.6 Job summary

JSON sau chỉ minh họa schema. Mọi numeric count/metric là illustrative và không
được trích làm verification evidence.

```json
{
  "schema_version": "geometry_v2_summary/2.0",
  "paddle_schema_version": "geometry_v2_result/2.0",
  "inspection_mode": "geometry_v2",
  "success": true,
  "failure_reason": "",

  "count_certified": true,
  "possible_event_count_min": 42,
  "possible_event_count_max": 42,
  "report_export_allowed": true,
  "identity_conflict_groups": [],

  "total_bars": 42,
  "normal_bars": 34,
  "confirmed_defect_bars": 6,
  "uncertain_bars": 2,
  "review_required_bars": 2,

  "defect_bars": 8,
  "frames_scanned": 936,

  "status_counts": {
    "normal": 34,
    "bent_left": 1,
    "bent_right": 1,
    "bent_both": 1,
    "broken_left": 1,
    "broken_right": 1,
    "broken_center": 1,
    "uncertain": 2
  },

  "model": {
    "sha256": "ef05955f43c8db6d2ff76b72fb65806e69afe525e85d8486eeb2dfb7566dcd65",
    "input_size": 640,
    "hash_verified": true,
    "artifact_manifest_id": "<immutable-id>",
    "adapter_version": "current_yolo_seg_adapter/2.0",
    "preprocess_fingerprint": "<sha256>",
    "postprocess_fingerprint": "<sha256>"
  },

  "capability": {
    "record_hash": "<sha256>",
    "system_signature_hash": "<sha256>",
    "validated_domain_id": "<deployment-profile-id>"
  },

  "runtime": {
    "fingerprint": "<sha256>",
    "execution_provider": "CPUExecutionProvider",
    "timestamp_source": "decoder_pts"
  },

  "geometry": {
    "input_schema_version": "geometry_input/2.0",
    "rule_version": "geometry_v2_rules/2.0.0",
    "algorithm_config_hash": "<sha256-of-canonical-json>",
    "resolved_deployment_hash": "<sha256>",
    "centerline": {
      "top": {"x": 500.0, "y": 0.0},
      "bottom": {"x": 515.0, "y": 540.0}
    }
  },

  "diagnostics": {
    "raw_detections": 4100,
    "accepted_components": 612,
    "deduplicated_components": 18,
    "online_tracklets": 45,
    "fused_events": 42,
    "ambiguous_events": 2,
    "dropped_synthetic_frames": 0
  },

  "vlm_request_count": 0,
  "defects": [],
  "normals": []
}
```

Compatibility:

```text
defect_bars = confirmed_defect_bars + uncertain_bars
```

vì UI cũ chỉ có hai buckets và `uncertain` phải được đưa ra review thay vì nằm
trong normal.

Khi `count_certified=true`, invariant:

```text
normal_bars
+ confirmed_defect_bars
+ uncertain_bars
== total_bars
```

Khi `count_certified=false`:

```text
success = false
failure_reason = "event_cardinality_unresolved"
report_export_allowed = false
success-path count/bucket fields are omitted from failure summary
```

Summary vẫn giữ diagnostics và `possible_event_count_min/max` cùng từng
`identity_conflict_group` để audit/review, nhưng không phải report inspection có
count chính xác và không thay bằng count một partition tùy ý.

Result summary MAY giữ `possible_event_count_min/max` và conflict diagnostics để
audit, nhưng không trả completed inspection counts/buckets. Điều này giữ
`BatchInspectionResult` legacy count fields là integer cho success path và tránh
đưa fake zero/null count vào consumer cũ như một report thành công.

## 24.7 Empty inspection

Nếu video đọc được nhưng:

```text
fused reportable physical events < minimum_reportable_events
```

bootstrap:

```text
minimum_reportable_events = 1
```

thì job:

```text
success = false
failure_reason = "no_reportable_paddles"
```

Không trả một báo cáo “0 lỗi” như thể inspection đã thành công.

Tất cả events `uncertain` vẫn là job thành công về mặt xử lý, nhưng report export
phải yêu cầu review/correction.

---

# 25. Stable reason-code registry

## 25.1 Quy tắc

Reason code:

- ASCII lowercase snake_case;
- stable giữa patch releases;
- machine-readable;
- không nhúng số động vào string;
- chi tiết số nằm trong `reason_details`;
- một result có một `primary_reason` và danh sách `reason_codes`.

## 25.2 Job-level failures

```text
invalid_video_fps
invalid_video_timestamps
video_open_failed
video_geometry_changed
geometry_invalid_roi
geometry_invalid_centerline
geometry_invalid_chain_band
geometry_insufficient_field_of_view
geometry_config_invalid
model_file_missing
model_hash_mismatch
model_contract_mismatch
model_inference_failed
geometry_resource_limit_exceeded
evidence_artifact_integrity_error
snapshot_write_failed
result_serialization_failed
no_reportable_paddles
event_cardinality_unresolved
```

Job-level failure không được chuyển thành per-paddle `uncertain`.

## 25.3 Tracking/identity

```text
track_too_short
partial_start_event
partial_end_event
crossing_time_unresolved
association_ambiguous
fusion_ambiguous
trajectory_gate_failed
multiple_paddles_merged
duplicate_instance_removed
event_order_conflict
```

## 25.4 Observability/FOV/model

```text
single_side_only_location_unidentifiable
insufficient_independent_evidence
left_field_of_view_truncated
right_field_of_view_truncated
center_corridor_truncated
mask_bbox_crop_limits_negative_evidence
model_dropout_suspected
model_capability_not_validated
operating_domain_not_validated
duplicate_mask_geometry_disagreement
synthetic_frame_excluded
```

## 25.5 Center topology

```text
center_connected_multi_bin
center_disconnected_same_frame_multi_bin
center_disconnected_temporal_multi_bin
center_topology_unresolved
center_topology_conflict
single_connected_frame_insufficient
single_disconnected_frame_insufficient
center_break_with_side_state_unresolved
```

## 25.6 Side integrity

```text
left_localized_shortness
right_localized_shortness
left_localized_internal_gap
right_localized_internal_gap
side_reference_unavailable
side_reference_insufficient_samples
side_integrity_unresolved
side_geometry_unstable
both_sides_broken_no_canonical_label
conflicting_geometry_evidence
```

## 25.7 Angle

```text
insufficient_angle_frames
unstable_angle_measurement
angle_threshold_guard_band
left_side_fit_invalid
right_side_fit_invalid
angle_not_allowed_for_broken_event
angle_not_allowed_for_uncertain_event
```

## 25.8 Decision

```text
normal_geometry_within_thresholds
bent_left_side_angle
bent_right_side_angle
bent_both_side_angles
bent_both_global_tilt
bent_both_center_kink
manual_review_required
```

Primary-reason precedence:

1. identity/data invalidity;
2. observability impossibility;
3. topology conflict;
4. positive break evidence;
5. angle insufficiency/conflict;
6. definitive status reason.

---

# 26. Snapshot và visual evidence

## 26.1 Filename contract

Giữ nguyên:

```text
track_{track_id:06d}_frame_{frame_id:09d}.jpg
```

Ví dụ:

```text
track_000017_frame_000000105.jpg
```

`frame_id` là primary original source frame ID.

## 26.2 Snapshot types

```text
single_frame_geometry
temporal_composite
uncertain_diagnostic
```

Selection:

- `broken_center` topology: best `DISCONNECTED_SAME_FRAME`;
- `broken_center` temporal: best left frame + best right frame trong
  `temporal_composite`;
- `broken_left/right`: best localized-break frame;
- `bent_*`/`normal`: best valid same-frame angle sample;
- `uncertain`: frame/composite thể hiện primary reason rõ nhất.

## 26.3 Temporal composite

Temporal composite:

- đặt panels cạnh nhau;
- mỗi panel hiển thị source frame ID và timestamp;
- không union masks từ hai thời điểm lên một ảnh;
- không giả vờ hai cánh xuất hiện cùng một frame;
- ghi rõ `TEMPORAL EVIDENCE`;
- metadata giữ `evidence_source_frame_ids`.

## 26.4 Overlay bắt buộc

Snapshot geometry-v2 SHOULD hiển thị:

- ROI boundary;
- chain centerline;
- chain band;
- trigger strip;
- accepted left/right/center components bằng màu khác nhau;
- duplicate/ambiguous components bằng nét đứt;
- fitted side axes;
- inner/outer endpoints;
- `paddle_id`, `track_ids`;
- `vision_status`;
- primary reason;
- angle values chỉ khi hợp lệ.

Không overlay confidence/model text lên vùng che mất geometry cần review.

## 26.5 Raw evidence preservation

Trong debug/audit mode, MAY lưu:

- raw source crop;
- raw model masks;
- topology mask;
- geometry mask;
- rendered snapshot;
- observation/evidence JSON.

Production retention phải tuân policy dung lượng và bảo mật. Canonical report chỉ
phụ thuộc rendered snapshot và result JSON.

## 26.6 Snapshot failure

Nếu classification thành công nhưng snapshot của một event không ghi được:

- không đổi geometry verdict;
- job failure theo bootstrap strict policy:
  `snapshot_write_failed`;
- không silently hoàn tất báo cáo thiếu evidence.

Policy MAY đổi thành completed-with-warning sau khi server schema hỗ trợ trạng thái
đó; release hiện tại chưa có.

---

# 27. Configuration contract

## 27.1 File riêng

Tạo:

```text
config/geometry_v2.json
```

Không nhét các field geometry vào `base_profile.json` hiện tại, vì parser profile
đang strict. Cách này:

- không phá hai inspection modes cũ;
- không buộc migration profile ngay;
- cho phép version/hash config độc lập.

Unknown keys:

```text
extra = forbid
```

Tất cả numbers:

- finite;
- đúng type;
- trong range;
- cross-field validated.

## 27.2 Bootstrap configuration đầy đủ

```json
{
  "schema_version": "geometry_v2_config/2.0",
  "rule_version": "geometry_v2_rules/2.0.0",

  "model": {
    "artifact_path": "weights/model_imgsz_640/best.onnx",
    "expected_sha256": "ef05955f43c8db6d2ff76b72fb65806e69afe525e85d8486eeb2dfb7566dcd65",
    "expected_input_size": 640,
    "fail_on_hash_mismatch": true,
    "artifact_manifest_version": "model_artifact/1.0"
  },

  "video": {
    "decoder_backend": "pyav",
    "timestamp_epsilon_sec": 0.000000001,
    "allow_cfr_index_fallback": true,
    "require_cfr_confirmation_for_fallback": true
  },

  "deployment": {
    "capability_record_version": "geometry_capabilities/1.0",
    "outside_domain_policy": "uncertain"
  },

  "geometry": {
    "minimum_roi_width_px": 160,
    "minimum_roi_height_px": 160,
    "minimum_centerline_span_ratio": 0.70,
    "maximum_allowed_roll_deg": 15.0,
    "minimum_side_field_of_view_ratio": 0.25,
    "chain_band_width_ratio_min": 0.02,
    "chain_band_width_ratio_max": 0.20,
    "default_chain_band_width_ratio": 0.05,
    "motion_direction": "positive_s"
  },

  "components": {
    "connectivity": 8,
    "minimum_absolute_area_px": 16,
    "minimum_roi_area_ratio": 0.00002,
    "minimum_instance_area_ratio": 0.005,
    "anchor_nearest_pixel_ratio": 0.10,
    "minimum_anchor_pixels": 5,
    "anchor_histogram_bin_height_ratio": 0.002,
    "anchor_histogram_window_bins": 3,
    "maximum_anchor_spread_ratio": 0.05,
    "secondary_anchor_peak_ratio": 0.80,
    "boundary_margin_ratio": 0.003,
    "topology_morphology": "none",
    "geometry_fill_small_holes": false
  },

  "deduplication": {
    "minimum_overlap_over_smaller": 0.85,
    "minimum_iou": 0.70,
    "anchor_gate_ratio": 0.03
  },

  "observations": {
    "same_frame_anchor_gate_ratio": 0.03,
    "multi_anchor_separation_ratio": 0.08,
    "pairing_ambiguity_margin": 0.01,
    "pairing_uncertainty_weight": 0.25,
    "unmatched_cost": 0.05
  },

  "tracking": {
    "minimum_track_hits": 2,
    "minimum_track_duration_sec": 0.04,
    "maximum_track_gap_sec": 0.35,
    "maximum_nis": 9.0,
    "maximum_absolute_innovation_ratio": 0.08,
    "maximum_reverse_ratio": 0.015,
    "sigma_acceleration_ratio_per_sec2": 0.15,
    "minimum_measurement_sigma_px": 1.0,
    "minimum_velocity_sigma_ratio_per_sec": 0.02,
    "miss_track_cost": 0.65,
    "new_track_cost": 0.65,
    "association_ambiguity_margin": 0.03,
    "cost_weights": {
      "nis": 0.55,
      "anchor": 0.30,
      "type": 0.15
    },
    "seed_cost_weights": {
      "anchor": 0.80,
      "type": 0.20
    }
  },

  "trigger": {
    "center_ratio": 0.50,
    "height_ratio": 0.20,
    "preferred_evidence_window_half_height_ratio": 0.20,
    "minimum_velocity_ratio_per_sec": 0.05,
    "maximum_crossing_extrapolation_sec": 0.10,
    "maximum_crossing_sigma_sec": 0.05
  },

  "evidence": {
    "minimum_spacing_frames": 2,
    "minimum_spacing_sec": 0.05,
    "top_k_per_type": 8,
    "maximum_metadata_observations_per_track": 256,
    "minimum_left_presence_bins": 2,
    "minimum_right_presence_bins": 2,
    "minimum_joint_two_side_opportunity_bins": 4
  },

  "fusion": {
    "maximum_crossing_delta_sec": 0.12,
    "maximum_crossing_interval_ratio": 0.35,
    "minimum_uncertainty_gate_sec": 0.04,
    "uncertainty_sigma_multiplier": 3.0,
    "maximum_relative_velocity_delta": 0.25,
    "velocity_epsilon_ratio_per_sec": 0.001,
    "maximum_trajectory_residual_ratio": 0.035,
    "maximum_fusion_extrapolation_sec": 0.10,
    "ambiguity_margin": 0.10,
    "maximum_identity_conflict_tracklets": 32,
    "maximum_identity_hypotheses_per_group": 8,
    "minimum_unambiguous_events_for_interval": 5,
    "expected_paddle_interval_sec": null,
    "expected_paddle_interval_tolerance_ratio": 0.20
  },

  "center_topology": {
    "q_bins": 20,
    "minimum_q_coverage": 0.90,
    "minimum_cross_section_thickness_ratio": 0.20,
    "corridor_half_thickness_multiplier": 0.75,
    "corridor_minimum_half_height_ratio": 0.012,
    "minimum_plausible_side_thickness_roi_width_ratio": 0.002,
    "maximum_plausible_side_thickness_roi_width_ratio": 0.08,
    "inner_extent_chain_band_multiplier": 3.0,
    "inner_extent_available_side_ratio": 0.20,
    "minimum_connected_bins": 2,
    "minimum_disconnected_same_frame_bins": 2
  },

  "side_integrity": {
    "exclusion_margin_roi_width_ratio": 0.005,
    "minimum_side_pixels": 80,
    "minimum_projected_span_px": 20,
    "minimum_projected_span_roi_width_ratio": 0.04,
    "side_outlier_residual_thickness_ratio": 0.35,
    "maximum_side_fit_iterations": 5,
    "minimum_linearity_ratio": 0.90,
    "maximum_median_residual_thickness_ratio": 0.20,
    "coverage_bins": 20,
    "minimum_coverage_pixels_per_bin": 5,
    "minimum_coverage_thickness_ratio": 0.25,
    "minimum_intrinsic_profile_bins": 10,
    "valid_minimum_coverage_ratio": 0.85,
    "valid_maximum_internal_gap_ratio": 0.08,
    "broken_minimum_internal_gap_ratio": 0.15,
    "broken_maximum_length_ratio": 0.72,
    "minimum_evidence_bins": 2,
    "minimum_support_ratio": 0.60,
    "minimum_reference_other_paddles": 5,
    "reference_percentile": 90.0,
    "reference_anchor_gate_ratio": 0.03,
    "minimum_fov_margin_ratio": 0.05,
    "expected_left_extent_px_at_trigger": null,
    "expected_right_extent_px_at_trigger": null,
    "expected_extent_tolerance_ratio": 0.12
  },

  "angle": {
    "window_half_height_ratio": 0.08,
    "minimum_frames": 3,
    "maximum_mad_deg": 1.5,
    "side_threshold_deg": 8.0,
    "global_tilt_threshold_deg": 5.0,
    "center_kink_threshold_deg": 10.0,
    "sign_deadband_deg": 1.0,
    "decision_guard_deg": 0.5,
    "axis_orientation_epsilon": 0.10,
    "minimum_outer_endpoint_separation_px": 2,
    "minimum_outer_endpoint_separation_roi_width_ratio": 0.02
  },

  "decision": {
    "minimum_reportable_events": 1,
    "single_side_only_policy": "uncertain",
    "both_sides_broken_policy": "uncertain",
    "vlm_policy": "disabled"
  },

  "snapshots": {
    "strict_write": true,
    "save_debug_artifacts": false,
    "jpeg_quality": 92
  },

  "limits": {
    "maximum_roi_pixels": 2073600,
    "maximum_instances_for_mask_reconstruction_per_frame": 64,
    "maximum_active_tracks": 256,
    "maximum_components_per_frame": 256,
    "maximum_events_per_job": 100000,
    "maximum_transient_mask_bytes": 268435456,
    "maximum_in_memory_evidence_bytes": 134217728,
    "maximum_spooled_evidence_bytes": 4294967296,
    "maximum_spooled_evidence_bytes_per_event": 8388608,
    "binary_mask_encoding": "bbox_local_rle"
  },

  "determinism": {
    "cost_quantization": 0.000000001,
    "single_thread_geometry_reductions": true
  }
}
```

## 27.3 Resolved runtime values

Config loader phải resolve và lưu:

```text
H
chain_band_width_px
chain_band_half_width_px
boundary_margin_px
side_exclusion_margin_px
minimum_component_area_px formula inputs
minimum_side_projected_span_px resolved max
minimum_plausible/maximum_plausible_side_thickness_px
minimum_outer_endpoint_separation_px resolved max
angle_decision_guard_deg
```

Canonical JSON serialization:

- UTF-8;
- keys sort;
- compact separators;
- no NaN;
- SHA-256.

Serialization này áp dụng cho projection thuật toán; phải bỏ
`model.artifact_path` trước khi hash. Path nguồn có thể được log riêng để chẩn đoán,
nhưng không được đi vào signature/identity. Hash kết quả là
`algorithm_config_hash` và được lưu vào job result. Deployment binding được tính
riêng:

```text
resolved_deployment_hash = SHA256(
    algorithm_config_hash || capability_record_hash || artifact_manifest_id
)
```

Nó không thay thế ba IDs/hashes thành phần.

## 27.4 Cross-field validation

Ít nhất phải kiểm:

```text
0 < valid_maximum_internal_gap_ratio
  < broken_minimum_internal_gap_ratio < 1

0 < broken_maximum_length_ratio
  < 1 - expected_extent_tolerance_ratio <= 1

0 < minimum_plausible_side_thickness_roi_width_ratio
  < maximum_plausible_side_thickness_roi_width_ratio

0 <= angle.decision_guard_deg
  < min(angle.side_threshold_deg,
        angle.global_tilt_threshold_deg,
        angle.center_kink_threshold_deg)

0 < minimum_support_ratio <= 1
0 < minimum_q_coverage <= 1

minimum_connected_bins >= 2
minimum_disconnected_same_frame_bins >= 2
minimum_angle_frames >= 3

maximum_track_gap_sec > 0
maximum_crossing_delta_sec > 0
maximum_crossing_sigma_sec > 0
maximum_fusion_extrapolation_sec > 0
velocity_epsilon_ratio_per_sec > 0
timestamp_epsilon_sec > 0

maximum_identity_conflict_tracklets >= 2
maximum_identity_hypotheses_per_group >= 1

chain_band_width_ratio_min
  <= default_chain_band_width_ratio
  <= chain_band_width_ratio_max

sum(tracking.cost_weights) == 1 within 1e-9
```

Unknown enum/rule/policy phải làm geometry job initialization fail closed, không
fallback. Compatibility release không được làm toàn server/legacy jobs chết chỉ vì
geometry config lỗi; health/runtime-config phải báo geometry unavailable.

## 27.5 Semantic versioning

- đổi typo/comment: không cần bump rule version;
- đổi threshold mặc định: bump patch;
- đổi decision semantics/reason mapping: bump minor;
- đổi canonical status/schema incompatibly: bump major.

Mọi result phải lưu cả:

```text
schema_version
rule_version
algorithm_config_hash
capability_record_hash
resolved_deployment_hash
model_hash
```

---

# 28. Tích hợp với API, worker, DB, frontend và report

## 28.1 API request models

Server hiện dùng Pydantic `extra="forbid"`. Vì vậy phải mở rộng schema chính thức,
không thể gửi thêm geometry ad hoc.

Schema đề xuất:

```python
class PointIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    x: float
    y: float


class ChainCenterlineIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    top: PointIn
    bottom: PointIn


class GeometryIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: Literal["geometry_input/2.0"]
    chain_centerline: ChainCenterlineIn
    chain_band_width_ratio: float | None = None
    motion_direction: Literal["positive_s", "negative_s"] | None = None


class CreateJobIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    content_type: str
    size_bytes: int
    roi: RoiIn
    geometry: GeometryIn | None = None
    inspector_name: str
    conveyor_name: str
    inspection_mode: str | None = None
```

Mọi geometry number phải có `mode="before"` validator:

```text
reject bool
reject string/coercion
accept only int/float
require math.isfinite(value)
convert accepted value to float
apply closed/open range
```

`extra="forbid"` một mình không bảo đảm strict/finite. `NaN`/`Infinity` phải bị
reject trước `json.dumps`.

Cross-field:

```text
effective_mode =
    body.inspection_mode or base_profile.inspection.mode

effective_mode == "geometry_v2"
    -> geometry MUST be present

effective_mode != "geometry_v2"
    -> geometry MUST be absent
```

Compatibility release giữ base profile default là legacy. Geometry frontend luôn
gửi explicit `inspection_mode="geometry_v2"`; geometry-v2 không được là implicit
server default, nếu không old payload thiếu geometry sẽ bị phá compatibility.

`chain_band_width_ratio`/`motion_direction` optional override server defaults:

```text
job override > geometry_v2.json default
```

Server không nhận `vision_config_path` từ remote client. Path config là
server-controlled setting để tránh path traversal/config injection.

CLI local MAY nhận path đã resolve/validate.

## 28.2 Coordinate contract ở API

`chain_centerline` luôn là ROI-local pixels.

Không nhận normalized coordinates ở schema 2.0. Frontend có thể lưu normalized
tạm thời, nhưng trước request phải chuyển sang ROI-local pixel bằng actual source
frame dimensions.

Frontend conversion:

```text
roi_local_x =
    (handle_canvas_x - roi_canvas_x) * roi_width / roi_canvas_width

roi_local_y =
    (handle_canvas_y - roi_canvas_y) * roi_height / roi_canvas_height
```

Clamp chỉ trong continuous bounds `[0,w] × [0,h]`; không silently clamp một
invalid value ngoài tolerance.

Server phải trả validation error có field path rõ ràng, ví dụ:

```json
{
  "detail": [
    {
      "loc": ["body", "geometry", "chain_centerline"],
      "msg": "centerline span must be at least 70% of ROI height",
      "type": "value_error"
    }
  ]
}
```

### Runtime-config response

`GET /api/runtime-config` giữ keys legacy và bổ sung:

```json
{
  "inspection": {
    "mode": "auto_baseline",
    "supported_modes": [
      "auto_baseline",
      "average_ratio",
      "geometry_v2"
    ]
  },
  "geometry_v2": {
    "available": true,
    "unavailable_reason": null,
    "input_schema_version": "geometry_input/2.0",
    "defaults": {
      "chain_band_width_ratio": 0.05,
      "motion_direction": "positive_s"
    },
    "validation": {
      "minimum_centerline_span_ratio": 0.70,
      "maximum_allowed_roll_deg": 15.0,
      "minimum_chain_band_width_ratio": 0.02,
      "maximum_chain_band_width_ratio": 0.20
    },
    "trigger": {
      "center_ratio": 0.50,
      "height_ratio": 0.20
    }
  }
}
```

Frontend lấy preview/default từ response; không hardcode duplicate thresholds
trong JavaScript.

## 28.3 Persistence không cần DB migration

Giữ schema SQLite hiện tại.

Persist vào `roi_config_json`:

```json
{
  "x": 100,
  "y": 80,
  "w": 1000,
  "h": 540,
  "frame_width": 1280,
  "frame_height": 720,
  "geometry": {
    "schema_version": "geometry_input/2.0",
    "chain_centerline": {
      "top": {"x": 500.0, "y": 0.0},
      "bottom": {"x": 515.0, "y": 540.0}
    },
    "chain_band_width_ratio": 0.05,
    "motion_direction": "positive_s"
  }
}
```

`result_summary_json` đã là JSON linh hoạt, nên chứa summary v2 mà không thêm cột.

## 28.4 Worker extraction

Parser profile hiện tại chỉ chấp nhận đúng sáu ROI keys. Worker phải tách:

```python
stored = json.loads(row["roi_config_json"])

roi_only = {
    key: stored[key]
    for key in (
        "x", "y", "w", "h",
        "frame_width", "frame_height",
    )
}

geometry_input = stored.get("geometry")
profile = profile.with_roi(roi_only)
```

Không truyền nguyên `stored` vào `Profile.with_roi`.

Legacy rows chỉ có sáu ROI keys phải tiếp tục hoạt động.

`Profile.with_roi()` hiện auto-select model 320/416/640 theo ROI long side. Sau
call này, geometry branch phải tạo model config riêng:

```text
geometry_model = deepcopy(profile.model)
geometry_model.path = geometry_config.model.artifact_path
geometry_model.input_size = geometry_config.model.expected_input_size
geometry_model.model_zoo = {}
verify path + input + SHA-256 before inference
```

Không mutate shared/base profile và không thay behavior legacy model-zoo.

Tests bắt buộc:

```text
ROI long side 300 -> geometry still 640
ROI long side 400 -> geometry still 640
ROI long side 500 -> geometry still 640
```

## 28.5 Bypass slow-motion đúng mode

Current worker tạo slow-motion video trước inference. Với geometry-v2:

Worker phải load base profile, resolve `effective inspection_mode`, parse stored
geometry và validate mode/geometry trước preprocessing branch. Tức di chuyển
mode-resolution logic hiện nằm sau slowdown lên trước nó.

```python
if inspection_mode == "geometry_v2":
    inference_source = original_video_path
else:
    inference_source = existing_slow_motion_policy(...)
```

Không sửa `server/preprocess.py` và không đổi behavior legacy.

Nếu optical flow được thêm về sau:

- frame nội suy chỉ hỗ trợ prediction/tracking;
- `is_original = false`;
- không vote topology/side/angle;
- không được làm snapshot primary.

## 28.6 Inspection-mode registry

Thêm:

```python
GEOMETRY_V2_INSPECTION_MODE = "geometry_v2"
```

vào registry hiện hữu và `SUPPORTED_INSPECTION_MODES`.

Không đổi tên:

```text
auto_baseline
average_ratio
```

## 28.7 Dispatch tại biên pipeline

`run_batch_inspection` giữ entry point hiện tại và dispatch sớm:

```python
if inspection_mode == GEOMETRY_V2_INSPECTION_MODE:
    return run_geometry_v2_inspection(
        profile=profile,
        source=source,
        run_id=run_id,
        snapshots_root=snapshots_root,
        geometry_input=geometry_input,
        geometry_config=geometry_config,
    )

# Không return/helper mới ở đây.
# Toàn bộ legacy body hiện tại tiếp tục fall through nguyên chỗ.
```

Không extract legacy body thành `run_legacy_batch_inspection`; helper đó chưa tồn
tại và sẽ tạo diff lớn. Chỉ early-return geometry rồi để legacy body hiện tại fall
through nguyên chỗ. Không chèn hàng loạt `if geometry_v2` vào legacy frame loop.

Signature mới chỉ thêm optional keyword arguments ở cuối:

```text
geometry_input = None
geometry_config = None
```

Legacy callers không đổi.

## 28.8 Mở rộng dataclasses tương thích

Thêm optional/default fields ở cuối `BarResult`:

```python
paddle_id: int | None = None
track_ids: tuple[int, ...] = ()
vision_status: str | None = None
final_reviewed_status: str | None = None
classification_source: str | None = None
decision_confidence: float | None = None
evidence_support_score: float = 0.0
suspected_breakage: bool = False
possible_breakage_statuses: tuple[str, ...] = ()
review_required: bool = False
geometry_analysis: dict = field(default_factory=dict)
snapshot_metadata: dict = field(default_factory=dict)
legacy_measurements_available: bool = True
```

Thêm optional/default fields ở cuối `BatchInspectionResult`:

```python
confirmed_defect_bars: int = 0
uncertain_bars: int = 0
review_required_bars: int = 0
status_counts: dict[str, int] = field(default_factory=dict)
geometry_diagnostics: dict = field(default_factory=dict)
inspection_mode: str | None = None
paddle_schema_version: str | None = None
summary_schema_version: str | None = None
rule_version: str | None = None
model_metadata: dict = field(default_factory=dict)
geometry_metadata: dict = field(default_factory=dict)
capability_metadata: dict = field(default_factory=dict)
timestamp_source: str | None = None
```

Đặt field mới ở cuối giữ các constructor keyword/positional hiện có.

Geometry-v2 điền fields legacy bắt buộc:

```text
score:
    evidence_support_score

reasons:
    reason_codes

measurements:
    compatibility length/width Mục 24.5

thresholds:
    flat finite numeric thresholds áp cho event

margins:
    flat finite numeric gate margins áp cho event

bbox_frame_xyxy:
    union bbox của accepted primary-frame components, diagnostic only

contour_frame:
    empty float32 shape (0, 1, 2), diagnostic compatibility only

latency_ms:
    primary source-frame pipeline latency

source_frame:
    None; geometry renderer đã quản lý evidence/snapshot riêng

rule_result:
    coarse legacy result
```

Geometry-v2 không gọi legacy snapshot helpers phụ thuộc contour/source frame.

## 28.9 Summary builder

`server.worker._build_summary`:

- giữ fields cũ;
- copy fields geometry khi present;
- dùng `vision_status` làm canonical;
- vẫn tìm snapshot bằng filename cũ;
- không phụ thuộc `vlm_called` để biết đã phân loại;
- thêm counts v2;
- với legacy result, fields v2 có thể absent.

Mỗi item trong `defects`/`normals` phải giữ flat legacy fields:

```text
bar_id, track_id, frame_id, score, reasons, rule_result,
defect_type, vlm_called, length, width, thresholds, margins,
snapshot_key
```

và thêm canonical fields:

```text
paddle_id, track_ids, vision_status, final_reviewed_status,
classification_source, evidence_support_score,
suspected_breakage, possible_breakage_statuses,
review_required, geometry_analysis, snapshot_metadata
```

Không tạo nested `compatibility` object thứ hai. Trước `db.save_result`, summary
phải qua plain-JSON conversion và recursive finite-number assertion.

Không biến `uncertain` thành normal. Nó nằm trong `defects` compatibility bucket
với:

```text
defect_type = null
review_required = true
```

Riêng `event_cardinality_unresolved` là failed geometry job: summary builder giữ
failure/candidate diagnostics nhưng không build `defects`, `normals` hoặc legacy
count fields từ hypotheses. Không trả buckets rỗng như inspection thành công.

## 28.10 Frontend

Frontend phải:

1. cho chọn `geometry_v2`;
2. sau khi chọn ROI, vẽ/điều chỉnh centerline bằng hai handles;
3. preview chain band và trigger strip;
4. validate line trước upload;
5. gửi `geometry`;
6. ưu tiên display `vision_status`;
7. hiển thị tám labels mới;
8. hiển thị badge `Cần xem xét` cho `uncertain`;
9. không dùng `vlm_called` làm điều kiện hiện label;
10. hiển thị angles chỉ khi non-null;
11. hiển thị `—` thay numeric compatibility measurement không hợp lệ.

Centerline default MAY là đường dọc giữa ROI, nhưng người dùng phải xác nhận trước
submit geometry-v2 job.

Correction UI phải giữ toàn bộ bar object khi chuyển giữa defects/normals. Không
được tạo object rút gọn làm mất `vision_status`, evidence, snapshot metadata hoặc
review revision. Canonical effective status:

```text
effective_status =
    final_reviewed_status
    ?? vision_status
    ?? legacy_effective_status

legacy_effective_status =
    "normal"                         # item legacy trong normals
    OR defect_type                    # legacy defect đã classified hợp lệ
    OR "_unclassified"
```

`vlm_called` chỉ là historical compatibility field, không phải human-review flag.
`applyCorrection` MUST NOT set nó thành `true`. `collectCorrections` gửi effective
canonical status cho **mọi** bar, để `broken_center` không bị suy giảm thành legacy
`broken` trước khi server persist/report.

## 28.11 Manual correction

Correction payload giữ field item cũ để giảm API diff:

```json
{
  "expected_review_revision": 3,
  "corrections": [
    {"track_id": 17, "defect_type": "broken_center"}
  ]
}
```

Allowed corrections mới:

```text
normal
bent_left
bent_right
bent_both
broken_left
broken_right
broken_center
```

Legacy `broken` vẫn được chấp nhận cho old jobs/reports.

`expected_review_revision` ở envelope report request, không cần lặp từng item.
Server normalize legacy `broken` theo context; geometry-v2 reviewer phải gửi một
trong bảy physical labels, không gửi coarse `broken` vì sẽ mất location.

Correction không được ghi đè raw geometry result. Report data phải lưu:

```json
{
  "machine_vision_status": "uncertain",
  "final_reviewed_status": "broken_center",
  "correction_source": "human"
}
```

Workflow persistence canonical:

1. load immutable machine summary và verify job `completed`, `count_certified`
   và `report_export_allowed`;
2. resolve effective status cho mọi bar, reject unknown/unresolved `uncertain`;
3. build một `report_data` normalized duy nhất; PDF và Excel dùng cùng object;
4. ghi PDF và Excel thành công theo contracts hiện có;
5. chỉ sau cả hai artifact thành công, `db.save_review_revision(...)` CAS theo
   `job_id + expected_review_revision`, update `result_summary_json` atomically;
6. JSON review append revision immutable gồm reviewer, time UTC, full per-track
   final statuses, PDF/Excel filenames, và source machine summary revision.

`save_review_revision` chỉ cập nhật review metadata/final reviewed status, không
overwrite `vision_status`, evidence, model/config/capability hashes hay machine
counts. CAS conflict trả HTTP 409; service MUST attempt cleanup PDF vừa tạo và log
reconciliation item cho Excel upsert (Excel hiện upsert theo job ID). Không báo
`saved=true` khi persistence revision không thành công. Đây là saga có compensating
audit, không tuyên bố atomic transaction xuyên SQLite/PDF/XLSX filesystem.

## 28.12 Report compatibility

Để tối thiểu thay đổi PDF/Excel:

```text
broken_left
broken_right
broken_center
legacy broken
    -> report group "broken"
```

Các nhóm cũ:

```text
bent_left
bent_right
bent_both
broken
```

được giữ.

`uncertain` không được export như defect type cuối. Trước export:

```text
mọi uncertain phải có human correction
```

Nếu còn unresolved:

```text
ReportError("bar track <id> is still unclassified (...)")
```

Đây tương thích với behavior `_unclassified` hiện hữu.

`build_report_data` phải hiểu canonical corrections/effective statuses trước khi
map groups: `broken_left/right/center -> broken`, còn legacy Excel/PDF columns và
layout giữ nguyên. Report endpoint block ngay nếu `count_certified=false` hoặc
`report_export_allowed=false`; không viết total count từ ambiguity hypothesis.

Một report v2 SHOULD thêm phụ lục chi tiết location cho broken events, nhưng đây là
enhancement không bắt buộc cho compatibility release.

## 28.13 R2/storage

Giữ object-key layout:

```text
results/{job_id}/snapshots/defects/{filename}
results/{job_id}/snapshots/normals/{filename}
```

`uncertain` đi vào `snapshots/defects` để UI cũ vẫn tải evidence.

Không đổi upload/delete lifecycle.

## 28.14 Packaging

Linux PyInstaller phải thêm:

```python
("config/geometry_v2.json", "config")
```

Vì `.gitignore` hiện ignore `*.spec`, phải thêm explicit exception:

```text
!DragConveyor_linux.spec
```

để Linux spec mới thực sự được version-control. Linux spec bổ sung
`server/excel_log.py`, `config/geometry_v2.json`, `drag_conveyor/geometry_v2/*`,
PyAV runtime imports/data cần thiết, cùng hidden import `openpyxl`/`av` sau smoke
test thực tế.

Windows data directives nằm trong `gui/__main__.py`, không chỉ build batch. Bổ
sung geometry config/module và PyAV package tại đây; `.github/workflows/build-windows.yml`
phải assert exact staged files, bao gồm geometry config, model, server Excel
module, static assets và không có secrets. `pyproject.toml`/`uv.lock` pin PyAV.

Windows/Nuitka build phải kiểm chứng các artifact sau tồn tại trong staging:

```text
config/base_profile.json
config/geometry_v2.json
weights/model_imgsz_640/best.onnx
server/static/*
```

Build smoke test phải load geometry config và verify model hash từ packaged
runtime, không chỉ từ source tree.

---

# 29. Kiến trúc source code đề xuất

## 29.1 Module mới

```text
drag_conveyor/geometry_v2/
├── __init__.py
├── config.py
├── types.py
├── frame_source.py
├── coordinates.py
├── observations.py
├── tracking.py
├── analyzers.py
├── decision.py
├── snapshots.py
└── pipeline.py
```

Trách nhiệm:

- `config.py`: strict schema, resolve, hash;
- `types.py`: enums/dataclasses immutable;
- `frame_source.py`: PTS/CFR/EOF-vs-error decoder contract;
- `coordinates.py`: `(s, q)`, FOV intersection, robust projections;
- `observations.py`: component extraction, dedup, per-frame pairing;
- `tracking.py`: Kalman, order-preserving DP, tracklet fusion;
- `analyzers.py`: center topology, side integrity, angle;
- `decision.py`: truth tables, reason codes, compatibility mapping;
- `snapshots.py`: evidence selection/render;
- `pipeline.py`: bounded streaming + offline fusion orchestration.

Không tạo abstraction plugin/framework tổng quát. Module chỉ phục vụ geometry-v2.

## 29.2 Reuse trực tiếp

Reuse:

- `OnnxRuntimeEngine`;
- `preprocess_roi`;
- `postprocess_segmentation`/`Detection.mask_roi`;
- underlying video/container dependencies; geometry-v2 không gọi
  `open_video_source` trực tiếp nếu backend đó chưa thỏa `FrameSource`;
- path/run-id helpers;
- snapshot/R2/DB worker lifecycle;
- API/report infrastructure.

Reuse có điều kiện:

- model profile inference parameters;
- raw frame decode;
- source snapshot directories.

## 29.3 Không reuse

Không dùng trong geometry-v2:

- `CentroidTracker`;
- old `TriggerEngine`;
- contour length/width `RuleEngine` làm final classifier;
- auto-baseline/average-ratio classifier;
- local aspect/area detection filter;
- VLM inspector;
- slow-motion voting path.

Các phần này không bị xóa; legacy modes vẫn dùng.

## 29.4 Existing files cần sửa tối thiểu

```text
drag_conveyor/inspection_modes/average_ratio.py
drag_conveyor/inspection_modes/__init__.py
drag_conveyor/inference/_core.py
drag_conveyor/inference/yolo_seg_postprocess.py
drag_conveyor/app/batch.py
server/main.py
server/worker.py
server/db.py
server/report.py
server/excel_log.py
server/static/app.js
server/static/index.html
server/static/styles.css
DragConveyor_linux.spec
gui/__main__.py
.github/workflows/build-windows.yml
.gitignore
pyproject.toml
uv.lock
```

Thêm:

```text
config/geometry_v2.json
drag_conveyor/geometry_v2/*
tests/geometry_v2/*
```

Không cần:

- đổi jobs table;
- đổi R2 key structure;
- đổi filename snapshot;
- đổi hai legacy modes;
- retrain model ở release đầu;
- gọi VLM.

Hai inference-file edits chỉ thêm raw float bbox, exact rasterized crop bbox và
`model_output_row_index`, rồi truyền chúng trước representative overwrite; không
đổi legacy bbox/contour values hoặc decode/NMS behavior.

## 29.5 Impact analysis đã kiểm tra

Theo GitNexus index hiện tại:

- `run_batch_inspection`: risk `LOW`, một direct caller chính là
  `_process_job`;
- `_build_summary`: risk `LOW`, phạm vi trực tiếp nhỏ;
- `build_report_data`: risk `MEDIUM`, có nhiều caller/tests và là vùng cần
  regression kỹ;
- không có target trong kế hoạch này trả `HIGH` hoặc `CRITICAL`.

Khi implementation thực sự bắt đầu, phải chạy lại `impact` trên từng symbol ngay
trước edit vì index/source có thể đã đổi.

## 29.6 Thứ tự implementation tối thiểu rủi ro

1. types/config/coordinates thuần + unit tests;
2. components/dedup/observations + synthetic-mask tests;
3. tracking/fusion + sequence tests;
4. analyzers/decision + truth-table tests;
5. geometry pipeline + model regression fixtures;
6. early dispatch trong batch;
7. API/worker persistence và slowmo bypass;
8. summary/frontend/report compatibility;
9. packaging;
10. shadow validation.

Không sửa report/frontend trước khi canonical result contract đã được test.

## 29.7 Definition of a source-level seam

Legacy/new boundary phải có test:

```text
inspection_mode != geometry_v2
    -> legacy function path được gọi
    -> geometry modules không thay đổi result

inspection_mode == geometry_v2
    -> geometry pipeline được gọi
    -> legacy tracker/rules/VLM không được gọi
```

---

# 30. Non-functional requirements

## 30.1 Determinism

Với cùng:

- exact video bytes;
- ROI/geometry input;
- model bytes/hash;
- resolved config;
- Python/OpenCV/ONNX Runtime versions;
- execution provider;
- thread settings;

hệ thống phải tạo cùng:

- `count_certified`, event-count bounds và, khi certified, số physical events;
- `paddle_id`/`track_ids`;
- `vision_status`;
- reason codes;
- evidence frame IDs;
- snapshot selection.

Floating metrics phải bằng trong tolerance test đã định:

```text
coordinate tolerance = 1e-6 for pure geometry unit tests
angle tolerance = 1e-6 degree for pure formula unit tests
runtime inference golden tolerance = provider-specific
```

Không dựa vào:

- iteration order của `set`/unordered dict;
- OpenCV contour return order;
- filesystem order;
- wall-clock time.

Bootstrap không dùng RANSAC/random sampling; robust fit là deterministic MAD/TLS.

## 30.2 Provider reproducibility

CPU và GPU execution provider MAY cho mask pixels khác sát threshold.

Mỗi deployment phải pin:

```text
onnxruntime version
execution provider
provider order
OpenCV version
NumPy version
```

Acceptance golden được chạy riêng trên provider production. Không được dùng kết
quả CPU để tuyên bố GPU đã pass nếu chưa test.

Mỗi deployment profile trước production phải pin và approve:

```text
inference/postprocess/end-to-end p50 và p95
real-time factor hoặc offline throughput target
peak RSS
maximum job duration, resolution và video duration
provider/session thread options
```

Benchmark phải dùng representative original videos/camera domain, gồm I/O và
snapshot/report policy nếu đó là SLA E2E. Không suy SLA từ benchmark Mục 36.10.

## 30.3 Complexity

Cho:

```text
A = ROI pixel count
D = detections accepted for mask reconstruction in a frame
C = accepted components in a frame
N = active tracks
M = observations in current frame
E = final physical events
K = top-K evidence per type
```

Target:

```text
mask reconstruction/components: O(D * A) per frame
dedup candidate indexing:       O(C log C + P)
online association:         O(N * M)
evidence memory:            O(N * K)
active scalar metadata:     O(N * maximum_metadata_observations_per_track)
lossless scalar spool:      O(total eligible observations), byte-quota bounded
final result metadata:      O(E)
```

Fusion phải index theo crossing-time window/order, không tạo all-pairs
`O(E^2)` graph cho video dài.

`P` là số candidate-overlap pairs sau spatial/anchor indexing. Không so sánh mọi
full-size mask pair. Intersection chỉ chạy trên overlap của cropped/RLE component
bounds; worst-case vẫn được chặn bởi component/resource limits.

Không giữ:

- toàn bộ decoded frames;
- toàn bộ full-size masks;
- toàn bộ crops cho mọi observation.

## 30.4 Memory boundedness

Sau warm-up, tăng thời lượng video 10 lần nhưng giữ mật độ paddle tương tự:

- peak memory cho active processing không được tăng tuyến tính theo frame count;
- phần tăng hợp lệ là scalar final results `O(E)` và lossless scalar/artifact
  spool đã ghi disk, bị byte quota chặn;
- top-K buffers giữ bound cấu hình.

Long-run acceptance:

```text
video >= 60 minutes
no unbounded active-track growth
no unbounded frame/mask retention
active processing memory drift <= 10% after warm-up
```

Ngưỡng absolute RAM phải được đặt theo hardware deployment; spec không bịa một con
số hardware-independent.

Binary evidence mask được lưu dưới dạng bbox-local deterministic RLE, không giữ
full-ROI array. Source crop được JPEG-encode rồi spool vào job temp directory.
In-memory artifact cache dùng byte-accounted LRU. Scalar evidence được spool
lossless và không phụ thuộc artifact còn trong cache.

## 30.5 Resource limits

Khi vượt:

```text
maximum_roi_pixels
maximum_instances_for_mask_reconstruction_per_frame
maximum_components_per_frame
maximum_active_tracks
maximum_events_per_job
maximum_identity_conflict_tracklets
maximum_transient_mask_bytes
maximum_in_memory_evidence_bytes
maximum_spooled_evidence_bytes
maximum_spooled_evidence_bytes_per_event
```

không được silently truncate theo score.

Trước reconstruct masks:

```text
retained_mask_bytes =
    accepted_detection_count * roi_height * roi_width

scratch_bytes =
    input_size * input_size * sizeof(float32)
  + roi_height * roi_width
      * (
          sizeof(float32)   # mask_roi_prob
        + sizeof(uint8)     # threshold mask
        + sizeof(uint8)     # bbox crop buffer
        )
  + actual_proto_output_bytes
  + actual_detection_output_bytes

estimated_transient_mask_bytes =
    ceil(1.25 * (retained_mask_bytes + scratch_bytes))
```

Formula là conservative plan cho current NumPy decoder; implementation phải update
estimate nếu live buffers/dtypes đổi và đo peak RSS trong benchmark. Gate
post-conf/class/NMS instance count và estimated bytes trước reconstruct toàn bộ
masks; không nhầm với 300 raw model output rows và không đợi tới component stage.

Policy:

```text
job failure
failure_reason = "geometry_resource_limit_exceeded"
```

Log actual count/limit, không log raw frame bytes.

## 30.6 Runtime errors

Các lỗi sau fail job:

- model inference exception;
- non-finite model output;
- frame geometry đổi;
- corrupt decode trước declared end;
- config/model hash mismatch;
- result cannot serialize;
- strict snapshot failure;
- resource limit.
- event cardinality unresolved.

Không được tiếp tục và tạo verdict từ một phần video mà không ghi rõ.

## 30.7 Event uncertainty, không phải job failure

Các trường hợp sau là per-event `uncertain`:

- single-side-only;
- insufficient independent bins;
- identity association cạnh tranh;
- FOV cắt;
- center/side/angle conflict;
- temporal center evidence chưa validated;
- reference side length chưa đủ;
- angle không ổn định.

`event_cardinality_unresolved` không phải per-event `uncertain`: nó không biết
bao nhiêu physical events để gán result, nên fail job theo Mục 17.5/30.6.

Job vẫn success nếu có ít nhất một reportable event và pipeline hoàn tất.

## 30.8 Logging/diagnostics

Job-level counters tối thiểu:

```text
decoded_original_frames
inference_calls
raw_detections
raw_components
small_components_rejected
duplicate_components_removed
ambiguous_observations
online_tracks_created
tracks_rejected
tracks_fused
physical_events
uncertain_events
snapshot_failures
```

Per-event diagnostic log:

```text
paddle_id
track_ids
crossing_time
observability_grade
center/left/right states
vision_status
primary_reason
decision_confidence
```

Production default không log images/masks/base64.

## 30.9 Auditability

Từ một result phải truy được:

- model hash;
- config/rule/schema versions;
- ROI/centerline;
- original frame IDs/timestamps;
- component/observation/track aliases;
- evidence bins;
- threshold values;
- decision branch;
- human correction nếu có.

Không chỉ lưu final label.

## 30.10 Security

- remote client không chọn model/config filesystem path;
- validate all numeric inputs finite/ranged;
- run/job ID vẫn cấm path separators;
- snapshot path được tạo từ integer IDs, không từ free text;
- debug artifacts mặc định tắt;
- không bundle secrets `.env`/`app_settings.json`;
- model hash mismatch fail closed.

## 30.11 Backward compatibility

Regression invariant:

```text
for inspection_mode in {"auto_baseline", "average_ratio"}:
    result semantics, VLM policy, preprocessing and snapshots remain unchanged
```

Geometry-v2 fields là optional đối với old results. Frontend/report phải đọc được
cả old và new summary.

---

# 31. Test specification

## 31.1 Test pyramid

```text
Pure geometry/math unit tests
        ↓
Synthetic mask/sequence tests
        ↓
Pinned-model regression tests
        ↓
Raw-video event integration tests
        ↓
API/worker/report/frontend contract tests
        ↓
Packaged-runtime smoke tests
```

Mỗi bug production phải có regression fixture nhỏ nhất có thể.

## 31.2 Coordinate tests

Bắt buộc:

- vertical centerline: `(s, q)` đúng;
- rolled centerline: projection đúng;
- reversed endpoints được canonicalize;
- `q < 0` là left, `q > 0` là right;
- ROI translation không đổi local geometry result;
- FOV intersection đúng ở bốn biên;
- exact `I={s | P0+s*d thuộc ROI}` cho rolled/off-center line;
- pixel-center và half-open bbox/raster boundaries;
- type-7 quantile/MAD tại interpolation boundaries;
- trigger strip polygon clip đúng;
- invalid zero-length line bị reject;
- line span/roll/range boundaries;
- positive/negative motion normalization.

## 31.3 Component tests

Synthetic masks:

1. one connected whole component;
2. two large left/right components;
3. one large + several tiny noise blobs;
4. component touching ROI boundary;
5. component touching model bbox boundary;
6. multiple anchors in one component;
7. representative contour chỉ là left nhưng `mask_roi` có left+right;
8. duplicate masks có IoU thấp nhưng IoS cao;
9. two legitimate sides gần anchor không bị dedup;
10. stable IDs khi input detection order bị shuffle.

Assertions:

- topology morphology không nối components;
- area threshold đúng equality;
- anchor dùng near-chain pixels, không centroid;
- duplicate không tạo independent vote.
- IoS containment variants disagreement -> geometry `UNKNOWN`, không score-winner
  verdict.

## 31.4 Observation tests

Cases:

```text
LEFT + RIGHT unique                    -> DISCONNECTED_BOTH/bridge-dependent
LEFT only                              -> LEFT_ONLY
RIGHT only                             -> RIGHT_ONLY
spanning connected mask                -> CONNECTED_WHOLE after bridge pass
two candidate paddles ordered          -> two observations
ambiguous cross-pairing                 -> AMBIGUOUS
multi-anchor component                  -> MULTI_PADDLE_MERGED
```

Shuffling components không đổi observations/IDs.

## 31.5 Tracking tests

Sequence fixtures:

1. `LEFT, RIGHT, LEFT, RIGHT` cùng trajectory -> một track;
2. representative centroids cách nhau >80 px nhưng anchors gần -> một track;
3. one/two missed frames trong `0.35 sec` -> giữ track;
4. gap vượt gate -> split tracklets;
5. two adjacent paddles -> không swap identity;
6. observation order shuffle -> cùng association;
7. reverse jitter trong tolerance -> không tạo event mới;
8. true reverse/mismatched trajectory -> reject match;
9. best/second-best gần -> ambiguity flag;
10. tentative one-hit noise -> rejected.
11. predict/commit đúng một lần mỗi original frame kể cả MISS;
12. Joseph covariance update vẫn symmetric/finite;
13. VFR positive 66.667 ms gap advance exact `dt`, không gây timestamp failure.

## 31.6 Fusion tests

- complementary left/right tracklets có crossing/velocity/trajectory phù hợp ->
  merge;
- same-side split sau short gap -> merge nếu unique;
- two neighboring events trong candidate window -> không cross-merge;
- competitor trong ambiguity margin -> không definitive merge;
- interval chưa đủ 5 events -> không dùng untrusted interval;
- configured interval được ưu tiên;
- fusion không đảo event order;
- final `paddle_id` deterministic;
- evidence duplicate sau merge chỉ vote một lần.
- `possible_event_count_min/max` exact khi conflict; diagnostic k-best cap không
  đổi bounds.
- group vượt identity DP limit -> resource failure, không certify count.

## 31.7 Center-topology tests

1. true bridge qua ba gates, coverage/thickness đủ -> `PRESENT`;
2. 1-pixel false bridge -> không `PRESENT`;
3. morphology closing có thể nối nhưng topology raw rời -> vẫn rời;
4. two components, corridor observable -> `ABSENT`;
5. bboxes crop không bao center -> `UNKNOWN`;
6. corridor/ROI truncated -> `UNKNOWN`;
7. two connected bins -> `INTACT`;
8. one connected bin only -> `UNKNOWN`;
9. two disconnected bins -> `BROKEN_TOPOLOGICAL`;
10. connected≥2 + disconnected≥2 -> `CONFLICT`;
11. connected=1 + broken strong -> `CONFLICT`;
12. temporal alternating evidence provisional -> final `uncertain`;
13. cùng evidence với validated capability -> `BROKEN_TEMPORAL`;
14. temporal sides có competitor -> `UNKNOWN`.
15. independent instance masks không được OR để tạo bridge;
16. bridge run có gap theo s không pass span bằng `max-min`;
17. ABSENT đòi every required raster cell được bbox-crop coverage chứng minh.

## 31.8 Side-integrity tests

- intact full coverage -> `VALID`;
- internal gap exactly `0.08` -> valid side of boundary;
- gap in gray zone -> `UNKNOWN`;
- gap exactly/above `0.15` -> broken criterion;
- length exactly `0.72` -> broken criterion;
- length gray zone -> unknown;
- length exactly `1 - expected_extent_tolerance_ratio` -> valid criterion;
- reference absent -> no shortness conclusion;
- fewer than 5 reference paddles -> no online reference;
- P90 reference robust to a short outlier;
- reference at anchor farther than `0.03H` -> not comparable;
- expected endpoint outside FOV -> unknown;
- bbox-crop truncation -> weak/unknown, not broken;
- two broken bins/60% support -> broken;
- two valid and two broken -> conflict;
- raw absence not counted in eligible denominator.
- internal-gap broken không cần reference; shortness broken bắt buộc validated
  reference + FOV.

## 31.9 Angle formula tests

Pure vectors:

| `theta_left` | `theta_right` | Expected geometry |
| ---: | ---: | --- |
| `0°` | `0°` | normal |
| `-6°` | `+6°` | global `\`, not 12° sum |
| `+6°` | `-6°` | global `/` |
| `+6°` | `+6°` | center kink about `12°` |
| `-6°` | `-6°` | opposite kink direction, same magnitude |
| `+10°` | `+2°` | `bent_left` |
| `+2°` | `-11°` | `bent_right` |
| `+10°` | `-12°` | `bent_both` |

Additional:

- vector sign canonicalization;
- endpoint robust to one extreme pixel;
- global tilt uses outer endpoints;
- clamp before `acos`;
- threshold/guard gray zone -> `uncertain`, không bent/normal;
- degenerate side orientation/global endpoint ordering -> reject sample;
- three independent samples required;
- MAD exactly `1.5°` accepted;
- left/right from different frames cannot form sample;
- broken/uncertain yields all angle fields null.

## 31.10 Decision truth-table tests

Mỗi row Mục 23.3 phải có unit test.

Invariant tests:

- single-side-only current model always uncertain;
- `broken_both` never emitted;
- center broken + unresolved side is uncertain;
- both side broken is uncertain;
- insufficient angle never normal;
- VLM never called;
- diagnostic support score never changes a hard-rule verdict;
- every final status belongs canonical enum;
- every uncertain has primary reason;
- every broken has `suspected_breakage=true`;
- every normal has valid angle evidence and no positive break evidence.

## 31.11 Serialization/config tests

- unknown config key rejected;
- non-finite number rejected;
- invalid enum rejected;
- every cross-field boundary;
- canonical config hash stable;
- JSON contains no NaN/Infinity;
- old summary without geometry fields still parses;
- new summary compatibility mapping exact;
- schema/rule/model/config versions present.

## 31.12 Pinned current-model regression

Artifact:

```text
weights/model_imgsz_640/best.onnx
SHA-256:
ef05955f43c8db6d2ff76b72fb65806e69afe525e85d8486eeb2dfb7566dcd65
```

Fixtures là **toàn bộ** `data/example/*.jpg` hiện có, whole image as ROI, theo
đúng deterministic 640 profile/runtime đã pin:

```text
vid1_frame_0001_normal.jpg              -> 1 detection
vid4_frame_0002_normal.jpg              -> 1
vid4_frame_0019_broken.jpg              -> 2
vid4_frame_0020_broken.jpg              -> 1
vid4_frame_0023_broken.jpg              -> 1
vid4_frame_0025_broken.jpg              -> 1
vid4_frame_0026_bent_both.jpg           -> 1
vid4_frame_0028_broken.jpg              -> 1
vid4_frame_0029_broken.jpg              -> 1
vid4_frame_0031_bent_right.jpg          -> 0
vid4_frame_0067_bent_right.jpg          -> 0
vid4_frame_0071_bent_left.jpg           -> 0
vid5_frame_0001_normal.jpg              -> 1
vid5_frame_0005_bent_right.jpg          -> 1
vid5_frame_0015_bent_left.jpg           -> 1
vid5_frame_0019_bent_both.jpg           -> 1
```

Golden regression phải pin bytes model, CPU provider/order/options, Python 3.12.3,
ONNX Runtime 1.27.0, OpenCV 4.13.0 và NumPy 2.4.6; provider khác có golden riêng.
Verify output row provenance, raw float bbox và exact floor/clipped crop bbox ở
mỗi accepted instance. Với cùng raw output/mask threshold, đổi representative
`largest`/`union` chỉ được đổi legacy contour/bbox/moments; `mask_roi` phải
bit-identical. Significant component logic dùng exact threshold/config và phải
cho golden 3 zero, 12 one, 1 two detections; broken sample components theo Mục
36.4. Lặp 20 inference CPU và verify hash raw/postprocess identical trong exact
environment chỉ là determinism spot check, không là guarantee GPU/deployment.

Test trên production provider phải xác nhận ít nhất các morphology đã quan sát:

- có sample phát nhiều detections cho một physical candidate;
- có sample một detection chứa nhiều significant components;
- có sample chỉ một significant component;
- postprocessor `largest` không làm component extractor bỏ component còn trong
  `mask_roi`;
- dedup IoS xử lý overlap containment.

Không dùng filename label làm ground truth event-level. Đây là model-contract
regression, không phải benchmark accuracy.

## 31.13 Raw-video mechanics test

`data/raw_data/vid_1.mp4` dùng để test:

- decode/provenance;
- canonical PyAV PTS/VFR invariants Mục 10.1;
- smoke/invariants trên nhiều detections/components;
- result serialization/snapshot naming.

Không dùng video 31 giây này để chứng minh tracker/fusion correctness, bounded
memory 60 phút hoặc broken/bent accuracy vì không có event ground truth được xác
nhận. Các claims đó cần annotated raw sequences và synthetic/looped long-run test
riêng.

## 31.14 API/worker integration

Tests:

- create geometry job with valid geometry;
- geometry job missing geometry -> 422;
- unknown geometry key -> 422;
- legacy create payload unchanged;
- persistence nested trong `roi_config_json`;
- worker strips exact six ROI keys;
- geometry mode receives original video path;
- legacy mode vẫn đi qua slowdown policy hiện tại;
- geometry-v2 VLM call count zero;
- model hash mismatch fails job;
- result saved before cleanup;
- R2 snapshot keys unchanged.
- capability/runtime/domain mismatch -> shadow/uncertain policy, never definitive;
- cardinality unresolved -> completed diagnostics but report export block/no fake count.

## 31.15 Frontend/report tests

- frontend reads `vision_status` dù `vlm_called=false`;
- bảy physical labels và `uncertain` render đúng;
- uncertain badge/correction flow;
- manual correction retains machine result;
- moving normal/defect keeps the complete object and does not mutate `vlm_called`;
- collectCorrections preserves `broken_center` and sends all effective statuses;
- report review CAS conflict -> 409/no saved success; PDF/Excel reconciliation log;
- persisted review revision is append-only and leaves machine verdict/hashes intact;
- report maps three broken locations to group `broken`;
- unresolved uncertain blocks export;
- count_certified=false blocks export;
- old `broken` report still works;
- snapshot temporal composite renders;
- invalid/null angles display `—`;
- old result JSON remains viewable.

## 31.16 Legacy regression

Trước và sau implementation:

```text
all existing tests pass
```

Thêm spies:

```text
auto_baseline:
    geometry pipeline not called

average_ratio:
    geometry pipeline not called

geometry_v2:
    CentroidTracker not called
    legacy RuleEngine not called
    VlmInspector not called
```

## 31.17 Packaging tests

Trên artifact cài đặt, không phải source checkout:

- app starts;
- `geometry_v2.json` load;
- model hash verify;
- one short geometry job runs;
- frontend loads centerline UI;
- report export after correction;
- secrets not bundled.

## 31.18 Property/metamorphic tests

- shuffle detection/component input order -> same result;
- add duplicate detection -> same physical event/evidence count;
- add tiny noise components -> same verdict;
- translate ROI/frame consistently -> same normalized geometry;
- small allowed centerline roll with transformed mask -> same relative angle;
- duplicate frames marked synthetic -> no extra votes;
- extend video with empty tail -> same existing events;
- process same input twice -> same IDs/status/evidence selection.

## 31.19 Fuzz tests

Fuzz:

- empty masks;
- one-pixel masks;
- extreme but valid ROI sizes;
- masks at every boundary;
- large component counts near limit;
- zero/very high model scores;
- near-equal DP costs;
- timestamps close to bin boundaries;
- threshold equality.

Expected:

- no crash;
- no NaN;
- deterministic;
- fail/uncertain theo contract.

---

# 32. Ground-truth dataset và acceptance gates

## 32.1 Vì sao dữ liệu hiện tại chưa đủ

Các ảnh `data/example/*` là frame/crop rời. Chúng không cung cấp:

- full temporal sequence cho từng paddle;
- physical paddle identity;
- event entry/exit;
- confirmed defect location;
- true mechanical angle;
- camera/FOV variation đủ rộng.

`vid_1.mp4` không có ground-truth defect annotation.

Do đó có thể kiểm chứng mechanics/model emission, nhưng chưa thể trung thực tuyên
bố production precision/recall cho bài toán bảy physical classes với
`uncertain` là abstention/review outcome.

## 32.2 Annotation unit

Ground truth chính là physical paddle event, không phải từng detection.

Mỗi event:

```json
{
  "video_id": "lineA_20260724_001",
  "physical_paddle_id": 27,
  "entry_frame": 812,
  "entry_timestamp_sec": 27.093333333,
  "trigger_crossing_frame": 846,
  "trigger_crossing_timestamp_sec": 28.226666667,
  "exit_frame": 891,
  "exit_timestamp_sec": 29.726666667,
  "status": "broken_center",
  "left_angle_deg": null,
  "right_angle_deg": null,
  "visible_left": true,
  "visible_right": true,
  "center_visible": true,
  "occlusion_flags": [],
  "annotator_ids": ["a1", "a2"],
  "adjudicated": true
}
```

Timestamp annotation phải lấy từ cùng canonical decoder PTS/provenance table đã
pin với video bytes. Với VFR, evaluator MUST NOT suy timestamp từ frame index,
`r_frame_rate` hoặc average FPS. Nếu annotation UI chỉ lưu frame ID, dataset phải
kèm immutable frame-ID → canonical-PTS table/hash để evaluator lookup.

## 32.3 Required scenarios

Dataset phải cố ý chứa:

- normal;
- từng bent label;
- từng broken location;
- center break với same-frame two components;
- center break luân phiên left/right;
- center break chỉ emit một side;
- single-side-only non-center causes/dropout;
- partial side break;
- nearly full side break;
- complete side loss;
- duplicate instances;
- merged neighboring paddles;
- missed frames;
- speed slow/nominal/fast;
- centerline roll/camera perspective trong operating range;
- ROI/FOV truncation;
- lighting, blur, dirt, shadow;
- start/end-of-video partial events;
- multiple conveyor runs/cameras/days.

## 32.4 Split policy

Split theo:

```text
camera + conveyor run + recording session
```

Không split các frames cùng video/event qua train/validation/test.

Model retraining và rule tuning không được nhìn final held-out test set.

## 32.5 Annotation quality

- tối thiểu hai annotators độc lập cho broken location và bent side;
- disagreement được adjudicate;
- angle ground truth dùng jig/calibrated measurement nếu dùng để chấm độ;
- ambiguous physical cases có annotation riêng, không ép vào bảy physical labels;
- lưu annotation version/history.

## 32.6 Event matching

Predicted event match ground truth bằng one-to-one crossing-time/order matching.

Một prediction chỉ được match một ground-truth paddle và ngược lại.

Events ground-truth đã ở trong trigger khi video bắt đầu hoặc chưa ra khỏi trigger
khi video kết thúc được chấm riêng là partial-boundary handling; không đưa vào
core reportable-event recall denominator.

Eligibility:

```text
abs(predicted_crossing_time - ground_truth_crossing_time)
    <= event_match_gate_sec

event_match_gate_sec =
    min(0.12 sec, 0.35 * local_ground_truth_paddle_interval)
```

Nếu local interval unavailable, dùng `0.12 sec`.

Giải order-preserving dynamic programming với objective:

1. maximize matched pairs;
2. minimize tổng absolute crossing-time error;
3. lexicographic tuple `(ground_truth_id, predicted_paddle_id)`.

Unmatched ground-truth event là miss/FN. Unmatched prediction là FP; prediction
thừa trong gate của một already-matched event được report thêm là duplicate.
Classification metrics chỉ tính trên matched pairs.

Metrics tracking:

```text
event precision
event recall
duplicate-event rate
missed-event rate
order/identity error rate
```

Không chấm bằng frame-level detection accuracy thay cho paddle-level accuracy.

## 32.7 Classification metrics

Report:

- per-class precision/recall/F1;
- macro and micro;
- confusion matrix;
- uncertain rate;
- selective accuracy trên definitive results;
- coverage = definitive / all events;
- safety alert recall:

```text
ground-truth broken counted captured if prediction is:
    correct broken_* OR uncertain with suspected_breakage=true
```

- dangerous false-normal rate cho mỗi broken class.
- definitive exact-location recall/coverage cho từng broken class;
- break-alert precision và false-alert rate trên ground-truth normal;
- normal recall;
- human-review load.

Denominator/abstention semantics bắt buộc:

- recall của ground-truth physical class: denominator là tất cả matched GT events
  của class; prediction `uncertain` là FN cho class đó, không phải FP của một
  physical predicted class;
- precision của một physical predicted class: denominator là predictions
  definitive của chính class; wrong-location là FP của predicted class và FN của
  GT class;
- selective exact-label precision/accuracy chỉ xét predictions definitive;
  `uncertain` bị loại khỏi numerator/denominator và coverage báo riêng;
- normal recall coi `uncertain` hoặc bất kỳ `broken_*`/`bent_*` là non-normal;
- safety alert predicate là **bất kỳ** `broken_*` hoặc `uncertain` có
  `suspected_breakage=true`; exact location không phải điều kiện của safety alert
  vì đã có metric exact-location riêng.

Go/no-go statistical bounds dùng one-sided 95% Clopper–Pearson intervals:

- metric cần tối thiểu: lower confidence bound phải đạt threshold;
- error/FPR cần tối đa: upper confidence bound phải không vượt threshold;
- không pass chỉ bằng point estimate.

## 32.8 Bootstrap production acceptance

Trên held-out production-like test set:

```text
event precision >= 0.99
event recall >= 0.99
duplicate-event rate <= 0.005
identity/order error rate <= 0.005

selective exact-label precision >= 0.98
broken safety-alert recall >= 0.99
definitive exact-location recall per broken class >= 0.90
definitive coverage per broken class >= 0.90
break-alert precision >= 0.95
false break-alert rate on ground-truth normal <= 0.02
dangerous false-normal rate on broken events <= 0.01
normal precision >= 0.98
normal recall >= 0.95

single-side-only definitive localization violations = 0
synthetic-frame vote violations = 0
non-deterministic status/ID violations = 0
```

Coverage target:

```text
overall uncertain rate <= 0.10
```

nhưng precision/safety có ưu tiên hơn việc giảm uncertain.

Nếu không đạt coverage nhưng safety/precision đạt, system có thể pilot với human
review; không được hạ hard safety rules chỉ để giảm uncertain.

## 32.9 Per-class sample floor

Held-out test SHOULD có tối thiểu:

```text
300 adjudicated events cho mỗi physical class
300 single-side-only/ambiguous events
```

Mốc 300 cũng cho phép zero-error one-sided 95% upper bound xấp xỉ 1% cho một error
rate. Nếu có lỗi hoặc denominator/precision subset nhỏ hơn, bound thực tế quyết
định pass; sample floor không tự bảo đảm pass.

Nếu lớp hiếm chưa đạt, giữ feature/policy liên quan ở provisional/shadow.

## 32.10 Same-frame center-topology enablement gate

Để bật:

```text
production_enabled.same_frame_center_topology = true
```

held-out set phải có tối thiểu:

```text
300 adjudicated same-frame broken-center events
300 intact/side-break hard-negative events có split/noisy masks
```

Yêu cầu one-sided 95% bounds:

```text
precision lower bound >= 0.98
recall lower bound >= 0.95
definitive coverage lower bound >= 0.90
hard-negative false-positive upper bound <= 0.01
```

Fixtures VLM crops hiện tại không thay thế gate này.

Promotion phải dùng protocol immutable capability record/deployment binding Mục 39,
không chỉ sửa boolean trong config.

## 32.11 Temporal broken-center enablement gate

Để đổi cả hai fields:

```text
validation.temporal_complementary_emission:
    provisional -> validated

production_enabled.temporal_center_break:
    false -> true
```

test set phải có tối thiểu:

```text
300 broken-center events có alternating/temporal-only emission
300 intact events có model dropout/fragment switching tương tự
```

Yêu cầu:

```text
temporal broken-center precision lower bound >= 0.98
temporal broken-center recall lower bound >= 0.95
intact -> definitive broken_center false-positive upper bound <= 0.01
```

Cho tới khi pass, temporal evidence tạo:

```text
vision_status = uncertain
suspected_breakage = true
possible_breakage_statuses includes broken_center
```

## 32.12 Side-break enablement

Shortness-based side localization chỉ production khi:

- expected extent source đã validated;
- FOV checks pass;
- ít nhất 300 adjudicated events mỗi side-break class;
- dùng one-sided 95% confidence bounds bên dưới, không point estimate.

Complete-side absence tự nó không tạo location rule trong spec này, kể cả khi
shortness-based side gate pass.

Side-break gate riêng cho mỗi side:

```text
precision lower bound >= 0.98
recall lower bound >= 0.90
definitive coverage lower bound >= 0.90
intact false-positive upper bound <= 0.01
```

Gate này đồng thời phải validate `side_geometry_validity` cho candidate `VALID`
trên intact/hard-negative set trước khi một verdict `normal`, `bent_*` hoặc
`broken_center` được dựa vào side `VALID`.

## 32.13 Angle acceptance

Candidate bootstrap gates:

```text
median absolute error <= 2.0 degrees
P95 absolute error <= 4.0 degrees
bent per-class precision lower bound >= 0.95
bent per-class recall lower bound >= 0.95
normal-vs-bent false-positive upper bound <= 0.02
```

Các con số phải được chủ cơ khí phê duyệt theo tolerance thật. Nếu yêu cầu vật lý
khắt khe hơn hoặc perspective lớn, phải calibration camera trước production.

Promotion `angle_classification` chỉ hợp lệ khi same-frame center topology và
side-geometry validity đã là validated/enabled trong cùng compatible deployment
profile; record validator phải reject combination angle=true với dependency=false.

## 32.14 Threshold tuning

Tuning:

- dùng train/calibration split;
- tối ưu constrained objective ưu tiên false-normal và wrong-location;
- freeze config/rule version;
- đánh giá đúng một lần trên held-out test;
- mọi retune tạo config hash/version mới.

Không tuning bằng chính các sample được dùng trong acceptance report.

---

# 33. Hợp đồng cho model retrain trong tương lai

## 33.1 Mục tiêu retrain

Model mới không cần làm downstream schema đổi. Mục tiêu:

- tăng recall cả hai side;
- giảm side switching/dropout;
- giảm duplicate instances;
- giảm merged neighboring paddles;
- cung cấp mask topology trung thực hơn;
- giảm `uncertain`;
- nếu cần single-side localization, cung cấp observable semantic mới có thể phân
  biệt location, không chỉ tăng absence recall.

Không coi việc đổi input size là retrain thành công nếu downstream event metrics
không tăng.

## 33.2 Annotation semantics ưu tiên

Ưu tiên Contract A:

```text
one physical paddle = one instance
```

Khi gãy giữa, một instance mask được phép có hai disconnected polygons. Annotation
tool/export phải giữ cả hai polygons trong cùng instance identity.

Lợi ích:

- model có physical-paddle grouping signal;
- downstream vẫn tách connected components;
- ít phải temporal pair hơn;
- không phụ thuộc representative contour.

Nếu framework/export không giữ multi-polygon instance ổn định, dùng Contract B:

```text
fragment instances + explicit physical association strategy
```

Không trộn hai annotation semantics trong cùng model version mà không có manifest.

## 33.3 Training data

Phải gồm full operating distribution và hard cases ở Mục 32.3.

Đặc biệt:

- nhiều consecutive frames cùng physical paddle;
- center break phát left/right alternating;
- intact paddle bị occlusion/dropout để làm hard negative;
- complete/partial side break;
- two neighboring paddles;
- chain/background blobs;
- boundary/FOV partial views;
- motion blur/speed variation.

Frame sampling không được tạo train/test leakage giữa cùng event.

## 33.4 Loss/architecture

Spec không bắt buộc YOLO version, loss hoặc backbone cụ thể.

Model chỉ cần thỏa adapter contract và benchmark. Một kiến trúc phức tạp hơn không
được ưu tiên nếu:

- latency/resource xấu hơn;
- output contract khó ổn định;
- event-level metrics không tăng.

## 33.5 Artifact manifest

Mỗi artifact:

Ví dụ shape-only; giá trị opset/output phải đọc từ artifact thực, không copy mẫu.

```json
{
  "model_id": "drag-conveyor-seg-2027-01",
  "sha256": "<64-hex>",
  "format": "onnx",
  "opset": 13,
  "input_shape": [1, 3, 640, 640],
  "output_shapes": [],
  "class_names": ["paddle"],
  "export_metadata": {
    "producer": "<verbatim-from-artifact>",
    "export_date": "<verbatim-from-artifact>"
  },
  "license_metadata": "<verbatim-from-artifact>"
}
```

Không load nếu manifest/artifact hash không khớp.

Postprocess thresholds, runtime/provider và topology/temporal/absence validation
không phải intrinsic artifact capabilities. Chúng nằm trong full-system record
Mục 8.2 keyed bởi toàn bộ evaluated signature. Một artifact có thể có nhiều
capability records theo adapter/runtime/deployment; artifact manifest không
reverse-reference một record. `mask_semantics` là adapter/model-contract claim
được version và benchmark trong capability evaluation, không được tự suy từ ONNX
bytes.

## 33.6 Model-specific thresholds

Confidence, IoU, mask threshold, component size và capability status gắn với full
system signature, không chỉ model hash.

Không copy threshold của model 640 hiện tại sang model mới theo mặc định.

Mỗi combo:

```text
model hash
+ preprocess/postprocess fingerprints
+ adapter version
+ geometry rule/config hashes
+ runtime/provider fingerprint
```

là một evaluated system version.

## 33.7 Shadow replacement

Model mới phải chạy shadow trên cùng original frames:

- compare event counts/IDs;
- compare definitive/uncertain statuses;
- inspect disagreements;
- run held-out acceptance;
- verify runtime/packaging.

Chỉ đổi production manifest sau approval. Rollback chỉ cần trả lại previous
model/config hashes.

## 33.8 Stable downstream interface

Adapter luôn xuất:

```text
list[SegmentationInstance]
```

Geometry pipeline luôn xuất:

```text
PaddleResult geometry_v2_result/2.x
```

Model-specific classes/keypoints MAY nằm trong `adapter_diagnostics`, nhưng final
consumer không được phụ thuộc trực tiếp nếu schema chưa version.

---

# 34. Rollout và migration

## 34.1 Phase 0 — implementation validation

- implement pure/synthetic tests;
- giữ temporal policy `provisional`;
- chạy existing tests;
- chạy pinned model fixtures;
- xác nhận bounded memory;
- không dùng result cho quyết định vận hành.

Exit:

```text
all deterministic/contract tests pass
no legacy regression
```

## 34.2 Phase 1 — offline replay/shadow

- geometry-v2 chạy trên video đã lưu;
- không thay report production;
- human review mọi event;
- thu ground truth/disagreements;
- đo uncertain và tracking errors;
- tune calibration split.

Exit:

```text
core event association gates pass
dangerous false-normal investigated
```

## 34.3 Phase 2 — reviewed pilot

- frontend hiển thị geometry verdict;
- mọi `uncertain` bắt buộc correction;
- broken/bent definitive được reviewer audit theo sampling hoặc 100% ban đầu;
- report chỉ export sau review policy;
- temporal-only center remains uncertain nếu chưa pass enablement.

## 34.4 Phase 3 — production

Chỉ khi Mục 32 pass:

- freeze model/config/rule hashes;
- enable approved capabilities;
- monitor metrics/drift;
- retain quick rollback;
- audit sample định kỳ.

## 34.5 Rollback

Rollback options:

1. chọn inspection mode cũ cho new jobs;
2. rollback geometry config/rule version;
3. rollback model manifest/hash;
4. disable validated temporal capability về provisional.

Không migrate/delete old result JSON. Mỗi result tự mô tả version.

## 34.6 Không reclassify im lặng

Nếu rule/model đổi:

- old jobs giữ verdict/version cũ;
- reprocessing phải tạo revision/new job result;
- lưu source result reference;
- không ghi đè report đã ký/xuất.

## 34.7 Monitoring

Theo ngày/camera/conveyor:

```text
events per minute
duplicate/missed-review rate
uncertain rate
single-side-only rate
center temporal evidence rate
model detection/component distributions
side-reference drift
angle distribution/MAD
processing latency
memory/resource failures
human correction confusion matrix
```

Alerts:

- model hash/config mismatch;
- uncertain rate tăng quá baseline;
- event count giảm đột ngột;
- side reference drift;
- camera/centerline geometry change;
- snapshot/report failure.

---

# 35. Traceability tới yêu cầu chuyển đổi

## 35.1 Các trường hợp model hiện tại

| Yêu cầu/trường hợp | Cách xử lý bắt buộc |
| --- | --- |
| Một frame chỉ detect left, frame sau right | anchor theo `s`, order-preserving tracking, fusion/evidence bins |
| Một frame right, frame sau left | xử lý đối xứng, side không ảnh hưởng identity |
| Chỉ detect một bên suốt event | `uncertain`, reason `single_side_only_location_unidentifiable` |
| Một detection có hai components | tách từ `mask_roi`, không dùng representative contour |
| Nhiều detections cùng paddle | post-component IoS/IoU dedup |
| Gãy giữa có hai side ở các frame khác nhau | temporal center evidence; provisional cho tới validation |
| Gãy toàn bộ/một phần một side nhưng còn fragment | side extent/gap với reference + FOV + multi-bin; recall emission là domain assumption cần benchmark |
| Model mất vài frame | Kalman gap gate + bounded tracklet fusion |
| Hai paddle gần nhau | order preservation + crossing-time uniqueness |
| Model tương lai | adapter + capability manifest, result schema không đổi |

## 35.2 Những điểm sửa so với mô tả `NewChange.md`

1. `largest` được sửa đúng nghĩa:
   - nó chọn representative contour;
   - không quyết định số detection;
   - không nhất thiết xóa components khỏi `mask_roi`.

2. Multi-frame fragment association là core release đầu, không để Phase 2.

3. Một connected frame không tự động thắng mọi bằng chứng gãy:
   - cần hai independent connected bins để definitive intact;
   - connected/broken evidence mạnh xung đột -> uncertain.

4. Không dùng hard presence ratio cho left/right:
   - alternating model có thể làm mỗi side gần 50%;
   - dùng independent bin counts.

5. `mask_roi` không được gọi là full global model mask:
   - nó đã threshold;
   - đã undo letterbox;
   - hiện tại bị crop theo predicted bbox.

6. Không tắt bbox crop mù quáng:
   - uncropped masks thực nghiệm có nhiều blobs;
   - thay bằng capability-aware negative-evidence rules.

7. Optical-flow frames không được vote.

8. Single-side-only không được ép thành một broken location.

9. `broken_both` không được tạo; combined/unresolved damage là `uncertain`.

10. Angle chỉ dùng same-frame connected evidence, không ghép hai thời điểm.

11. Temporal `broken_center` có explicit validation gate, vì behavior do domain
    cung cấp chưa đủ raw labeled data để đo false positive.

## 35.3 Điểm giữ lại từ ý tưởng tài liệu

- geometry-first, VLM off;
- chain centerline và left/right local coordinates;
- detect breakage trước angle;
- labels bent/broken theo vị trí;
- multi-frame robust aggregation;
- median/MAD cho angle;
- global tilt và center kink formulas;
- bounded evidence;
- original frame provenance;
- compatibility với job/report hiện tại.

---

# 36. Verification record tại baseline

## 36.1 Source baseline

```text
repository: Drag_Conveyor
branch: v2
commit: fec1e6b587547e4bd111973039e041c487e05c43
verification date: 2026-07-24
```

PDF mô tả dự án hữu ích để hiểu nghiệp vụ, nhưng branch/source đang chạy mới là
nguồn đúng cho symbol/contracts.

`NewChange.md` là design input, không phải bằng chứng tên hàm/biến hiện tại.

## 36.2 Source facts

`drag_conveyor/inference/yolo_seg_postprocess.py`:

- reconstruct mask probability;
- threshold mask;
- optional crop theo bbox;
- `findContours` trên full cropped binary mask;
- `largest` chọn representative contour;
- `Detection.mask_roi` vẫn được truyền riêng.

`drag_conveyor/pipeline/tracking.py`:

- greedy nearest-centroid;
- `max_jump_px`;
- Y-direction reverse gate;
- bbox-area ratio gate;
- không có fragment/physical-paddle fusion.

`drag_conveyor/app/batch.py`:

- public entry `run_batch_inspection`;
- legacy tracker/trigger/measurement/rules;
- optional VLM stage;
- snapshot filename compatibility.

`server/worker.py`:

- slowdown preprocessing trước inference;
- strict call `profile.with_roi(roi_config)`;
- summary chia normal/suspected;
- R2 upload trước result save/cleanup.

`server/main.py`:

- `RoiIn` và `CreateJobIn` dùng `extra="forbid"`;
- create payload hiện chỉ có ROI, chưa có centerline.

`server/db.py`:

- `roi_config_json` và `result_summary_json` cho phép migration bằng nested JSON;
- không cần thêm column geometry.

`server/report.py` và `server/static/app.js`:

- report labels hiện là `bent_left/right/both`, `broken`;
- frontend hiện dựa vào `vlm_called` để xem defect type đã classified;
- current `applyCorrection` làm mất metadata khi chuyển bucket và set
  `vlm_called=true`;
- current report chỉ xử lý correction khi render, chưa persist review vào DB;
- đây là consumer bắt buộc sửa cho geometry-v2.

## 36.3 Current model artifact

Pinned 640:

```text
path:
weights/model_imgsz_640/best.onnx

SHA-256:
ef05955f43c8db6d2ff76b72fb65806e69afe525e85d8486eeb2dfb7566dcd65

input:
name `images`, float32 [1, 3, 640, 640]

outputs:
`output0`, float32 [1, 300, 38]
`output1`, float32 [1, 32, 160, 160]

ONNX IR = 7
opset = 13
producer = Ultralytics 8.4.67
task = segment; class 0 = `white_bar`; stride = 32; end2end = true
export date metadata = 2026-06-14
license metadata = AGPL-3.0 / ultralytics.com/license
```

Current postprocess:

```text
confidence = 0.4
IoU NMS = 0.5
mask threshold = 0.5
crop mask to bbox = true
contour mode = largest
target class = [0]
min_contour_area = 1.0
preprocess = RGB, normalize=true, letterbox pad value=114
output0 decode includes 32 mask coefficients
```

License metadata được ghi nguyên văn để release owner thực hiện legal/distribution
review; record này không phải kết luận pháp lý.

Other artifacts:

```text
320 SHA-256:
9436aa7be5bba0480585ef4d9afb2c794204b2d02435dc6ac9ce49eeea69dc44

416 SHA-256:
99a7a093a0455b829aa7c189aeac9bc261181b0b2170cb333a8a81d0da1cf835
```

## 36.4 Broken-image emission observations

Với pinned 640/profile hiện tại:

| Fixture | Detections | Significant component areas quan sát |
| --- | ---: | --- |
| `0019_broken` | 2 | một mask có `[1991, 1909]` |
| `0020_broken` | 1 | `[2188, 2118]` |
| `0023_broken` | 1 | `[2280]` |
| `0025_broken` | 1 | `[1992, 1869]` |
| `0028_broken` | 1 | `[3221]` |
| `0029_broken` | 1 | `[2724]` |

Điều này kiểm chứng pipeline phải hỗ trợ cả:

- one detection / one significant component;
- one detection / multiple significant components;
- multiple detections cho cùng visual candidate.

Một same-candidate two-detection case có:

```text
predicted-box IoU        ≈ 0.29587
full detection-mask IoU  ≈ 0.439
full detection-mask IoS  ≈ 0.909
matched-component IoU    ≈ 0.85728
matched-component IoS    ≈ 0.93766
```

nên predicted-box NMS/pre-split IoU đơn thuần không đủ; component split phải xảy
ra trước dedup.

Khoảng cách giữa centers của two-side components trong **cùng frame** quan sát
khoảng:

```text
116–137 px
```

lớn hơn legacy tracker default jump `80 px`; nó chứng minh risk nếu có temporal
side-switching, không phải bằng chứng trực tiếp rằng temporal switch đã xảy ra
trong fixture.

## 36.5 Model-zoo non-equivalence

Trên 16 example crops với từng artifact/profile tương ứng:

| Model | Zero detections | One detection | Two detections |
| --- | ---: | ---: | ---: |
| 320 | 6 | 8 | 2 |
| 416 | 8 | 6 | 2 |
| 640 | 3 | 12 | 1 |

Các model không được coi là drop-in equivalent.

## 36.6 Raw-video mechanics observation

`data/raw_data/vid_1.mp4`:

```text
936 frames
480 x 854
```

Sampling mỗi 10 frame:

```text
94 sampled frames
637 detections
4–8 detections/frame trong phần quan sát
26 multi-component masks
```

Theo heuristic `max(25 px, 5% largest component)`, 25/26 có secondary component
nhỏ; đây là signal cần filter/dedup, **không** phải adjudication rằng chúng là
noise.

Container không strict CFR dù `r_frame_rate=30`, average FPS khoảng `29.96798`.
PTS canonical tăng nghiêm ngặt; frame 33→34 gap xấp xỉ `0.066667 sec`. OpenCV
`CAP_PROP_PTS` từng trả duplicate tại 468/469 trong khi best-effort PTS vẫn tăng,
nên không được dùng OpenCV rounded property làm canonical timestamp.

Video không có adjudicated defect ground truth, nên các số này không phải accuracy
metrics.

## 36.7 Baseline automated tests

Tại thời điểm verification:

```text
58 tests passed
1 PDF smoke test skipped
```

`pytest` command không phải test contract duy nhất trong environment; baseline được
chạy bằng unittest discovery của repository.

Mọi implementation phải ghi lại command/environment exact trong CI artifact.

## 36.8 GitNexus verification

Index:

```text
Drag_Conveyor
1306 symbols
2431 relationships
89 execution flows
```

GitNexus query/context/impact được dùng để kiểm tra:

- inference → postprocess → tracking → trigger → classification;
- batch → worker → summary → frontend/report;
- blast radius các entry/consumer chính.

Trước source edit/commit trong implementation phase vẫn phải chạy lại
`impact`/`detect_changes` theo repository policy.

## 36.9 Những điều chưa được kiểm chứng bằng dữ liệu hiện có

Chưa có đủ evidence để khẳng định:

- xác suất/recall luân phiên left-right của center break;
- false-positive temporal pairing trên intact paddles;
- exact per-class precision/recall;
- angle error so với cơ khí thật;
- side extent thresholds production;
- behavior trên mọi camera/speed/lighting;
- reliable inference từ complete-side absence.
- recall của side remnant cho partial/complete one-side break (đây là domain input
  của chủ dự án, chưa có annotated event benchmark);
- end-to-end deployment latency, peak RSS và throughput SLA.

Spec xử lý các khoảng trống này bằng:

- `uncertain`;
- capability gates;
- provisional policies;
- ground-truth acceptance trước enablement.

## 36.10 Runtime/determinism spot check

Exact environment đã đo:

```text
Python 3.12.3
ONNX Runtime 1.27.0
OpenCV 4.13.0
NumPy 2.4.6
CPUExecutionProvider
host: Intel Xeon E3-1505M v6 (4C/8T)
```

Whole-frame `vid_1` ROI 480×854, input 640, 5 warm-up + 40 runs:

| Phase | p50 | p95 |
| --- | ---: | ---: |
| Inference | 107.43 ms | 126.29 ms |
| Postprocess | 48.95 ms | 83.17 ms |
| Preprocess + inference + postprocess | 160.13 ms | 190.29 ms |

Tương đương khoảng 6.2 fps cho đoạn pipeline này, không đạt realtime 30 fps trên
host đo. 20 repeated inference trên fixture 0019 cho raw output/postprocess hash
bit-identical trong exact environment. Đây là baseline feasibility/determinism
spot check, không thay thế benchmark E2E, RSS hoặc production provider/hardware.

---

# 37. Feasibility conclusion

## 37.1 Kết luận tổng thể

Giải pháp **khả thi và thực hiện được** với model hiện tại nếu mục tiêu release đầu
là:

- tận dụng positive mask evidence;
- ghép fragment đa frame;
- giữ output `uncertain` khi dữ liệu không đủ;
- không hứa định vị chính xác từ single-side-only;
- validation trước khi bật temporal-only definitive label.

Không cần retrain model để xây framework/pipeline v2.

Khả thi ở đây nghĩa là functional/offline workflow có audit và abstention an toàn.
Realtime 30 fps không khả thi trên host baseline đã đo cho current 640 CPU path
(khoảng 6.2 fps partial pipeline); throughput production chỉ được kết luận sau
benchmark deployment profile.

## 37.2 Khả thi theo chức năng

| Chức năng | Với model hiện tại | Điều kiện |
| --- | --- | --- |
| Tách nhiều components trong một mask | Khả thi | đọc `mask_roi`, không contour representative |
| Dedup nhiều detections cùng paddle | Khả thi | component IoS/IoU + anchor |
| Giữ identity khi left/right luân phiên | Khả thi | `s` tracker + temporal fusion |
| Center break có two-side same-frame evidence | Khả thi | ≥2 independent topology bins |
| Center break temporal-only | Khả thi về thuật toán | definitive chỉ sau validation |
| Chỉ một side toàn event | Không thể định vị chắc chắn | bắt buộc `uncertain` |
| Partial side break | Khả thi có điều kiện | expected extent, FOV, repeated positive geometry |
| Complete side loss từ absence | Không đủ để định vị | cần observable/model contract mới, không chỉ flag |
| Bent classification | Khả thi có điều kiện | connected same-frame, stable fit, camera validation |
| Tái dùng API/DB/R2/report | Khả thi | adapter/compatibility fields |

## 37.3 Giới hạn mang tính thông tin

Hai thế giới vật lý có thể sinh cùng observations:

```text
World A:
    broken_right, chỉ left còn thấy

World B:
    broken_center, model chỉ emit left

World C:
    intact/other defect, right bị FOV/dropout
```

Nếu toàn event chỉ có `LEFT_ONLY`, software không có measurement nào phân biệt ba
thế giới đó. Đây là giới hạn nhận dạng, không phải thiếu heuristic.

Mọi spec ép một nhãn definitive trong tình huống này sẽ kém chính xác hơn, không
“mạnh” hơn.

## 37.4 Mức thay đổi code

Core CV là thay đổi lớn về thuật toán, nhưng blast radius code cũ được giới hạn
bằng parallel mode:

- legacy inference engine/video/job/storage reuse;
- legacy classifier/tracker không sửa logic;
- DB không migration;
- output/report có compatibility layer;
- frontend thêm consumer path;
- model hiện tại giữ nguyên/pin hash.

Đây là phương án ít thay đổi nhất mà vẫn đáp ứng đúng behavior model đã nêu. Cố vá
trực tiếp `CentroidTracker`/`RuleEngine` cũ sẽ tạo nhiều condition chéo, khó test và
dễ làm hỏng modes cũ hơn.

---

# 38. Definition of Done

Implementation chỉ được coi hoàn tất khi:

## 38.1 Functional

- `geometry_v2` có đầy đủ pipeline Mục 9;
- bảy physical statuses cộng `uncertain` abstention; không có `broken_both`;
- single-side invariant;
- center/side/angle decision tables implemented;
- VLM hard-off;
- original-frame-only evidence;
- deterministic IDs/results;
- snapshots/audit JSON.

## 38.2 Compatibility

- two legacy modes behavior giữ nguyên;
- current API payload vẫn hoạt động;
- geometry API validated;
- no DB migration;
- old/new summary đọc được;
- old/new report flow hoạt động;
- R2 paths/filenames giữ;
- packaged runtime chứa đúng artifact.

## 38.3 Verification

- all tests Mục 31 pass;
- existing baseline tests pass;
- pinned model contract tests pass;
- packaged smoke pass;
- no NaN/Infinity;
- bounded-memory test pass;
- deterministic replay pass;
- GitNexus impact rechecked trước mỗi source symbol edit;
- `detect_changes` xác nhận expected scope trước commit.

## 38.4 Production readiness

- ground-truth dataset/adjudication hoàn tất;
- acceptance Mục 32 pass;
- mechanical angle thresholds approved;
- side reference/calibration approved;
- capability statuses cập nhật theo evidence;
- rollout/shadow/pilot sign-off;
- rollback artifact đã thử.

## 38.5 Không được tuyên bố hoàn tất nếu

- temporal-only rule được bật definitive chỉ dựa trên assumption;
- single-side-only bị ép nhãn;
- report coi uncertain là normal;
- angle lấy từ hai frames khác nhau;
- synthetic frames tăng vote;
- model/config hash không lưu;
- old modes regression;
- chỉ test ảnh crop mà không test raw event sequences.

---

# 39. Quyết định cuối cùng của spec

Release bootstrap:

```text
inspection_mode = geometry_v2
model = current pinned 640 ONNX
VLM = disabled
slow-motion interpolation = disabled
single-side-only = uncertain
temporal broken-center = implemented, provisional/shadow
same-frame center topology = implemented, provisional/shadow
side geometry/localized side break = implemented, provisional/shadow
angle = implemented, provisional/shadow
production_enabled.* = false tại baseline
legacy modes = unchanged
```

Không được phát `normal`, `bent_*` hay `broken_*` canonical chỉ vì algorithm đã
implemented. Cho tới khi capability gate tương ứng pass, geometry-v2 dùng để
replay/shadow/reviewer workflow và canonical production outcome là `uncertain`
hoặc job/report block theo contract.

Sau khi một capability validation pass:

```text
tạo immutable full-system capability record mới
validation.<path> = validated
production_enabled.<path> = true
update deployment binding + capability_record_hash + resolved_deployment_hash
bump rule version nếu decision behavior thay đổi
chạy shadow/approval/rollback gate trước khi promote
```

Sau khi model mới được train:

```text
thay adapter manifest/model-specific thresholds
giữ geometry_v2 result schema và downstream consumers
```

Tài liệu này thay thế `NewChange.md` làm implementation specification. File
`NewChange.md` vẫn được giữ nguyên như design-history input.
