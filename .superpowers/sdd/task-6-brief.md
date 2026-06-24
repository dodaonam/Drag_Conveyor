# Task 6 brief (from approved plan)

### Task 6: Frontend — "Lưu kết quả" button, validation, API call

**Files:**
- Modify: `server/static/index.html` (result screen, near `#btn-new-job` ~line 163)
- Modify: `server/static/app.js`
- Modify: `server/static/styles.css`
- Test: `tests/test_report_frontend_contract.py` (create) — mirrors existing `test_frontend_trigger_preview_matches_runtime_contract`.

**Interfaces:**
- Consumes: `POST /api/jobs/{job_id}/report` (Task 5); `G.allDefects`, `G.allNormals`, `G.inspector`, `G.conveyor`, `G.jobId`, `api()`, `DEFECT_LABEL_KEYS`.

- [ ] **Step 1: Write the failing contract test**

Create `tests/test_report_frontend_contract.py`:

```python
from __future__ import annotations

import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "server" / "static"


class ReportFrontendContractTests(unittest.TestCase):
    def test_index_has_save_button(self) -> None:
        html = (STATIC / "index.html").read_text(encoding="utf-8")
        self.assertIn('id="btn-save-report"', html)

    def test_appjs_posts_to_report_endpoint(self) -> None:
        js = (STATIC / "app.js").read_text(encoding="utf-8")
        self.assertIn("/report", js)
        self.assertIn("collectCorrections", js)
        self.assertIn("saveReport", js)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m unittest discover -s tests -p "test_report_frontend_contract.py" -v`
Expected: FAIL — `AssertionError: 'id="btn-save-report"' not found`.

- [ ] **Step 3: Add the button + status element to `server/static/index.html`**

Find (around line 163): `<button class="btn btn-primary" id="btn-new-job">Kiểm tra tiếp</button>` and insert BEFORE it:

```html
          <button class="btn btn-success" id="btn-save-report">Lưu kết quả</button>
          <div id="msg-save" class="save-msg" style="display:none"></div>
```

- [ ] **Step 4: Add JS to `server/static/app.js`**

Add near the other result helpers (after `applyCorrection`):

```javascript
const VALID_DEFECT_TYPES = ['bent_left', 'bent_right', 'bent_both', 'broken'];

function collectCorrections() {
  const out = [];
  for (const d of G.allDefects) out.push({ track_id: d.track_id, defect_type: d.defect_type });
  for (const n of G.allNormals) out.push({ track_id: n.track_id, defect_type: 'normal' });
  return out;
}

function hasUnclassified() {
  return G.allDefects.some(d => !VALID_DEFECT_TYPES.includes(d.defect_type));
}

function setSaveMsg(text, isErr) {
  const el = document.getElementById('msg-save');
  el.style.display = 'block';
  el.textContent = text;
  el.classList.toggle('err', !!isErr);
}

async function saveReport() {
  if (hasUnclassified()) {
    setSaveMsg('Còn thanh chưa phân loại — hãy phân loại hết (trái/phải/2 bên/gãy) hoặc đánh dấu bình thường trước khi lưu.', true);
    return;
  }
  const btn = document.getElementById('btn-save-report');
  btn.disabled = true;
  btn.textContent = 'Đang lưu...';
  try {
    const res = await api('/api/jobs/' + G.jobId + '/report', {
      method: 'POST',
      body: JSON.stringify({
        inspector_name: G.inspector,
        conveyor_name: G.conveyor,
        corrections: collectCorrections(),
      }),
    });
    btn.textContent = 'Đã lưu ✓';
    setSaveMsg('Đã lưu: ' + res.filename, false);
  } catch (e) {
    btn.disabled = false;
    btn.textContent = 'Lưu kết quả';
    setSaveMsg('Lỗi khi lưu: ' + e.message, true);
  }
}

document.getElementById('btn-save-report').addEventListener('click', saveReport);
```

- [ ] **Step 5: Add styles to `server/static/styles.css`**

Append:

```css
.btn-success { background: #15803d; color: #fff; }
.btn-success:disabled { opacity: .6; }
.save-msg { margin-top: 8px; font-size: 13px; color: #166534; }
.save-msg.err { color: #b91c1c; }
```

- [ ] **Step 6: Run the contract test to verify it passes**

Run: `.venv/bin/python -m unittest discover -s tests -p "test_report_frontend_contract.py" -v`
Expected: PASS (2 tests).

- [ ] **Step 7: Manual verification**

Start the server and exercise the flow end-to-end:

```bash
cd server && uv run uvicorn main:app --host 127.0.0.1 --port 8000
```

Verify: complete a job → open results on the device → leave one defect as "Không xác định rõ" → tap "Lưu kết quả" → confirm it is blocked with the Vietnamese message. Reclassify all → tap again → confirm "Đã lưu: <filename>" and the PDF appears in `REPORTS_DIR` (open the Windows folder) with the XT-100 layout.

- [ ] **Step 8: Commit**

```bash
git add server/static/index.html server/static/app.js server/static/styles.css tests/test_report_frontend_contract.py
git commit -m "feat: add Lưu kết quả button + report save flow (frontend)" -m "Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---
