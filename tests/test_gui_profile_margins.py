from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from gui.app import (
    _parse_margin_value,
    _service_control_command,
    load_profile_margin_values,
    resolve_margin_values,
    update_profile_margin_values,
)


ROOT = Path(__file__).resolve().parents[1]
BASE_PROFILE = ROOT / "config" / "base_profile.json"


class GuiProfileMarginTests(unittest.TestCase):
    def test_parse_margin_value_accepts_decimal_comma(self) -> None:
        self.assertEqual(_parse_margin_value("0,09", "x"), 0.09)

    def test_parse_margin_value_rejects_out_of_range(self) -> None:
        with self.assertRaisesRegex(ValueError, r"\[0, 1\)"):
            _parse_margin_value("1.0", "x")

    def test_load_profile_margin_values_reads_base_profile_fields(self) -> None:
        values = load_profile_margin_values(BASE_PROFILE)
        self.assertIn("length_upper_margin", values)
        self.assertIn("width_lower_margin", values)

    def test_resolve_margin_values_uses_zero_for_blank_fields(self) -> None:
        resolved = resolve_margin_values(
            {
                "length_upper_margin": "",
                "width_lower_margin": "0.12",
            },
        )
        self.assertEqual(resolved["length_upper_margin"], 0.0)
        self.assertEqual(resolved["width_lower_margin"], 0.12)

    def test_update_profile_margin_values_writes_valid_profile(self) -> None:
        raw = json.loads(BASE_PROFILE.read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "base_profile.json"
            path.write_text(json.dumps(raw, indent=2, ensure_ascii=False), encoding="utf-8")

            update_profile_margin_values(
                path,
                {
                    "length_upper_margin": 0.12,
                    "width_lower_margin": 0.08,
                },
            )

            updated = json.loads(path.read_text(encoding="utf-8"))
            auto = updated["inspection"]["auto_baseline"]
            self.assertEqual(auto["length_upper_margin"], 0.12)
            self.assertEqual(auto["width_lower_margin"], 0.08)

    def test_service_control_command_restart_restarts_service(self) -> None:
        cmd = _service_control_command("restart")
        self.assertIn("sc stop DragConveyorTunnel", cmd)
        self.assertIn("sc start DragConveyorTunnel", cmd)

    def test_service_control_command_stop_stops_service(self) -> None:
        self.assertEqual(_service_control_command("stop"), "sc stop DragConveyorTunnel")


if __name__ == "__main__":
    unittest.main()
