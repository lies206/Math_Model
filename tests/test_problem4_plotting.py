from pathlib import Path
import tempfile
import unittest

import numpy as np


class Problem4PlotFallbackTests(unittest.TestCase):
    def test_standard_library_svg_chart_needs_no_optional_plotting_package(self):
        from problem4_plotting import write_svg_line_chart

        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "chart.svg"
            write_svg_line_chart(
                output,
                "测试图",
                "时段",
                "数值",
                [
                    ("曲线A", np.asarray([0.0, 1.0, 0.5]), "#355C7D"),
                    ("曲线B", np.asarray([0.2, 0.4, 0.8]), "#E76F51"),
                ],
            )
            content = output.read_text(encoding="utf-8")

        self.assertIn("<svg", content)
        self.assertIn("测试图", content)
        self.assertIn("曲线A", content)
        self.assertIn("polyline", content)


if __name__ == "__main__":
    unittest.main()
