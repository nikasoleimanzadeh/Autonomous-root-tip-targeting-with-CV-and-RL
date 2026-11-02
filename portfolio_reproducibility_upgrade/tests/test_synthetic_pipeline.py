"""Dependency-free tests for the public-safe demonstration."""
import importlib.util
import tempfile
import unittest
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parents[1] / 'demos' / 'synthetic_pipeline.py'
SPEC = importlib.util.spec_from_file_location('synthetic_pipeline', MODULE_PATH)
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


class SyntheticDemoTests(unittest.TestCase):
    def test_five_roots_and_tips(self):
        roots = module.synthetic_roots()
        self.assertEqual(len(roots), 5)
        self.assertEqual(module.detect_tips(roots), [root[-1] for root in roots])

    def test_mapping_and_axis_order(self):
        a = module.pixel_to_robot(10, 20)
        b = module.pixel_to_robot(11, 20)
        c = module.pixel_to_robot(10, 21)
        self.assertAlmostEqual(b[0], a[0])
        self.assertGreater(b[1], a[1])
        self.assertGreater(c[0], a[0])
        self.assertAlmostEqual(c[1], a[1])

    def test_rejects_out_of_bounds(self):
        for xy in [(-1, 1), (600, 1), (1, float('nan'))]:
            with self.subTest(xy=xy), self.assertRaises(ValueError):
                module.pixel_to_robot(*xy)

    def test_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = module.run(Path(tmp))
            self.assertEqual(report['plant_count'], 5)
            self.assertFalse(report['calibration_verified'])
            for name in ('targets.csv', 'synthetic_roots.svg', 'report.json'):
                self.assertTrue((Path(tmp) / name).exists())


if __name__ == '__main__':
    unittest.main()
