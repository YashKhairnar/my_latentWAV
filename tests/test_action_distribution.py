import unittest
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from xml.etree import ElementTree

import numpy as np

from latentwav.action_distribution import compare_action_distributions
from scripts.plot_cem_action_distributions import plot_report


class ActionDistributionTests(unittest.TestCase):
    def test_identical_distributions_have_zero_divergence(self):
        actions = np.array(
            [[-1.0, 0.0], [-0.5, 0.5], [0.5, -0.5], [1.0, 0.0]],
            dtype=np.float32,
        )
        report = compare_action_distributions([actions], [actions], [actions], bins=4)
        self.assertEqual(report["groups"]["expert_train"]["count"], 4)
        self.assertAlmostEqual(
            report["jensen_shannon_bits_vs_expert"]["expert_cem"], 0.0
        )
        self.assertAlmostEqual(
            report["jensen_shannon_bits_vs_expert"]["fine_tuned_cem"], 0.0
        )

    def test_shift_and_nonfinite_actions(self):
        expert = np.array(
            [[-1.0, -1.0], [-0.5, -0.5], [0.5, 0.5], [1.0, 1.0], [np.nan, 0.0]]
        )
        baseline = np.array([[0.9, 0.9], [1.1, 1.1]])
        report = compare_action_distributions([expert], [baseline], [baseline], bins=4)
        self.assertEqual(report["groups"]["expert_train"]["count"], 4)
        self.assertGreater(
            report["jensen_shannon_bits_vs_expert"]["expert_cem"], 0.0
        )
        self.assertEqual(report["groups"]["expert_cem"]["count"], 2)
        self.assertEqual(
            np.asarray(report["groups"]["expert_cem"]["histogram"]).sum(), 2
        )
        self.assertEqual(
            report["groups"]["expert_cem"]["fraction_outside_expert_bounds"],
            0.5,
        )

    def test_plot_is_valid_svg(self):
        actions = np.array([[-1.0, -1.0], [0.0, 0.0], [1.0, 1.0]])
        comparison = compare_action_distributions([actions], [actions], [actions])
        with TemporaryDirectory() as directory:
            report_path = Path(directory) / "report.json"
            plot_path = Path(directory) / "plot.svg"
            report_path.write_text(json.dumps({"action_distributions": comparison}))
            plot_report(report_path, plot_path)
            root = ElementTree.parse(plot_path).getroot()
            self.assertTrue(root.tag.endswith("svg"))
            self.assertEqual(len(root.findall("{*}rect")), 3 * 32 * 32 + 4)


if __name__ == "__main__":
    unittest.main()
