import unittest

import numpy as np

from higgslens.core import select_threshold_for_recall, summarize_at_threshold


class ThresholdLogicTests(unittest.TestCase):
    def test_validation_threshold_reaches_target_recall(self):
        y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        p = np.array([0.05, 0.15, 0.25, 0.60, 0.45, 0.70, 0.80, 0.95])
        threshold = select_threshold_for_recall(y, p, 0.75)
        summary = summarize_at_threshold(y, p, threshold)
        self.assertGreaterEqual(summary["signal_efficiency"], 0.75)
        self.assertGreaterEqual(summary["background_rejection"], 0.0)
        self.assertLessEqual(summary["background_rejection"], 1.0)

    def test_invalid_probability_is_rejected(self):
        with self.assertRaises(ValueError):
            select_threshold_for_recall([0, 1], [0.2, 1.2], 0.8)

    def test_label_and_probability_lengths_must_match(self):
        with self.assertRaises(ValueError):
            summarize_at_threshold([0, 1], [0.1], 0.5)

    def test_must_have_both_classes(self):
        with self.assertRaises(ValueError):
            select_threshold_for_recall([1, 1, 1], [0.2, 0.5, 0.9], 0.8)


if __name__ == "__main__":
    unittest.main()
