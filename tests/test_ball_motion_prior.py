"""Check reachability without suppressing impacts or inventing observations."""

import unittest

import numpy as np

from hawk_eye.models.ball_detector.classical import ClassicalBallTracker


class MotionPriorTest(unittest.TestCase):
    def setUp(self):
        self.tracker = ClassicalBallTracker(motion_prior=True)

    def apply(self, points, replacements=None):
        candidates = [np.zeros((0, 5)) for _ in points]
        for t, xy in (replacements or {}).items():
            candidates[t] = np.column_stack([xy, np.ones((len(xy), 3))])
        return self.tracker.enforce_motion(points, [1. if p else 0. for p in points], candidates)

    def test_short_excursion_requires_an_observed_candidate(self):
        points = [(10., 100.), (20., 100.), (900., 600.), (40., 100.), (50., 100.)]
        out, score = self.apply(points)
        self.assertIsNone(out[2])
        self.assertEqual(score[2], 0.)
        out, score = self.apply(points, {2: [(29., 100.)]})
        np.testing.assert_allclose(out[2], (30., 100.))
        self.assertEqual(score[2], .75)

    def test_impacts_and_long_disappearances_are_allowed(self):
        points = [(100., 100.), (120., 100.), (140., 100.), (120., 100.), (100., 100.)]
        self.assertEqual(self.apply(points)[0], points)
        points = [(10., 10.)] + [None] * 6 + [(900., 600.)]
        self.assertEqual(self.apply(points)[0], points)

    def test_unreachable_nearest_candidate_does_not_hide_a_valid_one(self):
        self.tracker.velocity_offset = 0.
        out, _ = self.apply([(0., 0.), (900., 600.), (156., 0.)], {1: [(81., 0.), (78., 4.)]})
        np.testing.assert_allclose(out[1], (78., 4.))

    def test_offset_candidate_stays_inside_frame(self):
        self.tracker.frame_size = (720, 1280)
        self.tracker.repair_radius = 30
        out, _ = self.apply([(1258., 50.), (200., 600.), (1278., 50.)], {1: [(1279., 50.)]})
        self.assertIsNone(out[1])

    def test_output_lengths_and_fps_scaling(self):
        for count in range(8):
            outputs = self.tracker([np.zeros((72, 128, 3), np.uint8)] * count)
            self.assertTrue(all(len(output) == count for output in outputs))
        self.tracker.fps = 25
        self.tracker([np.zeros((720, 1280, 3), np.uint8)])
        self.assertAlmostEqual(self.tracker.motion_scale * self.tracker.max_step, 96.)


if __name__ == '__main__':
    unittest.main()
