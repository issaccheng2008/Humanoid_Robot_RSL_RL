import argparse
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'hpc'))
from fixed_stick_control import resolve_control, validate_control_resume


class ResumeTests(unittest.TestCase):
    def args(self, **kwargs):
        values = dict(command_mode=None, walk_end_s=None, lead_end_s=None, sequence_end_s=None)
        values.update(kwargs)
        return argparse.Namespace(**values)

    def test_old_checkpoint_warm_starts_clock_but_cannot_resume_as_clock(self):
        control = resolve_control(self.args(), {})
        self.assertEqual(control['command_mode'], 'phase_clock')
        validate_control_resume({}, control, False)
        with self.assertRaises(ValueError):
            validate_control_resume({}, control, True)
        validate_control_resume({}, resolve_control(self.args(command_mode='touchdown'), {}), True)

    def test_saved_timing_restored_and_change_requires_warm_start(self):
        original = resolve_control(self.args(lead_end_s=.48), {})
        checkpoint = {'infos':{'fixed_stick_control':original}}
        restored = resolve_control(self.args(), checkpoint)
        self.assertEqual(restored, original)
        validate_control_resume(checkpoint, restored, True)
        changed = resolve_control(self.args(lead_end_s=.5), checkpoint)
        with self.assertRaises(ValueError):
            validate_control_resume(checkpoint, changed, True)
        validate_control_resume(checkpoint, changed, False)
