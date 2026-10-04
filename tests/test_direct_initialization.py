import sys
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'hpc'))
from direct_crossing_control import ensure_direct_crossing_initialized

class InitializationTests(unittest.TestCase):
    def env(self):
        class Env:
            def __init__(self):
                self.cfg=NS(direct_curriculum_enabled=True,direct_curriculum_level=5,
                    direct_curriculum_auto_advance=True,direct_curriculum_saved=None)
                self._fixed_stick_state=NS(episode_level=[-1],command_mode='touchdown',collisionless_mode=False)
                self.reset_count=0
            def reset(self):self.reset_count+=1
        return Env()
    def test_missing_curriculum_is_initialized_and_geometry_reset_before_action(self):
        env=self.env()
        created=[]
        def factory(**kwargs):
            result=NS(**kwargs)
            created.append(result)
            return result
        self.assertTrue(ensure_direct_crossing_initialized(env,factory))
        self.assertEqual(env.reset_count,1)
        self.assertEqual(env._fixed_stick_state.curriculum.level,5)
        self.assertFalse(ensure_direct_crossing_initialized(env,factory))
        self.assertEqual(len(created),1)
    def test_stale_task_files_fail_with_sync_diagnostic(self):
        env=self.env()
        del env._fixed_stick_state.episode_level
        with self.assertRaisesRegex(RuntimeError,'fixed_stick_state.py'):
            ensure_direct_crossing_initialized(env,lambda **kw:NS(**kw))
        self.assertEqual(env.reset_count,0)
    def test_disabled_curriculum_does_not_silently_train_old_task(self):
        env=self.env()
        env.cfg.direct_curriculum_enabled=False
        with self.assertRaisesRegex(RuntimeError,'curriculum'):
            ensure_direct_crossing_initialized(env,lambda **kw:NS(**kw))

if __name__=='__main__':unittest.main()
