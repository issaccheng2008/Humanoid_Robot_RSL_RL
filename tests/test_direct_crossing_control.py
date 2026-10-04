import ast
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'hpc'))
from direct_crossing_control import configure_direct_crossing

class DirectControlTests(unittest.TestCase):
    def cfg(self):
        return NS(fixed_command_mode='touchdown',fixed_walk_step=.1,fixed_initial_gap=.08)
    def saved(self):
        return {'infos':{'direct_crossing_curriculum':{'level':5}}}
    def test_warm_start_starts_level_zero_and_replay_freezes_saved_level(self):
        cfg=self.cfg()
        configure_direct_crossing(cfg,self.saved(),True)
        self.assertEqual(cfg.direct_curriculum_level,0)
        self.assertTrue(cfg.direct_curriculum_auto_advance)
        cfg=self.cfg()
        configure_direct_crossing(cfg,self.saved(),False)
        self.assertEqual(cfg.direct_curriculum_level,5)
        self.assertFalse(cfg.direct_curriculum_auto_advance)
        self.assertEqual(cfg.direct_curriculum_saved, self.saved()['infos']['direct_crossing_curriculum'])
    def test_old_replay_uses_new_level_zero_but_old_resume_is_rejected(self):
        cfg=self.cfg()
        configure_direct_crossing(cfg,{},False)
        self.assertTrue(cfg.direct_curriculum_enabled)
        self.assertEqual(cfg.fixed_walk_step,.10)
        with self.assertRaises(ValueError): configure_direct_crossing(self.cfg(),{},True,True)
    def test_resume_restores_level_and_cannot_change_it_silently(self):
        cfg=self.cfg()
        configure_direct_crossing(cfg,self.saved(),True,True)
        self.assertEqual(cfg.direct_curriculum_level,5)
        with self.assertRaises(ValueError): configure_direct_crossing(self.cfg(),self.saved(),True,True,3)
        cfg.fixed_command_mode='phase_clock'
        with self.assertRaises(ValueError):configure_direct_crossing(cfg,{},True)
    def test_train_and_play_default_to_physical_stage_two(self):
        for filename in ('train_walk_stop_cross.py','play_walk_stop_cross.py'):
            tree=ast.parse((ROOT/'hpc'/filename).read_text(encoding='utf-8'))
            calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)
                   and n.func.attr=='add_argument' and n.args and isinstance(n.args[0],ast.Constant)
                   and n.args[0].value=='--stage']
            self.assertEqual(len(calls),1)
            default=next(k.value for k in calls[0].keywords if k.arg=='default')
            self.assertEqual(ast.literal_eval(default),2)

if __name__=='__main__':unittest.main()
