import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
import torch

ROOT=Path(__file__).resolve().parents[1]
def load(path, name):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module

class CheckpointTests(unittest.TestCase):
    def setUp(self):
        path=ROOT/'hpc/fine_tune_checkpoint.py'
        self.assertTrue(path.is_file(), 'fine-tune checkpoint loader missing')
        self.helper=load(path,'fine_tune_checkpoint')
        cls=load(ROOT/'tasks/manager_based/humanoid_robot_policy_rsl_rl/agents/entropy_schedule.py','entropy_schedule').EntropyScheduledPPO
        self.alg=cls(torch.nn.Linear(2,2),torch.nn.Linear(2,1),None,training_phase=5,
                     training_stage='walk_stop_cross_v32',entropy_schedule=((0,0.0005),),
                     learning_rate=5e-5,schedule='fixed')
        self.runner=SimpleNamespace(alg=self.alg,current_learning_iteration=0)
        self.saved=self.alg.save()
        self.saved['iter']=100
        self.saved['entropy_schedule_state']={'training_phase':5,'phase_iteration':15451}
        self.saved['optimizer_state_dict']['param_groups'][0]['lr']=0.001

    def test_warm_start_does_not_restore_old_optimizer_or_clock(self):
        self.helper.load_fine_tune_state(self.runner,self.saved,resume=False)
        self.assertEqual(self.runner.current_learning_iteration,0)
        self.assertEqual(self.alg.entropy_schedule_iteration,0)
        self.assertEqual(self.alg.optimizer.param_groups[0]['lr'],5e-5)

    def test_resume_rejects_legacy_checkpoint(self):
        with self.assertRaises(ValueError): self.helper.load_fine_tune_state(self.runner,self.saved,resume=True)

    def test_resume_restores_clock_and_enforces_low_lr(self):
        self.saved['entropy_schedule_state'].update(training_stage='walk_stop_cross_v32',phase_iteration=101)
        self.helper.load_fine_tune_state(self.runner,self.saved,resume=True)
        self.assertEqual(self.runner.current_learning_iteration,101)
        self.assertEqual(self.alg.entropy_schedule_iteration,101)
        self.assertEqual(self.alg.optimizer.param_groups[0]['lr'],5e-5)

    def test_warm_start_from_same_stage_still_resets_schedule(self):
        self.saved['entropy_schedule_state']['training_stage']='walk_stop_cross_v32'
        self.helper.load_fine_tune_state(self.runner,self.saved,resume=False)
        self.assertEqual(self.alg.entropy_schedule_iteration,0)

if __name__=='__main__': unittest.main()
