import ast
import math
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
import torch

class OverspeedTests(unittest.TestCase):
    def test_threshold_direction_sum_and_weight(self):
        path = Path(__file__).resolve().parents[1]/'tasks/manager_based/humanoid_robot_policy_rsl_rl/mdp/fixed_stick.py'
        tree = ast.parse(path.read_text(encoding='utf-8'))
        node = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='fixed_joint_overspeed_penalty')
        ns={'torch':torch,'math':math,'SceneEntityCfg':object}
        exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),ns)
        threshold=25*2*math.pi/60
        env=NS(scene={'robot':NS(data=NS(joint_vel=torch.tensor([
            [0.,threshold,-threshold], [threshold+1,-threshold-2,0.],
            [threshold*2,0.,0.]],dtype=torch.float64)))})
        fn=ns['fixed_joint_overspeed_penalty']
        result=fn(env,threshold,NS(name='robot',joint_ids=slice(None)))
        torch.testing.assert_close(result,torch.tensor([0.,5.,threshold**2],dtype=torch.float64))
        self.assertAlmostEqual(float(result[2]*-.1*.02),-.013707783890401887)
        with self.assertRaises(ValueError): fn(env,0.,NS(name='robot',joint_ids=slice(None)))

if __name__=='__main__': unittest.main()
