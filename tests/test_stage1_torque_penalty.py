import ast
import __future__
import math
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
import torch

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'tasks/manager_based/humanoid_robot_policy_rsl_rl'

def load_function():
    path=BASE/'mdp/rewards.py'
    node=next(n for n in ast.parse(path.read_text(encoding='utf-8')).body
              if isinstance(n,ast.FunctionDef) and n.name=='joint_torque_over_nominal')
    ns={'torch':torch,'math':math}
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec',
                 flags=__future__.annotations.compiler_flag),ns)
    return ns['joint_torque_over_nominal']

class TorqueTests(unittest.TestCase):
    def call(self,values,**params):
        env=NS(scene={'robot':NS(data=NS(applied_torque=torch.tensor(values,dtype=torch.float32)))})
        return load_function()(env,asset_cfg=NS(name='robot',joint_ids=slice(None)),**params)

    def test_continuous_piecewise_linear_per_joint_and_both_signs(self):
        values=[[0],[1.5],[2],[3],[3.1],[4],[5],[-4]]
        actual=self.call(values,nominal_torque=1.5,upper_torque_threshold=3.,upper_torque_slope=10.)
        torch.testing.assert_close(actual,torch.tensor([0,0,.5,1.5,2.5,11.5,21.5,11.5]))
        # The reward remains a rate: weight -0.1, integrated by 20ms.
        self.assertAlmostEqual(float(-.1*.02*actual[5]),-.023,places=6)

    def test_joint_sum_and_legacy_stage2_behavior(self):
        torch.testing.assert_close(self.call([[2,-4]],nominal_torque=1.5,
            upper_torque_threshold=3.,upper_torque_slope=10.),torch.tensor([12.]))
        torch.testing.assert_close(self.call([[3,4,5,-5]],nominal_torque=4.),torch.tensor([2.]))

    def test_invalid_thresholds_rejected(self):
        for args in ({'nominal_torque':float('nan')},
                     {'nominal_torque':1.5,'upper_torque_threshold':1.},
                     {'nominal_torque':1.5,'upper_torque_threshold':3.,'upper_torque_slope':.5}):
            with self.assertRaises(ValueError):
                self.call([[2]],**args)

    def test_stage_configuration_switches_reward_without_touching_asset_limits(self):
        path=BASE/'fixed_stick_env_cfg.py'
        tree=ast.parse(path.read_text(encoding='utf-8'))
        node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='configure_fixed_stick_stage')
        ns={'EL05_RATED_TORQUE':4.,'LEG_JOINT_NAMES':[],
            'ContactSensorCfg':lambda **kwargs:NS(**kwargs),
            'FixedStickGeometryRecorderCfg':lambda:NS(),
            'RecorderManagerBaseCfg':lambda **kwargs:NS(**kwargs),
            'DatasetExportMode':NS(EXPORT_NONE=0)}
        exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),ns)
        cfg=NS(scene=NS(wooden_bar=NS(spawn=NS(collision_props=NS())),bar_contacts=None),
               sim=NS(dt=.005),decimation=4,
               rewards=NS(collisionless_hit_penalty=NS(weight=-100.),fixed_joint_overspeed=NS(weight=0.),
                          dof_torque_over_nominal=NS(weight=-.1,params={'nominal_torque':4.,'asset_cfg':'keep'})))
        ns['configure_fixed_stick_stage'](cfg,1)
        self.assertEqual(cfg.rewards.dof_torque_over_nominal.params,
            {'nominal_torque':1.5,'asset_cfg':'keep','upper_torque_threshold':3.,'upper_torque_slope':10.})
        ns['configure_fixed_stick_stage'](cfg,2)
        self.assertEqual(cfg.rewards.dof_torque_over_nominal.params['nominal_torque'],1.5)
        self.assertEqual(cfg.rewards.dof_torque_over_nominal.params['upper_torque_threshold'],3.)
        self.assertEqual(cfg.rewards.dof_torque_over_nominal.weight,-1.)
        self.assertEqual(cfg.rewards.fixed_joint_overspeed.weight,-1.)
        self.assertTrue(cfg.scene.wooden_bar.spawn.collision_props.collision_enabled)
        self.assertFalse(cfg.collisionless_mode)

if __name__=='__main__': unittest.main()
