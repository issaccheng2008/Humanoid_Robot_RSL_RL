"""Exercise the real crossing updater with deterministic physical measurements."""
from copy import deepcopy
from types import SimpleNamespace as NS
import sys
from pathlib import Path
import unittest
from unittest.mock import patch
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tasks.manager_based.humanoid_robot_policy_rsl_rl.mdp import wooden_bar as wb
from tasks.manager_based.humanoid_robot_policy_rsl_rl.humanoid_robot_policy_rsl_rl_env_cfg import _crossing_state_update_params, HumanoidRobotPolicyEnvCfg

class Bar:
    def __init__(self): self.writes=0
    def write_root_pose_to_sim(self,pose,env_ids): self.writes+=1
    def write_root_velocity_to_sim(self,velocity,env_ids): pass

class Scene(dict): pass

def vector(value): return NS(torch=value)

class CrossingIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.env=NS(num_envs=2,device='cpu',step_dt=0.02,common_step_counter=0,
                    episode_length_buf=torch.zeros(2,dtype=torch.long),
                    cfg=NS(stop_before_crossing=True,stop_hold_s=0.5,stop_timeout_s=3.0))
        self.velocity=torch.zeros(2,3); self.velocity[1,0]=0.2
        scene=Scene(robot=NS(data=NS(root_pos_w=torch.zeros(2,3),
                    root_lin_vel_w=vector(self.velocity),root_ang_vel_w=vector(torch.zeros(2,3)))),
                    wooden_bar=Bar(),collisionless_wooden_bar=Bar())
        scene.env_origins=torch.zeros(2,3)
        scene.sensors={'contact_forces':NS(data=NS(current_contact_time=torch.ones(2,2),last_air_time=torch.zeros(2,2)),
                                         compute_first_contact=lambda dt:torch.zeros(2,2,dtype=torch.bool))}
        self.env.scene=scene
        self.syncs=0
        def sync(): self.syncs+=1
        self.env.command_manager=NS(get_term=lambda name:NS(_update_command=sync))
        self.params=_crossing_state_update_params()
        self.params['feet_cfg'].body_ids=[0,1];self.params['sensor_cfg'].body_ids=[0,1]
        self.sole=torch.zeros(2,2,4,3)
        self.state=wb._get_state(self.env)
        self.state.initialized[:]=True;self.state.training_phase[:]=4
        self.state.trigger_touchdown_index[:]=3;self.state.touchdown_count[:]=2
        self.state.swing_foot_index[:]=0
        self.state.step_distance[:]=0.08
        self.geometry=patch.object(wb,'_sole_geometry_w',lambda *args:self.sole)
        self.forward=patch.object(wb,'_base_forward_and_sole_front',lambda *args:(torch.tensor([[1.,0.],[1.,0.]]),torch.tensor([[1.,0.,0.,0.],[1.,0.,0.,0.]]),torch.zeros(2,2)))
        self.geometry.start();self.forward.start()
        self.addCleanup(self.geometry.stop);self.addCleanup(self.forward.stop)

    def tick(self):
        self.env.common_step_counter+=1;self.env.episode_length_buf+=1
        return wb._update_crossing_state_once(self.env,**self.params)

    def test_bar_stays_hidden_until_continuous_stop_and_spawns_once(self):
        self.tick()
        self.assertTrue(self.state.stop.active.all())
        for _ in range(24):
            self.tick();self.assertFalse(self.state.spawned.any())
        self.tick()
        self.assertEqual(self.state.spawned.tolist(),[True,False])
        self.assertEqual(self.env.scene['wooden_bar'].writes,1)
        self.assertTrue(self.state.crossing_command[0])
        # A repeated manager call in the same control step must not advance the gate.
        count=self.state.stop.elapsed_steps.clone()
        wb._update_crossing_state_once(self.env,**self.params)
        self.assertTrue(torch.equal(count,self.state.stop.elapsed_steps))
        for _ in range(10): self.tick()
        self.assertEqual(self.env.scene['wooden_bar'].writes,1)

    def test_completion_requires_geometry_and_double_support(self):
        self.tick()
        for _ in range(25): self.tick()
        self.assertFalse(self.state.task_success.any())
        # Move both complete soles beyond the bar, not merely the ankle centers.
        self.sole[0,:,:,0]=1.0
        self.tick();self.assertFalse(self.state.task_success_event.any())
        self.tick();self.assertEqual(self.state.task_success_event.tolist(),[True,False])
        self.tick();self.assertFalse(self.state.task_success_event.any())

    def test_timeout_uses_resolved_event_parameters(self):
        self.env.event_manager=NS(get_term_cfg=lambda name:NS(params=self.params))
        self.tick()
        for _ in range(150): self.tick()
        self.assertEqual(wb.stop_before_crossing_timeout(self.env).tolist(),[False,True])
        self.assertFalse(self.state.spawned[1])

    def test_airborne_clearance_then_retreat_is_not_success(self):
        self.tick()
        for _ in range(25): self.tick()
        contact=self.env.scene.sensors['contact_forces'].data.current_contact_time
        contact[0]=0
        self.sole[0,:,:,0]=1.0
        self.tick()
        self.assertTrue(self.state.crossed[0])
        contact[0]=1
        self.sole[0,:,:,0]=0.0
        self.tick();self.tick()
        self.assertFalse(self.state.task_success[0])

    def test_terminal_failure_cannot_award_crossing_success(self):
        self.tick()
        self.state.task_success_event[0] = True
        self.state.task_success[0] = True
        self.env.termination_manager = NS(terminated=torch.tensor([True,False]))
        reward = wb.physical_bar_crossing_completion_reward(self.env,**self.params)
        self.assertFalse(reward.any())
        self.assertFalse(self.state.task_success[0])

    def test_stop_failure_has_early_termination_penalty(self):
        cfg=HumanoidRobotPolicyEnvCfg()
        self.assertIn('stop_failed', cfg.rewards.termination_penalty.params['term_keys'])

    def test_config_has_registered_terms_and_compatible_interface(self):
        cfg=HumanoidRobotPolicyEnvCfg()
        self.assertTrue(callable(cfg.rewards.stop_stability.func))
        self.assertTrue(callable(cfg.terminations.stop_failed.func))
        self.assertEqual(cfg.episode_length_s,8.0)
        self.assertTrue(cfg.scene.robot.spawn.usd_path.endswith('v3.2.usd'))

if __name__=='__main__': unittest.main()
