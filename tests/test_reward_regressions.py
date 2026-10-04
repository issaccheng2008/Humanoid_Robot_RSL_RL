"""Reward regressions without starting Isaac Sim.

Load the real reward logic through AST, replacing only simulator measurements.
Run with a Python environment that has PyTorch: python -B -m unittest discover
-s tests -p test_reward_regressions.py -v.
"""
from __future__ import annotations

import ast
import math
from pathlib import Path
from types import ModuleType, SimpleNamespace as NS
import unittest
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
MDP = ROOT / "tasks/manager_based/humanoid_robot_policy_rsl_rl/mdp"
FLAGS = __import__("__future__").annotations.compiler_flag


def execute_nodes(nodes, path, namespace):
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec", flags=FLAGS), namespace)


def load_logic():
    wb = ModuleType("reward_logic")
    wb.__dict__.update(torch=torch, math=math)
    stop_path = MDP / "stop_gate.py"
    execute_nodes([n for n in ast.parse(stop_path.read_text()).body if isinstance(n, ast.ClassDef)],
                  stop_path, wb.__dict__)
    path = MDP / "wooden_bar.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = {
        "_control_step", "_episode_time_s", "_as_env_ids", "_hide_bar", "reset_crossing_state", "_single_swing_foot", "_swing_cycle_touchdown_events",
        "_CrossingState", "_get_state", "_update_crossing_state_once", "_reward_crossing_state",
        "_sole_min_z_over_rectangle", "_course_gap_mask", "_course_inside", "_course_failure_mask", "_course_step_targets", "hurdle_skipped_gap",
        "record_hurdle_curriculum", "reset_hurdle_layout",
        "hurdle_out_of_bounds", "hurdle_course_completed",
        "stick_cleared_reward", "all_sticks_completed_reward", "stick_collision_penalty",
        "stick_failed_penalty", "hurdle_forward_progress_reward", "stick_over_clearance_reward",
        "stick_landing_center_reward", "step_distance_tracking_reward", "_base_forward_and_sole_front",
    }
    nodes = [n for n in tree.body if
             (isinstance(n, ast.Assign) and n.lineno < 45) or
             (isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names)]
    execute_nodes(nodes, path, wb.__dict__)
    real_forward = wb._base_forward_and_sole_front
    wb._sole_geometry_w = lambda *args: None
    wb.yaw_quat = lambda q: q
    wb.quat_apply = lambda q, v: v
    cfg_path = MDP.parent / "humanoid_robot_policy_rsl_rl_env_cfg.py"
    cfg = dict(math=math, Path=Path, WOODEN_BAR_TRAINING_PHASE=5,
               STOP_BEFORE_CROSSING=False, TRAINING_STAGE="cross_ten_sticks_v32")
    cfg["SceneEntityCfg"] = lambda name, **kwargs: NS(name=name, body_ids=[0, 1], **kwargs)
    for node in ast.parse(cfg_path.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign) and node.lineno < 250:
            try:
                execute_nodes([node], cfg_path, cfg)
            except (NameError, AttributeError):
                pass
        if isinstance(node, ast.FunctionDef) and node.name in {
            "_ordered_feet_cfg", "_ordered_feet_sensor_cfg", "_crossing_state_update_params"
        }:
            execute_nodes([node], cfg_path, cfg)
    test_path = ROOT / "tests/test_crossing_integration.py"
    fixtures = dict(torch=torch, unittest=unittest, patch=patch, NS=NS, wb=wb,
                    _crossing_state_update_params=cfg["_crossing_state_update_params"])
    execute_nodes([n for n in ast.parse(test_path.read_text(encoding="utf-8")).body
                   if isinstance(n, (ast.ClassDef, ast.FunctionDef))], test_path, fixtures)
    rewards_path = MDP / "rewards.py"
    reward_namespace = dict(torch=torch, math=math, yaw_quat=lambda q: q,
                            quat_apply_inverse=lambda q, v: v)
    execute_nodes([n for n in ast.parse(rewards_path.read_text()).body
                   if isinstance(n, ast.FunctionDef) and
                   n.name == "track_lin_vel_xy_yaw_frame_quadratic_relative"],
                  rewards_path, reward_namespace)
    return wb, cfg, fixtures["CrossingIntegrationTests"], real_forward, reward_namespace


wb, CONFIG, BaseCase, REAL_FORWARD, REWARDS = load_logic()


class RewardRegressionTests(BaseCase):
    # Simulator configuration construction is covered by the original suite.
    test_4_layer_rewards_exist_and_callable = None
    test_config_has_10_sticks_and_12s_duration = None

    def setUp(self):
        super().setUp()
        self.env.event_manager = NS(get_term_cfg=lambda name: NS(params=self.params))

    def new_step(self):
        self.env.common_step_counter += 1
        self.env.episode_length_buf += 1

    def land(self, foot=0):
        self.contact[0, foot] = 0
        self.tick()
        self.contact[0, foot] = 1
        self.tick()

    def arm_last_step(self, x=.65, y=0., stick=0):
        self.state.current_stick_index[0] = stick
        self.state.crossing_command[0] = True
        self.state.following_step_command_stage[0] = 1
        self.state.crossing_foot_index[0] = 0
        self.state.following_foot_index[0] = 1
        self.sole[0, :, :, 0] = x
        self.sole[0, :, :, 1] = y

    def arm_lead(self):
        self.state.crossing_command[0] = True
        self.state.crossing_foot_index[0] = 0
        self.state.following_foot_index[0] = 1
        self.state.following_step_command_stage[0] = 0
        self.sole[0, 1, :, 0] = .35

    def test_wrong_gap_does_not_advance_to_follow(self):
        self.arm_lead()
        self.sole[0, 0, :, 0] = .90
        self.land(0)
        self.assertEqual(self.state.following_step_command_stage[0].item(), 0)

    def test_follow_target_matches_gap_center(self):
        self.arm_lead()
        self.sole[0, 0, :, 0] = .615
        self.land(0)
        self.assertEqual(self.state.following_step_command_stage[0].item(), 1)
        self.assertAlmostEqual(self.state.step_distance[0].item(), 0., places=5)

    def test_wrong_foot_landing_has_no_center_reward(self):
        self.arm_lead()
        self.sole[0, 1, :, 0] = .615
        self.land(1)
        self.assertEqual(self.state.stick_landing_center_reward_val[0].item(), 0.)

    def test_both_feet_need_contact_before_advancing(self):
        self.arm_last_step(x=.615)
        self.contact[0, 1] = 0
        self.tick()
        self.contact[0, 1] = 1
        self.contact[0, 0] = 0
        self.tick()
        self.assertEqual(self.state.current_stick_index[0].item(), 0)

    def test_crossing_step_distance_reward_enabled(self):
        self.arm_lead()
        self.sole[0, 0, :, 0] = .615
        self.contact[0, 0] = 0
        self.tick()
        self.contact[0, 0] = 1
        sensor=self.env.scene.sensors['contact_forces']
        sensor.compute_first_contact=lambda dt: torch.tensor([[True,False],[False,False]])
        sensor.data.last_air_time[0,0]=.10
        self.new_step()
        reward=wb.step_distance_tracking_reward(self.env,gaussian_std=.03,**self.params)
        self.assertGreater(reward[0].item(), .9)

    def test_skip_terminates_only_grounded_foot(self):
        self.arm_lead()
        self.sole[0,0,:,0]=.90
        self.contact[0,0]=0
        args=(self.env,self.params['feet_cfg'],self.params['sensor_cfg'],CONFIG['FOOT_SOLE_VERTICES'])
        self.assertFalse(wb.hurdle_skipped_gap(*args)[0])
        self.contact[0,0]=1
        self.assertTrue(wb.hurdle_skipped_gap(*args)[0])

    def test_dynamic_spacing_used_by_collision_and_landing(self):
        self.state.course_spacing[:]=.28
        self.state.current_stick_index[0]=1
        self.arm_lead()
        self.sole[0,0,:,0]=.92
        self.sole[0,1,:,0]=.64
        self.land(0)
        self.assertEqual(self.state.following_step_command_stage[0].item(),1)
        self.assertAlmostEqual(self.state.step_distance[0].item(),0.,places=5)
        self.sole[1,0,:,0]=.78
        self.tick()
        self.assertTrue(self.state.stick_hit_flags[1,1])
        self.assertFalse(self.state.stick_hit_flags[1,2])

    def test_curriculum_waits_for_accuracy_and_completion(self):
        self.env.cfg.hurdle_gap_curriculum=True
        self.env.cfg.hurdle_gap_levels=(.25,.225,.20)
        self.state.curriculum_episodes=198
        self.state.curriculum_completed=198
        self.state.curriculum_attempts=100
        self.state.curriculum_correct=89
        wb.record_hurdle_curriculum(self.env,torch.tensor([0,1]))
        self.assertEqual(self.state.curriculum_level,0)
        self.state.curriculum_correct=90
        self.state.curriculum_completed=200
        wb.record_hurdle_curriculum(self.env,torch.tensor([],dtype=torch.long))
        self.assertEqual(self.state.curriculum_level,1)
        self.assertEqual(self.state.course_spacing[0].item(),self.state.course_spacing[1].item())

    def test_curriculum_resets_only_requested_layout(self):
        self.env.cfg.hurdle_gap_levels=(.25,.225,.20)
        self.state.course_spacing[:]=.28
        self.state.curriculum_level=1
        writes={}
        for j in range(10):
            def write(pose,env_ids,j=j): writes[j]=(pose.clone(),env_ids.clone())
            self.env.scene[f'course_stick_{j}']=NS(write_root_pose_to_sim=write,
                                                write_root_velocity_to_sim=lambda *a,**kw:None)
        wb.reset_hurdle_layout(self.env,torch.tensor([0]))
        self.assertAlmostEqual(self.state.course_spacing[0].item(),.255,places=5)
        self.assertAlmostEqual(self.state.course_spacing[1].item(),.28,places=5)
        self.assertAlmostEqual(writes[9][0][0,0].item(),.5+9*.255,places=5)
        self.assertEqual(writes[9][1].tolist(),[0])
        self.assertEqual(self.state.course_level.tolist(),[1,0])

    def test_old_level_episodes_cannot_upgrade_next_level(self):
        self.env.cfg.hurdle_gap_curriculum=True
        self.state.curriculum_level=1
        self.state.course_level[:]=0
        self.state.current_stick_index[:]=10
        self.state.landing_attempts[:]=20
        self.state.correct_landings[:]=20
        wb.record_hurdle_curriculum(self.env,torch.tensor([0,1]))
        self.assertEqual(self.state.curriculum_episodes,0)
        self.assertEqual(self.state.curriculum_attempts,0)

    def test_target_distance_accounts_for_real_foot_length(self):
        vertices=torch.tensor(CONFIG['FOOT_SOLE_VERTICES'])
        sole=vertices[None].clone()
        sole[:,:,:,0]+=.35
        forward=torch.tensor([[1.,0.]])
        target=wb._course_step_targets(sole,torch.tensor([.64]),forward)
        torch.testing.assert_close(target,torch.full((1,2),.319),atol=.001,rtol=0)

    def test_paired_full_course_with_real_feet_all_gap_levels(self):
        vertices=torch.tensor(CONFIG['FOOT_SOLE_VERTICES'])
        local_center=.5*(vertices[:,:,0].amin(dim=1)+vertices[:,:,0].amax(dim=1))
        self.sole=vertices[None].repeat(2,1,1,1)
        self.sole[...,2]+=.04379
        self.sole[...,1]+=torch.tensor([-.065,.065])[None,:,None]
        for pitch in [.28,.255,.23]:
            self.state.course_spacing[:]=pitch
            self.state.current_stick_index[:]=0
            self.state.sticks_cleared_count[:]=0
            self.state.stick_hit_flags[:]=False
            self.state.stick_landing_reward_paid[:]=False
            self.state.crossing_command[:]=True
            self.state.following_step_command_stage[:]=0
            self.state.crossing_foot_index[:]=0
            self.state.following_foot_index[:]=1
            self.state.task_success[:]=False
            for foot in range(2):
                self.sole[:,foot,:,0]=vertices[foot,:,0]+.35-local_center[foot]
            self.contact[:]=1
            for stick in range(10):
                center=.5+(stick+.5)*pitch
                lead=int(self.state.crossing_foot_index[0])
                follow=1-lead
                self.env.scene['robot'].data.root_pos_w[:,0]=center-.1
                self.contact[:,lead]=0
                self.tick()
                self.sole[:,lead,:,0]=vertices[lead,:,0]+center-local_center[lead]
                self.contact[:,lead]=1
                self.tick()
                self.assertEqual(self.state.following_step_command_stage[0].item(),1)
                self.assertAlmostEqual(self.state.step_distance[0].item(),0.,places=5)
                self.contact[:,follow]=0
                self.tick()
                self.sole[:,follow,:,0]=vertices[follow,:,0]+center-local_center[follow]
                self.contact[:,follow]=1
                self.tick()
                self.assertEqual(self.state.current_stick_index[0].item(),stick+1)
            self.assertTrue(self.state.task_success.all())
            self.assertFalse(self.state.stick_hit_flags.any())

    def test_target_approach_cannot_pay_zero_net_rocking(self):
        self.arm_lead()
        self.sole[0,0,:,0]=.35
        self.tick()
        total=0.
        for x in [.36,.35,.36,.35]:
            self.sole[0,0,:,0]=x
            self.tick()
            total+=self.state.target_approach_reward_val[0].item()
        self.assertLessEqual(total,1e-6)

    def test_config_paired_course_fields(self):
        path=MDP.parent/'humanoid_robot_policy_rsl_rl_env_cfg.py'
        tree=ast.parse(path.read_text(encoding='utf-8'))
        class DummyCfg:
            InitialStateCfg=NS
            def __new__(cls,**kwargs): return NS(**kwargs)
        namespace=dict(CONFIG,sim_utils=NS(CuboidCfg=NS,RigidBodyPropertiesCfg=NS,
                       CollisionPropertiesCfg=NS,PreviewSurfaceCfg=NS),RigidObjectCfg=DummyCfg)
        scene=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='HumanoidRobotPolicySceneCfg')
        loop=next(n for n in scene.body if isinstance(n,ast.For))
        execute_nodes([ast.fix_missing_locations(ast.ClassDef(name='CourseScene',bases=[],keywords=[],body=[loop],decorator_list=[]))],path,namespace)
        for j in range(10):
            bar=getattr(namespace['CourseScene'],f'course_stick_{j}')
            self.assertTrue(bar.spawn.rigid_props.kinematic_enabled)
            self.assertEqual(bar.spawn.size,(.03,.80,.03))
            self.assertAlmostEqual(bar.init_state.pos[0],.5+j*.28)
        events=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='EventCfg')
        reset=next(n for n in events.body if isinstance(n,ast.Assign) and n.targets[0].id=='reset_base')
        namespace.update(EventTerm=NS,mdp=NS(reset_root_state_uniform=lambda:None))
        execute_nodes([reset],path,namespace)
        self.assertEqual(namespace['reset_base'].params['pose_range']['x'],(.08,.12))
        self.assertEqual(CONFIG['PHASE_5_BAR_EPISODE_PROBABILITY'],1.)

    def test_failed_terminal_sample_cannot_upgrade_completion_rate(self):
        self.env.termination_manager=NS(terminated=torch.tensor([True,False]))
        self.state.current_stick_index[0]=10
        wb.record_hurdle_curriculum(self.env,torch.tensor([0]))
        self.assertEqual(self.state.curriculum_completed,0)

    def test_real_termination_compute_orders_skip_before_completion(self):
        path=Path(r'F:/IsaacLab/source/isaaclab/isaaclab/managers/termination_manager.py')
        if not path.exists(): self.skipTest('Local Isaac Lab source is unavailable')
        tree=ast.parse(path.read_text(encoding='utf-8'))
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='TerminationManager')
        fn=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='compute')
        namespace={'torch':torch}
        execute_nodes([fn],path,namespace)
        self.arm_last_step(x=2.70,stick=9)
        self.contact[0,1]=0
        self.tick()
        self.contact[0,1]=1
        self.new_step()
        terms=[NS(func=lambda env:torch.zeros(2,dtype=torch.bool),params={},time_out=True),
               NS(func=lambda env:torch.zeros(2,dtype=torch.bool),params={},time_out=False),
               NS(func=wb.hurdle_skipped_gap,params={'feet_cfg':self.params['feet_cfg'],
                  'sensor_cfg':self.params['sensor_cfg'],'sole_vertices':CONFIG['FOOT_SOLE_VERTICES']},time_out=False),
               NS(func=wb.hurdle_course_completed,params={},time_out=False)]
        names=['time_out','bad_orientation','hurdle_skipped_gap','hurdle_course_completed']
        terminated=torch.zeros(2,dtype=torch.bool)
        dones=torch.zeros(2,4,dtype=torch.bool)
        manager=NS(_env=self.env,_term_cfgs=terms,_truncated_buf=torch.zeros(2,dtype=torch.bool),
                   _terminated_buf=terminated,_term_dones=dones,_last_episode_dones=dones.clone(),
                   terminated=terminated,active_terms=names,get_term=lambda name:dones[:,names.index(name)],
                   get_term_cfg=lambda name:terms[names.index(name)])
        self.env.termination_manager=manager
        self.assertTrue(namespace['compute'](manager)[0])
        self.assertFalse(manager.get_term('hurdle_skipped_gap')[0])
        self.assertTrue(manager.get_term('hurdle_course_completed')[0])
        self.assertEqual(wb.all_sticks_completed_reward(self.env)[0].item(),1.)

    def test_first_trigger_uses_world_coordinates(self):
        self.env.scene.env_origins[:, 0] = torch.tensor([0., 10.])
        robot = self.env.scene["robot"]
        robot.data.root_pos_w[:, 0] = torch.tensor([.35, 10.35])
        robot.data.root_quat_w = NS(torch=torch.tensor([[1., 0., 0., 0.]]).repeat(2, 1))
        self.sole[:, 0, :, 0] = torch.tensor([.42, 10.42])[:, None]
        self.sole[:, 1, :, 0] = torch.tensor([.35, 10.35])[:, None]
        with patch.object(wb, "_base_forward_and_sole_front", REAL_FORWARD):
            self.contact[:, 1] = 0
            self.tick()
            self.contact[:, 1] = 1
            self.tick()
        self.assertEqual(self.state.crossing_command.tolist(), [True, True])

    def test_first_reward_refreshes_current_measurements(self):
        self.new_step()
        self.assertEqual(wb.hurdle_forward_progress_reward(self.env).tolist(), [1., 1.])
        self.assertEqual(self.state.last_control_update_step[0], self.env.common_step_counter)

    def test_terminal_first_hit_penalty_is_not_lost(self):
        self.tick()
        self.sole[0, 0, :, 0] = .5
        self.sole[0, 0, :, 2] = .01
        self.new_step()
        self.env.termination_manager = NS(terminated=torch.tensor([True, False]),
                                           time_outs=torch.tensor([False, False]))
        self.assertEqual(wb.stick_failed_penalty(self.env)[0].item(), 1.)

    def test_terminal_last_clear_reward_is_not_lost(self):
        self.arm_last_step(x=2.70, stick=9)
        self.contact[0, 1] = 0
        self.tick()
        self.contact[0, 1] = 1
        self.new_step()
        self.env.termination_manager = NS(terminated=torch.tensor([False, False]),
                                           time_outs=torch.tensor([True, False]))
        self.assertEqual(wb.all_sticks_completed_reward(self.env)[0].item(), 1.)
        self.assertEqual(wb.stick_cleared_reward(self.env)[0].item(), 1.)

    def test_partial_update_preserves_other_environment_rewards(self):
        self.sole[1, 0, :, 0] = .5
        self.sole[1, 0, :, 2] = .065
        self.contact[1, 0] = 0
        self.tick()
        before = self.state.stick_over_clearance_reward_val[1].clone()
        self.state.last_control_update_step[0] = -1
        self.state.forward_progress_reward_val[0] = 0
        self.state.stick_over_clearance_reward_val[0] = 0
        wb._update_crossing_state_once(self.env, **self.params)
        self.assertEqual(self.state.forward_progress_reward_val[1].item(), 1.)
        torch.testing.assert_close(self.state.stick_over_clearance_reward_val[1], before)

    def test_safe_pitched_sole_is_not_a_collision(self):
        vertices = torch.tensor(CONFIG["FOOT_SOLE_VERTICES"][0])
        theta = math.radians(30)
        rotation = torch.tensor([[math.cos(theta), 0, math.sin(theta)], [0, 1, 0],
                                 [-math.sin(theta), 0, math.cos(theta)]])
        self.sole = torch.zeros(2, 2, len(vertices), 3)
        self.sole[0, 0] = vertices @ rotation.T + torch.tensor([.54, 0, .09])
        self.tick()
        self.assertFalse(self.state.stick_hit_flags[0, 0])

    def test_yawed_sole_without_overlap_is_not_a_collision(self):
        vertices = torch.tensor(CONFIG["FOOT_SOLE_VERTICES"][0])
        theta = math.radians(45)
        rotation = torch.tensor([[math.cos(theta), -math.sin(theta), 0],
                                 [math.sin(theta), math.cos(theta), 0], [0, 0, 1]])
        self.sole = torch.zeros(2, 2, len(vertices), 3)
        self.sole[0, 0] = vertices @ rotation.T + torch.tensor([.5, .47, .043790001])
        self.tick()
        self.assertFalse(self.state.stick_hit_flags[0, 0])

    def test_landing_outside_course_has_no_reward(self):
        self.sole[0, 0, :, 0] = .615
        self.sole[0, 0, :, 1] = .5
        self.land()
        self.assertEqual(wb.stick_landing_center_reward(self.env)[0].item(), 0.)

    def test_entire_sole_must_fit_landing_gap(self):
        self.sole[0, 0, :, 0] = torch.tensor([.46, .61, .61, .46])
        self.sole[0, 0, :, 1] = torch.tensor([-.03, -.03, .03, .03])
        self.land()
        self.assertEqual(wb.stick_landing_center_reward(self.env)[0].item(), 0.)

    def test_bypassing_course_does_not_clear_stick(self):
        self.arm_last_step(y=.5)
        self.land(1)
        self.assertFalse(self.state.stick_cleared_event[0])
        self.assertEqual(self.state.current_stick_index[0].item(), 0)

    def test_overshooting_next_gap_does_not_clear_stick(self):
        self.arm_last_step(x=.9)
        self.land(1)
        self.assertFalse(self.state.stick_cleared_event[0])
        self.assertEqual(self.state.current_stick_index[0].item(), 0)

    def test_each_foot_receives_landing_reward_once_per_gap(self):
        self.arm_lead()
        self.sole[0, 0, :, 0] = .615
        scores = []
        for _ in range(3):
            self.land()
            scores.append(wb.stick_landing_center_reward(self.env)[0].item())
        self.assertEqual(scores, [1., 0., 0.])
        self.assertEqual(self.state.current_stick_index[0].item(), 0)

    def test_forward_reward_does_not_pay_zero_net_rocking(self):
        scores = []
        for vx in [.4, .4, .4, -1.2]:
            self.velocity[:, 0] = vx
            self.tick()
            scores.append(wb.hurdle_forward_progress_reward(self.env)[0].item())
        self.assertLessEqual(sum(scores), 1e-6)

    def test_velocity_tracking_does_not_pay_zero_net_rocking(self):
        velocity = torch.tensor([[.4, 0, 0], [.4, 0, 0], [.4, 0, 0], [-1.2, 0, 0]])
        env = NS(scene={"robot": NS(data=NS(
            root_quat_w=NS(torch=torch.tensor([[1., 0, 0, 0]]).repeat(4, 1)),
            root_lin_vel_w=NS(torch=velocity)))},
            command_manager=NS(get_command=lambda _: torch.tensor([[.4, 0, 0]]).repeat(4, 1)))
        score = REWARDS["track_lin_vel_xy_yaw_frame_quadratic_relative"](
            env, "base_velocity", NS(name="robot"))
        self.assertLessEqual(score.mean().item(), 0.)


    def test_new_episode_resets_paid_landings_only_for_reset_environment(self):
        self.arm_lead()
        self.sole[0, 0, :, 0] = .615
        self.land()
        self.assertTrue(self.state.stick_landing_reward_paid[0, 0, 0])
        self.state.stick_landing_reward_paid[1] = True
        wb.reset_crossing_state(
            self.env, [0], "collisionless_wooden_bar", "wooden_bar", 2.0,
            4, .15, (3, 10), 1.0, 0.0, (0.0, 5.0),
        )
        self.assertFalse(self.state.stick_landing_reward_paid[0].any())
        self.assertTrue(self.state.stick_landing_reward_paid[1].all())
        self.arm_lead()
        self.land()
        self.assertEqual(wb.stick_landing_center_reward(self.env)[0].item(), 1.)

    def test_partial_update_preserves_other_environment_collision_pulse(self):
        self.sole[1, 0, :, 0] = .5
        self.sole[1, 0, :, 2] = .01
        self.tick()
        self.assertTrue(self.state.stick_collision_event[1])
        self.state.last_control_update_step[0] = -1
        wb._update_crossing_state_once(self.env, **self.params)
        self.assertTrue(self.state.stick_collision_event[1])

    def test_collision_geometry_matches_independent_clipping(self):
        # Independent polygon clipping validates tilted footprints, edge/corner
        # crossings and disjoint polygons against the vectorized implementation.
        import random
        def clip(poly, axis, bound, sign):
            result = []
            if not poly:
                return result
            previous = poly[-1]
            previous_distance = sign * (previous[axis] - bound)
            for current in poly:
                distance = sign * (current[axis] - bound)
                if (distance >= 0) != (previous_distance >= 0):
                    fraction = previous_distance / (previous_distance - distance)
                    result.append([a + fraction * (b - a)
                                   for a, b in zip(previous, current)])
                if distance >= 0:
                    result.append(current)
                previous, previous_distance = current, distance
            return result

        rng = random.Random(407)
        vertices = torch.tensor(CONFIG["FOOT_SOLE_VERTICES"][0], dtype=torch.float64)
        polygons, expected = [], []
        for _ in range(100):
            roll, pitch, yaw = [rng.uniform(-math.pi, math.pi) for _ in range(3)]
            rx = vertices.new_tensor([
                [1, 0, 0], [0, math.cos(roll), -math.sin(roll)],
                [0, math.sin(roll), math.cos(roll)],
            ])
            ry = vertices.new_tensor([
                [math.cos(pitch), 0, math.sin(pitch)], [0, 1, 0],
                [-math.sin(pitch), 0, math.cos(pitch)],
            ])
            rz = vertices.new_tensor([
                [math.cos(yaw), -math.sin(yaw), 0],
                [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1],
            ])
            position = vertices.new_tensor([
                rng.uniform(-.12, .12), rng.uniform(-.22, .22), rng.uniform(.01, .15),
            ])
            polygon = vertices @ (rz @ ry @ rx).T + position
            clipped = polygon.tolist()
            for axis, bound, sign in [(0, -.015, 1), (0, .015, -1),
                                      (1, -.15, 1), (1, .15, -1)]:
                clipped = clip(clipped, axis, bound, sign)
            expected.append(min(point[2] for point in clipped) if clipped else float("inf"))
            polygons.append(polygon)
        actual = wb._sole_min_z_over_rectangle(torch.stack(polygons), .015, .15)
        torch.testing.assert_close(
            actual, vertices.new_tensor(expected), atol=1e-7, rtol=1e-7,
        )


    def test_requested_reward_weights(self):
        path = MDP.parent / "humanoid_robot_policy_rsl_rl_env_cfg.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "RewardsCfg")
        namespace = dict(CONFIG, RewTerm=lambda **kw: NS(**kw), mdp=NS(
            stick_collision_penalty=wb.stick_collision_penalty,
            stick_landing_center_reward=wb.stick_landing_center_reward,
            track_lin_vel_xy_yaw_frame_quadratic_relative=REWARDS[
                "track_lin_vel_xy_yaw_frame_quadratic_relative"],
        ))
        execute_nodes([n for n in cls.body if isinstance(n, ast.Assign) and
                       any(isinstance(t, ast.Name) and t.id in {
                           "stick_collision", "stick_landing_center", "track_lin_vel_xy_exp"
                       } for t in n.targets)], path, namespace)
        self.assertIsNotNone(namespace["stick_collision"])
        self.assertEqual(namespace["stick_collision"].weight, -5.0)
        self.assertEqual(namespace["stick_landing_center"].weight, 25.0)
        self.assertEqual(namespace["track_lin_vel_xy_exp"].weight, 3.0)

    def bounds(self):
        self.assertTrue(hasattr(wb, "hurdle_out_of_bounds"), "missing course boundary")
        return wb.hurdle_out_of_bounds(
            self.env, self.params["feet_cfg"], self.params["sole_vertices"], .40, 2.785,
        )

    def test_lateral_root_escape_terminates_and_stops_progress_reward(self):
        self.env.scene["robot"].data.root_pos_w[0, 1] = .45
        self.assertTrue(self.bounds()[0])
        self.tick()
        self.assertEqual(wb.hurdle_forward_progress_reward(self.env)[0].item(), 0.)

    def test_whole_foot_must_stay_in_course(self):
        self.sole[0, 0, :, 1] = .41
        self.assertTrue(self.bounds()[0])

    def test_walking_beyond_final_gap_without_clearing_terminates(self):
        self.env.scene["robot"].data.root_pos_w[0, 0] = 2.9
        self.sole[0, :, :, 0] = 2.9
        self.assertTrue(self.bounds()[0])
        self.tick()
        self.assertEqual(wb.hurdle_forward_progress_reward(self.env)[0].item(), 0.)
        self.assertFalse(self.state.task_success[0])

    def test_start_and_valid_gap_remain_in_bounds(self):
        self.assertFalse(self.bounds().any())
        self.sole[0, :, :, 0] = .615
        self.assertFalse(self.bounds()[0])

    def test_completion_terminates_without_cancelling_grand_prize(self):
        self.assertTrue(hasattr(wb, "hurdle_course_completed"), "missing course completion")
        self.arm_last_step(x=2.70, stick=9)
        self.contact[0, 1] = 0
        self.tick()
        self.contact[0, 1] = 1
        self.new_step()
        dones = {name: torch.zeros(2, dtype=torch.bool) for name in [
            "bad_orientation", "low_base_height", "hurdle_out_of_bounds", "hurdle_course_completed"]}
        manager = NS(
            terminated=torch.zeros(2, dtype=torch.bool), time_outs=torch.zeros(2, dtype=torch.bool),
            active_terms=list(dones), get_term=lambda name: dones[name],
            get_term_cfg=lambda name: NS(time_out=False),
        )
        self.env.termination_manager = manager
        dones["hurdle_course_completed"][:] = wb.hurdle_course_completed(self.env)
        manager.terminated |= dones["hurdle_course_completed"]
        self.assertTrue(manager.terminated[0])
        self.assertEqual(wb.all_sticks_completed_reward(self.env)[0].item(), 1.)

    def test_two_first_collisions_in_same_step_charge_each_stick_once(self):
        self.sole[0, 0, :, 0] = .5
        self.sole[0, 1, :, 0] = .73
        self.sole[0, :, :, 2] = .01
        self.tick()
        self.assertEqual(wb.stick_collision_penalty(self.env)[0].item(), 2.)
        self.tick()
        self.assertEqual(wb.stick_collision_penalty(self.env)[0].item(), 0.)


def load_tests(loader, tests, pattern):
    return loader.loadTestsFromTestCase(RewardRegressionTests)


if __name__ == "__main__":
    unittest.main()
