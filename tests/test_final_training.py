import sys
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'hpc'))
from final_training import SuccessBatch, learn_until_success, evaluate_deterministic

class FinalTrainingTests(unittest.TestCase):
    def test_first_episode_only_and_timeout_is_failure(self):
        batch = SuccessBatch(5)
        batch.record([0, 0, 1, 7], [False, True, True, True])
        batch.record([2, 3], [True, True])
        self.assertEqual(batch.successes, 3)
        self.assertEqual(batch.rate, .6)
        self.assertFalse(batch.complete)
        batch.record([4], [True])
        self.assertEqual(batch.rate, .8)
        self.assertTrue(batch.complete)

    def runner(self):
        calls = []
        writer = NS(add_scalar=lambda *a: calls.append(('scalar', a)))
        logger = NS(writer=writer, init_logging_writer=lambda: calls.append('init'),
                    stop_logging_writer=lambda: calls.append('close'))
        r = NS(current_learning_iteration=0, logger=logger, env=NS(unwrapped=NS(
            _fixed_stick_state=NS(curriculum=NS(level=5)))))
        def learn(num_learning_iterations, init_at_random_ep_len=False):
            logger.init_logging_writer()
            calls.append(('learn', r.current_learning_iteration, num_learning_iterations))
            r.current_learning_iteration += num_learning_iterations - 1
            logger.stop_logging_writer()
        r.learn = learn
        r.save = lambda path, infos=None: calls.append(('save', Path(path).name, infos))
        return r, calls

    def test_stop_at_eighty_and_do_not_repeat_iteration(self):
        r, calls = self.runner()
        rates = iter([.79, .8])
        result = learn_until_success(r, 200, '.', interval=50,
            evaluate=lambda: {'rate': next(rates), 'episodes': 200})
        self.assertTrue(result['target_reached'])
        self.assertEqual([c for c in calls if isinstance(c, tuple) and c[0]=='learn'],
                         [('learn', 0, 50), ('learn', 50, 50)])
        self.assertEqual(calls.count('init'), 1)
        self.assertEqual(calls.count('close'), 1)
        self.assertTrue(any(c[:2]==('save', 'model_success_80.pt') for c in calls if isinstance(c,tuple)))

    def test_small_sample_does_not_stop_and_budget_is_exact(self):
        r, calls = self.runner()
        result = learn_until_success(r, 55, '.', interval=50,
            evaluate=lambda: {'rate': 1., 'episodes': 2})
        self.assertFalse(result['target_reached'])
        self.assertEqual([c[2] for c in calls if isinstance(c,tuple) and c[0]=='learn'], [50,5])

    def test_evaluation_ignores_stale_outcomes_and_restores_training(self):
        import torch
        term = NS(noise='training_noise')
        state = NS(curriculum=NS(level=5, parameters=(0.,.03), auto_advance=False, outcomes=[True]),
                   clean_stats=NS(outcomes=[False]))
        raw = NS(_fixed_stick_state=state, common_step_counter=17, max_episode_length=4,
                 observation_manager=NS(_group_obs_term_cfgs={'policy':[term]}))
        obs = torch.zeros(2,49)
        class Env:
            unwrapped=raw
            num_envs=2
            tick=0
            def reset(self):
                self.tick=0
                return obs, {}
            def step(self, action):
                self.tick+=1
                raw.common_step_counter+=1
                state.clean_stats.outcomes.append(True)
                state.curriculum.outcomes.append(False)
                dones=torch.zeros(2,dtype=torch.bool)
                env_id=0 if self.tick<3 else 1
                if self.tick>1: dones[env_id]=True
                return obs,None,dones,{'fixed_stick_outcome':{
                    'env_ids':torch.tensor([env_id]),
                    'clean_success':torch.tensor([self.tick!=2])}}
        class Policy:
            def __call__(self, observations):
                self_test.assertIsNone(term.noise)
                torch.rand(1)
                return torch.zeros(2,12)
            def reset(self, dones=None): pass
        self_test=self
        r=NS(env=Env(),device='cpu',alg=NS(train_mode=lambda:None),
             get_inference_policy=lambda device:Policy())
        rng=torch.get_rng_state()
        result=evaluate_deterministic(r,3)
        self.assertEqual(result['successes'],1)
        self.assertEqual(result['episodes'],3)
        self.assertEqual(raw.common_step_counter,17)
        self.assertEqual(state.clean_stats.outcomes,[False])
        self.assertEqual(state.curriculum.outcomes,[True])
        self.assertEqual(term.noise,'training_noise')
        self.assertTrue(torch.equal(torch.get_rng_state(),rng))

if __name__ == '__main__': unittest.main()
