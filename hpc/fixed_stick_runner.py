"""Save clean-crossing readiness beside each policy checkpoint."""
from rsl_rl.runners import OnPolicyRunner


class FixedStickRunner(OnPolicyRunner):
    def save(self, path, infos=None):
        saved_infos = dict(infos or {})
        if hasattr(self, 'final_evaluation'):
            saved_infos['final_evaluation'] = dict(self.final_evaluation)
        state = self.env.unwrapped._fixed_stick_state
        saved_infos['fixed_stick_training'] = state.clean_stats.summary()
        saved_infos['fixed_stick_control'] = state.control_settings()
        if getattr(state, 'curriculum', None) is not None:
            saved_infos['direct_crossing_curriculum'] = state.curriculum.summary()
        return super().save(path, infos=saved_infos)
