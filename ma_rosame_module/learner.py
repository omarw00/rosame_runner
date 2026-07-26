import ma_sam_path  # noqa: F401 – adds libs/ma-sam to sys.path
from collections import defaultdict
from pathlib import Path

from pddl_plus_parser.lisp_parsers import DomainParser, ProblemParser, TrajectoryParser
from pddl_plus_parser.models import Observation
from sam_learning.learners import MASAMPlus

from rosame_runner import Rosame_Runner
from ma_rosame_module.noise_filter import filter_observation
from ma_rosame_module.adapter import to_multi_agent_observation

_NOP_NAMES = {"nop", "dummy-add-predicate-action", "dummy-del-predicate-action"}


def _to_single_agent_observation(ma_obs, kept_indices=None) -> Observation:
    """Extract a pseudo-single-agent Observation from a MultiAgentObservation.

    Per step: pick the first operational action whose name is not a NOP.
    If kept_indices is given, only include those step positions.
    """
    include = set(kept_indices) if kept_indices is not None else None
    sa_obs = Observation()
    sa_obs.add_problem_objects(ma_obs.grounded_objects)
    for i, component in enumerate(ma_obs.components):
        if include is not None and i not in include:
            continue
        for action in component.grounded_joint_action.operational_actions:
            if action.name.lower() not in _NOP_NAMES:
                sa_obs.add_component(
                    previous_state=component.previous_state,
                    call=action,
                    next_state=component.next_state,
                )
                break
    return sa_obs


class MARosame:
    def __init__(self, domain_path, agents: list[str], noise_threshold: float = 0.1, epochs: int = 100):
        self.domain_path = Path(domain_path)
        self.agents = agents
        self.noise_threshold = noise_threshold
        self.epochs = epochs
        self.domain = DomainParser(self.domain_path, partial_parsing=True).parse_domain()

    def fit(self, trajectory_paths: list[Path], problem_paths: list[Path]):
        """Run the full MA-ROSAME pipeline.

        Each trajectory_path must have a corresponding problem_path at the same index.
        Returns (learned_domain, report, macro_mapping).
        """
        trajectory_paths = [Path(p) for p in trajectory_paths]
        problem_paths = [Path(p) for p in problem_paths]

        # Phase 1 — parse each (problem, trajectory) pair as multi-agent observations
        pairs = []
        for traj_path, prob_path in zip(trajectory_paths, problem_paths):
            problem = ProblemParser(problem_path=prob_path, domain=self.domain).parse_problem()
            ma_obs = TrajectoryParser(self.domain, problem).parse_trajectory(
                traj_path, executing_agents=self.agents
            )
            pairs.append((traj_path, problem, ma_obs))

        # Phase 2 — two-pass ROSAME training
        #
        # Pass 1 (half epochs, all steps): Bootstrap the model on all data so it
        # learns the dominant clean dynamics. At 10% noise the clean signal is 9×
        # stronger, so the model learns mostly clean transitions.
        #
        # Intermediate scoring: use the pass-1 model to identify which steps are
        # "likely noisy" (high MSE). These are excluded from pass 2.
        #
        # Pass 2 (half epochs, clean steps only): Retrain on the filtered data.
        # Now the model has no noisy signal to fit, so clean-step MSE drops and
        # noisy-step MSE (when scored after) will be distinctly higher.

        rosame_runner = Rosame_Runner(self.domain_path)
        rosame_runner.add_problem(pairs[0][1])  # one-time model initialisation

        pass1_epochs = max(1, self.epochs // 2)
        pass2_epochs = self.epochs - pass1_epochs

        print(f"Pass 1: {pass1_epochs} epochs on all steps")
        for i, (_, problem, ma_obs) in enumerate(pairs):
            rosame_runner.problem = problem
            rosame_runner.ground_new_trajectory()
            sa_obs = _to_single_agent_observation(ma_obs)
            print(f"  [{i + 1}/{len(pairs)}] {len(sa_obs.components)} steps ({problem.name})")
            rosame_runner.learn_rosame(sa_obs, epochs=pass1_epochs)

        # Intermediate scoring — collect clean indices per trajectory
        print("Intermediate scoring (pass-1 model)…")
        clean_indices_per_pair = []
        for _, problem, ma_obs in pairs:
            rosame_runner.problem = problem
            rosame_runner.ground_new_trajectory()
            _, clean_indices = filter_observation(rosame_runner, ma_obs, self.noise_threshold)
            clean_indices_per_pair.append(clean_indices)
            dropped = len(ma_obs.components) - len(clean_indices)
            print(f"  {dropped:3d} steps flagged as noisy out of {len(ma_obs.components)}")

        print(f"Pass 2: {pass2_epochs} epochs on clean steps only")
        for i, ((_, problem, ma_obs), clean_indices) in enumerate(zip(pairs, clean_indices_per_pair)):
            rosame_runner.problem = problem
            rosame_runner.ground_new_trajectory()
            sa_obs_clean = _to_single_agent_observation(ma_obs, kept_indices=clean_indices)
            print(f"  [{i + 1}/{len(pairs)}] {len(sa_obs_clean.components)}/{len(ma_obs.components)} clean steps")
            rosame_runner.learn_rosame(sa_obs_clean, epochs=pass2_epochs)

        # Phases 3–5 — final scoring, filter, and rebuild each trajectory
        # Use rosame_runner.problem + ground_new_trajectory() (NOT add_problem())
        # to preserve the weights learned in both training passes.
        print("Final scoring (pass-2 model)…")
        cleaned_observations = []
        for traj_path, problem, ma_obs in pairs:
            rosame_runner.problem = problem
            rosame_runner.ground_new_trajectory()
            _, kept_indices = filter_observation(rosame_runner, ma_obs, self.noise_threshold)
            print(f"  {traj_path.name}: {len(kept_indices)}/{len(ma_obs.components)} steps kept")
            ma_obs_clean = to_multi_agent_observation(
                self.domain, problem, traj_path, self.agents, kept_indices
            )
            cleaned_observations.append(ma_obs_clean)

        # Phase 6 — symbolic learning with MA-SAM+
        learner = MASAMPlus(self.domain)
        learned_domain, report, macro_mapping = (
            learner.learn_combined_action_model_with_macro_actions(cleaned_observations)
        )
        return learned_domain, report, macro_mapping

    def export(self, learned_domain, path: Path):
        """Write learned_domain to a PDDL file."""
        Path(path).write_text(learned_domain.to_pddl())
