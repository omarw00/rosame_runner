"""
MA-ROSAME — single-file implementation.

Algorithm: docs/ma_rosame_algorithm.md

Key difference from ma_rosame_module/: Phase 2 trains on JOINT multi-agent
steps using the product-of-complements formula — consistent with how Phase 3
scores steps. The old module trained only on single-agent steps.
"""

import ma_sam_path  # noqa: F401 — adds libs/ma-sam to sys.path before any sam_learning import
import torch
import torch.nn.functional as F
import torch.optim as optim
from pathlib import Path

from pddl_plus_parser.lisp_parsers import DomainParser, ProblemParser, TrajectoryParser
from pddl_plus_parser.models import MultiAgentObservation
from sam_learning.learners import MASAMPlus
from rosame_runner import Rosame_Runner

_NOP_NAMES = {"nop", "dummy-add-predicate-action", "dummy-del-predicate-action"}

# Pipeline bookkeeping predicates excluded from loss/scoring. dummy-additional-predicate
# is toggled by whichever agent executes dummy-add/del-predicate-action; since _compose
# skips those actions entirely (_NOP_NAMES), a real change to this flag from a concurrent
# agent inflates MSE with noise unrelated to the action actually being scored. Same
# exclusion as ma_rosame_module/noise_filter.py.
_IGNORE_PREDS = {"dummy-additional-predicate"}


# ── Phase 2: neural helpers ───────────────────────────────────────────────────

def _encode(rr, state) -> torch.Tensor:
    """Encode a PDDL state as a binary proposition vector."""
    active = set()
    for _, preds in state.state_predicates.items():
        for p in preds:
            canon = rr.check_predicate(p.untyped_representation[1:-1])
            if canon is not None:
                active.add(canon)
    return torch.tensor([1.0 if p in active else 0.0 for p in rr.rosame.propositions])


def _keep_mask(rr) -> torch.Tensor:
    """Boolean mask over rr.rosame.propositions excluding _IGNORE_PREDS, applied at
    loss/scoring time (not inside _encode) so s1/_compose's tensor sizes stay unchanged."""
    return torch.tensor([
        not any(p.startswith(ignored) for ignored in _IGNORE_PREDS)
        for p in rr.rosame.propositions
    ])


def _compose(rr, s1: torch.Tensor, component):
    """
    Product-of-complements over all active agents → predicted s2.

    joint_add[p] = 1 - Π_i (1 - add_i[p])
    joint_del[p] = 1 - Π_i (1 - del_i[p])
    predicted    = s1 * (1 - joint_del) + (1 - s1) * joint_add
    """
    n = len(rr.rosame.propositions)
    comp_add = torch.ones(n)
    comp_del = torch.ones(n)
    valid = False
    for action in component.grounded_joint_action.operational_actions:
        if action.name.lower() in _NOP_NAMES:
            continue
        idx = rr.check_action(action.__str__()[1:-1])
        if idx is None:
            continue
        valid = True
        _, add, dlt = rr.rosame.build([idx])
        comp_add = comp_add * (1.0 - add[0])
        comp_del = comp_del * (1.0 - dlt[0])
    if not valid:
        return None
    joint_add = 1.0 - comp_add
    joint_del = 1.0 - comp_del
    return s1 * (1.0 - joint_del) + (1.0 - s1) * joint_add


def _count_active(component) -> int:
    """Count non-NOP actions in a joint step."""
    return sum(
        1 for action in component.grounded_joint_action.operational_actions
        if action.name.lower() not in _NOP_NAMES
    )


def _train_phase(rr, optimizer, pairs, epochs, target_size, keep_per_pair=None):
    """Train only on steps where exactly `target_size` agents are active (non-NOP)."""
    for i, (_, problem, ma_obs) in enumerate(pairs):
        rr.problem = problem
        rr.ground_new_trajectory()
        mask = _keep_mask(rr)
        keep = set(keep_per_pair[i]) if keep_per_pair else None
        indices = [
            j for j, comp in enumerate(ma_obs.components)
            if (keep is None or j in keep) and _count_active(comp) == target_size
        ]
        if not indices:
            continue
        print(f"  [{i + 1}/{len(pairs)}] {len(indices)} steps (active={target_size})")
        for epoch in range(epochs):
            total, count = 0.0, 0
            for j in indices:
                comp = ma_obs.components[j]
                s1   = _encode(rr, comp.previous_state)
                s2   = _encode(rr, comp.next_state)
                pred = _compose(rr, s1, comp)
                if pred is None:
                    continue
                optimizer.zero_grad()
                loss = F.mse_loss(pred[mask], s2[mask])
                loss.backward()
                optimizer.step()
                total += loss.item()
                count += 1
            if (epoch % 10 == 0 or epoch == epochs - 1) and count:
                print(f"Epoch {epoch} RESULTS: Average loss: {total / count:.10f}")


def _train(rr, optimizer, pairs, epochs, keep_per_pair=None, phases=(1, 2)):
    """
    Phase 2 training — curriculum: single-agent steps first (Phase A), then
    two-agent joint steps (Phase B), continuing from Phase A's weights.
    `epochs` is the total budget: split evenly when both phases run, or given
    entirely to the one phase requested via `phases`.

    keep_per_pair: list of index lists (one per trajectory); None = all steps.
    phases: which target-sizes to train on, in order — (1, 2) for the full
        curriculum, (1,) for single-agent only, (2,) for two-agent only.
    """
    if len(phases) == 1:
        budget = {phases[0]: epochs}
    else:
        phase_a_epochs = max(1, epochs // 2)
        budget = {1: phase_a_epochs, 2: epochs - phase_a_epochs}

    labels = {1: "A: single-agent", 2: "B: two-agent"}
    for target_size in phases:
        e = budget[target_size]
        print(f"  -- Phase {labels[target_size]} steps ({e} epochs) --")
        _train_phase(rr, optimizer, pairs, e, target_size=target_size, keep_per_pair=keep_per_pair)


# ── Phase 3: scoring ──────────────────────────────────────────────────────────

def _score(rr, pairs_clean, pairs_noisy, threshold):
    """
    Score every step using clean s_t (previous_state) + noisy s_{t+1} (next_state).

    Using the clean previous_state as reference avoids compounding: since the
    trajectory parser chains next_state -> previous_state for the following step,
    scoring straight off the noisy observation would mean a corrupted s_{t+1}
    becomes a corrupted s_t for the next step, masking the noise signal. Pairing
    each noisy outcome against its true predecessor isolates outcome noise cleanly.

    Returns kept index lists (per trajectory) and prints drop counts.
    """
    keep_per_pair = []
    with torch.no_grad():
        for (_, problem, clean_obs), (_, _, noisy_obs) in zip(pairs_clean, pairs_noisy):
            rr.problem = problem
            rr.ground_new_trajectory()
            mask = _keep_mask(rr)
            kept = []
            for j, (clean_comp, noisy_comp) in enumerate(zip(clean_obs.components, noisy_obs.components)):
                s1   = _encode(rr, clean_comp.previous_state)   # clean s_t
                s2   = _encode(rr, noisy_comp.next_state)        # noisy s_{t+1}
                pred = _compose(rr, s1, noisy_comp)
                mse  = F.mse_loss(pred[mask], s2[mask]).item() if pred is not None else 0.0
                if mse <= threshold:
                    kept.append(j)
            dropped = len(noisy_obs.components) - len(kept)
            print(f"  {dropped:3d} dropped / {len(noisy_obs.components)} total")
            keep_per_pair.append(kept)
    return keep_per_pair


def _filter_obs(problem, agents, ma_obs, kept_indices):
    """Slice an already-parsed MultiAgentObservation to keep only steps at kept_indices."""
    kept_set = set(kept_indices)
    filtered = MultiAgentObservation(executing_agents=agents)
    filtered.add_problem_objects(problem.objects)
    for i, comp in enumerate(ma_obs.components):
        if i in kept_set:
            filtered.components.append(comp)
    return filtered


# ── MARosame ──────────────────────────────────────────────────────────────────

class MARosame:
    def __init__(
        self,
        domain_path,
        agents: list[str],
        noise_threshold: float = 0.1,
        epochs: int = 600,
        phases: tuple = (1, 2),
    ):
        self.domain_path     = Path(domain_path)
        self.agents          = agents
        self.noise_threshold = noise_threshold
        self.epochs          = epochs
        self.phases          = phases
        self.domain = DomainParser(self.domain_path, partial_parsing=True).parse_domain()

    def fit(self, trajectory_paths, problem_paths, clean_trajectory_paths=None):
        """
        Run the full MA-ROSAME pipeline.

        trajectory_paths: the (possibly noisy) trajectories to learn from.
        clean_trajectory_paths: optional ground-truth trajectories, same order and
            length as trajectory_paths. When given, scoring uses the clean
            previous_state as reference for s_t and the (possibly noisy)
            next_state for s_{t+1} — this avoids compounding noise across steps.
            When omitted, trajectory_paths are used for both roles (no dual
            scoring; e.g. the clean/no-noise case).

        Returns (learned_domain, report, macro_mapping).
        """
        trajectory_paths = [Path(p) for p in trajectory_paths]
        problem_paths    = [Path(p) for p in problem_paths]
        clean_trajectory_paths = (
            [Path(p) for p in clean_trajectory_paths]
            if clean_trajectory_paths is not None else trajectory_paths
        )

        # ── Phase 1: Parse (both noisy and clean-reference observations) ──────
        print("Phase 1: Parsing…")
        pairs_noisy, pairs_clean = [], []
        for tp, ctp, pp in zip(trajectory_paths, clean_trajectory_paths, problem_paths):
            problem   = ProblemParser(problem_path=pp, domain=self.domain).parse_problem()
            noisy_obs = TrajectoryParser(self.domain, problem).parse_trajectory(
                tp, executing_agents=self.agents
            )
            clean_obs = (
                noisy_obs if ctp == tp else
                TrajectoryParser(self.domain, problem).parse_trajectory(
                    ctp, executing_agents=self.agents
                )
            )
            pairs_noisy.append((tp, problem, noisy_obs))
            pairs_clean.append((ctp, problem, clean_obs))
        print(f"  {len(pairs_noisy)} trajectories, "
              f"{sum(len(m.components) for _, _, m in pairs_noisy)} total steps")

        # ── Phase 2: ROSAME curriculum training (single pass) ─────────────────
        # Phase A (single-agent steps) then Phase B (two-agent steps), each getting
        # half of self.epochs — see _train(). No separate retrain-on-clean pass:
        # noise filtering happens once, after this training completes.
        rr = Rosame_Runner(self.domain_path)
        rr.add_problem(pairs_noisy[0][1])
        # action_schemas is a plain list (not nn.ModuleList) so .parameters() is empty;
        # collect MLP weights from each schema directly — same approach as learn_rosame()
        params = [p for schema in rr.rosame.action_schemas for p in schema.parameters()]
        optimizer = optim.Adam(params, lr=1e-3)

        print(f"\nPhase 2: {self.epochs} epochs total (phases={self.phases})")
        _train(rr, optimizer, pairs_noisy, self.epochs, phases=self.phases)

        # ── Phase 3: Final filter ─────────────────────────────────────────────
        print("\nPhase 3: Final scoring (clean s_t vs noisy s_t+1)…")
        final_keep  = _score(rr, pairs_clean, pairs_noisy, self.noise_threshold)
        cleaned_obs = []
        for (tp, problem, noisy_obs), kept in zip(pairs_noisy, final_keep):
            filtered = _filter_obs(problem, self.agents, noisy_obs, kept)
            cleaned_obs.append(filtered)
            print(f"  {tp.name}: {len(kept)}/{len(noisy_obs.components)} steps kept")

        # ── Phase 4: MA-SAM+ ───────────────────────────────────────────────────
        print("\nPhase 4: MA-SAM+…")
        sam_learner = MASAMPlus(self.domain)
        learned_domain, report, macro_mapping = (
            sam_learner.learn_combined_action_model_with_macro_actions(cleaned_obs)
        )

        print("\nPer-action model summary:")
        print(f"  {'Action':<35} {'Pre':>3}  {'Eff':>3}")
        print(f"  {'-'*35}  {'-'*3}  {'-'*3}")
        for aname, action in sorted(learned_domain.actions.items()):
            print(f"  {aname:<35} {len(action.preconditions_str_set):>3}  "
                  f"{len(action.discrete_effects):>3}")

        return learned_domain, report, macro_mapping

    def export(self, learned_domain, path: Path):
        """Write the learned domain PDDL to a file."""
        Path(path).write_text(learned_domain.to_pddl())
