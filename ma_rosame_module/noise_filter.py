import statistics
from collections import defaultdict

import torch
import torch.nn.functional as F

# Pipeline bookkeeping predicates to exclude from scoring. dummy-additional-predicate
# is a single global flag toggled by whichever agent executes dummy-add/del-predicate-
# action; under single-agent training a different (concurrent) agent can flip it
# without the scored action ever seeing why, making it structurally unpredictable
# from that action alone. Already excluded the same way in frequency counting.
_IGNORE_PREDS = {"dummy-additional-predicate"}

_NOP_NAMES = {"nop", "dummy-add-predicate-action", "dummy-del-predicate-action"}


def score_steps(rosame_runner, observation) -> list[float]:
    """Returns MSE score per step. High score = likely noisy."""
    scores = []
    prop_list = list(rosame_runner.rosame.propositions)
    n_props = len(prop_list)
    keep_mask = torch.tensor([
        not any(p.startswith(ignored) for ignored in _IGNORE_PREDS)
        for p in prop_list
    ])

    with torch.no_grad():
        for component in observation.components:
            joint_action = component.grounded_joint_action
            operational_actions = joint_action.operational_actions

            # Encode previous and next states as binary proposition vectors
            pre_state = {
                rosame_runner.check_predicate(pred.untyped_representation[1:-1])
                for _, val in component.previous_state.state_predicates.items()
                for pred in val
            }
            next_state = {
                rosame_runner.check_predicate(pred.untyped_representation[1:-1])
                for _, val in component.next_state.state_predicates.items()
                for pred in val
            }

            s1 = torch.tensor(
                [1.0 if p in pre_state else 0.0 for p in prop_list]
            )
            s2 = torch.tensor(
                [1.0 if p in next_state else 0.0 for p in prop_list]
            )

            # Accumulate product-of-complements over all agent actions in this step
            complement_add = torch.ones(n_props)
            complement_del = torch.ones(n_props)

            valid = False
            for action in operational_actions:
                action_str = action.__str__()[1:-1]
                idx = rosame_runner.check_action(action_str)
                if idx is None:
                    continue
                valid = True
                _, addeff, deleff = rosame_runner.rosame.build([idx])
                complement_add *= (1.0 - addeff[0])
                complement_del *= (1.0 - deleff[0])

            if not valid:
                scores.append(0.0)
                continue

            joint_add = 1.0 - complement_add
            joint_del = 1.0 - complement_del

            predicted_s2 = s1 * (1.0 - joint_del) + (1.0 - s1) * joint_add
            score = F.mse_loss(predicted_s2[keep_mask], s2[keep_mask]).item()
            scores.append(score)

    return scores


def filter_observation(rosame_runner, observation, threshold: float):
    """Returns (observation, kept_indices). kept_indices are the clean step positions."""
    scores = score_steps(rosame_runner, observation)
    kept_indices = [i for i, s in enumerate(scores) if s <= threshold]
    return observation, kept_indices


def _action_group_key(component, use_symbol_only: bool = False) -> tuple:
    """
    Group key for a joint step: sorted tuple of active (non-NOP) action identifiers.

    rosame.build([idx]) predicts purely from the trained action-schema weights and
    the grounded action index — it does not depend on the current state at all. So
    every occurrence of the SAME grounded action produces an identical predicted
    effect, and when the surrounding (masked) state also coincides, the whole MSE
    score is reproduced exactly. This makes the score distribution naturally
    clustered by grounded action rather than smoothly spread — grouping by the
    action that was actually executed (rather than pooling across a trajectory or
    the whole dataset) compares each step only against its true peers.

    use_symbol_only=True groups by action name alone (dropping object arguments),
    for the fallback when a specific grounding has too few occurrences on its own.
    """
    actions = []
    for action in component.grounded_joint_action.operational_actions:
        if action.name.lower() in _NOP_NAMES:
            continue
        actions.append(action.name.lower() if use_symbol_only else action.__str__()[1:-1])
    return tuple(sorted(actions))


def compute_adaptive_thresholds(
    rosame_runner, observation, k: float = 10.0, min_group_size: int = 3
):
    """
    Per-step adaptive thresholds: median + k*MAD computed within the group of
    steps sharing the same grounded action. Groups with fewer than
    min_group_size occurrences fall back to grouping by action symbol alone
    (ignoring object arguments), which pools more samples at the cost of
    comparing across different object bindings of the same action schema.

    Returns (scores, thresholds, diagnostics) where diagnostics reports how
    many grounded-action groups existed, how many were usable directly, and
    how many steps needed the symbol-level fallback.
    """
    scores = score_steps(rosame_runner, observation)

    grounded_groups = defaultdict(list)
    symbol_groups = defaultdict(list)
    for i, component in enumerate(observation.components):
        grounded_groups[_action_group_key(component, use_symbol_only=False)].append(i)
        symbol_groups[_action_group_key(component, use_symbol_only=True)].append(i)

    def _median_mad_threshold(indices):
        group_scores = [scores[i] for i in indices]
        median = statistics.median(group_scores)
        mad = statistics.median([abs(s - median) for s in group_scores]) or 1e-9
        return median + k * mad

    thresholds = [None] * len(scores)
    used_grounded_keys = set()
    for key, indices in grounded_groups.items():
        if len(indices) >= min_group_size:
            threshold = _median_mad_threshold(indices)
            for i in indices:
                thresholds[i] = threshold
            used_grounded_keys.add(key)

    fallback_steps = [i for i, t in enumerate(thresholds) if t is None]
    fallback_symbol_keys = set()
    for key, indices in symbol_groups.items():
        unresolved = [i for i in indices if thresholds[i] is None]
        if not unresolved:
            continue
        fallback_symbol_keys.add(key)
        if len(indices) >= 2:
            threshold = _median_mad_threshold(indices)
        else:
            # last resort: no peers even at symbol level — use the trajectory-wide stat
            threshold = _median_mad_threshold(list(range(len(scores))))
        for i in unresolved:
            thresholds[i] = threshold

    diagnostics = {
        "n_grounded_groups": len(grounded_groups),
        "n_grounded_groups_used": len(used_grounded_keys),
        "n_steps_fallback": len(fallback_steps),
        "n_symbol_groups_used_as_fallback": len(fallback_symbol_keys),
    }
    return scores, thresholds, diagnostics


def _sole_action_name(component):
    """Name of the single non-NOP action in this step, or None if 0 or 2+ are active.

    Mirrors the eligibility rule in frequency_filter.learn_with_frequency: only steps
    with exactly one real action contribute to an action's frequency counts.
    """
    real = [a for a in component.grounded_joint_action.operational_actions
            if a.name.lower() not in _NOP_NAMES]
    return real[0].name.lower() if len(real) == 1 else None


def build_drop_budget(observations, min_observations: int) -> dict:
    """Max steps that may be dropped per action without starving frequency counting.

    Frequency counting only trusts an action once it has >= min_observations
    single-agent examples (see frequency_filter.learn_with_frequency); below that it
    falls back to the raw MA-SAM+ estimate, which on sparse actions is markedly worse.
    Filtering a sparse action too aggressively therefore costs more accuracy than the
    noise it removes. Budget = however many observations exceed the minimum.

    observations: list of MultiAgentObservation, pooled exactly as frequency counting
    will see them (counts must span all trajectories, since it pools across them).
    """
    counts = defaultdict(int)
    for observation in observations:
        for component in observation.components:
            aname = _sole_action_name(component)
            if aname is not None:
                counts[aname] += 1
    return {aname: max(0, n - min_observations) for aname, n in counts.items()}


def filter_observation_adaptive(
    rosame_runner, observation, k: float = 10.0, min_group_size: int = 3,
    drop_budget: dict = None,
):
    """Returns (observation, kept_indices, diagnostics) using per-grounded-action
    adaptive thresholds instead of a single fixed or trajectory-pooled threshold.

    drop_budget: optional dict from build_drop_budget(), MUTATED IN PLACE so a caller
    can enforce one budget across successive per-trajectory calls. Flagged single-agent
    steps are dropped most-suspicious-first until that action's budget is exhausted;
    remaining flagged steps for that action are then kept rather than starving the
    downstream frequency counter. Flagged multi-agent steps are unbudgeted — they don't
    contribute to frequency counts, so dropping them costs nothing there.
    """
    scores, thresholds, diagnostics = compute_adaptive_thresholds(
        rosame_runner, observation, k=k, min_group_size=min_group_size
    )
    flagged = [i for i, (s, t) in enumerate(zip(scores, thresholds)) if s > t]

    if drop_budget is None:
        dropped = set(flagged)
        diagnostics["n_budget_blocked"] = 0
    else:
        dropped = set()
        budget_blocked = 0
        # most-suspicious-first, so a limited budget is spent on the worst offenders
        for i in sorted(flagged, key=lambda i: scores[i], reverse=True):
            aname = _sole_action_name(observation.components[i])
            if aname is None:                     # multi-agent: unbudgeted
                dropped.add(i)
            elif drop_budget.get(aname, 0) > 0:
                drop_budget[aname] -= 1
                dropped.add(i)
            else:
                budget_blocked += 1
        diagnostics["n_budget_blocked"] = budget_blocked

    kept_indices = [i for i in range(len(scores)) if i not in dropped]
    return observation, kept_indices, diagnostics
