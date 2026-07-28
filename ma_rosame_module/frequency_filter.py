"""
Layer 2 noise defence: frequency-based action model learning.

Priority 1 — single-agent steps (most reliable):
    Count how often each lifted literal appears as pre/add/del across all
    single-agent observations for an action schema.  Keep it if frequency >=
    threshold.  Requires at least MIN_OBSERVATIONS to trust.

Priority 2 — MA-SAM+ result (fallback):
    Used when Priority 1 has too few observations or produced empty preconditions.

Priority 3 — merge:
    Per action schema, pick the better of Priority 1 or Priority 2.
"""

from collections import Counter, defaultdict

_NOP_NAMES = {"nop", "dummy-add-predicate-action", "dummy-del-predicate-action"}
_IGNORE_PREDS = {"dummy-additional-predicate"}


# ── State lifting ─────────────────────────────────────────────────────────────

def _lift_state(state, obj_to_var: dict, action_objs: set) -> set:
    """
    Return a set of lifted predicate strings for predicates whose every argument
    belongs to the action's objects.

    e.g. (holding a1 f) with obj_to_var={'a1':'?a','f':'?x'} -> '(holding ?a ?x)'
    """
    lifted = set()
    for pred_set in state.state_predicates.values():
        for pred in pred_set:
            rep = pred.untyped_representation          # '(holding a1 f)'
            inner = rep[1:-1].strip()                  # 'holding a1 f'
            parts = inner.split()
            if not parts:
                continue
            name, objs = parts[0], parts[1:]
            if name in _IGNORE_PREDS:
                continue
            if not all(o in action_objs for o in objs):
                continue
            lifted.add('(' + ' '.join([name] + [obj_to_var[o] for o in objs]) + ')')
    return lifted


# ── MA-SAM+ action data extraction ───────────────────────────────────────────

def _ma_sam_action_data(learner_action) -> tuple:
    """
    Extract (pre, add, del) string sets from a LearnerAction.

    pre: PDDL precondition strings (may include '(not (...))' for negative ones)
    add: positive effect strings  '(pred ?x ?y)'
    del: positive base strings    '(pred ?x ?y)'  — (not ...) is added when writing
    """
    pre = set(learner_action.preconditions_str_set)
    add, dlt = set(), set()
    for eff in learner_action.discrete_effects:
        rep = eff.untyped_representation
        if rep.startswith('(not '):
            dlt.add(rep[5:-1])          # strip '(not ' prefix and trailing ')'
        else:
            add.add(rep)
    return pre, add, dlt


# ── Priority-1 action PDDL writer ─────────────────────────────────────────────

def _p1_action_pddl(aname: str, signature, pre: set, add: set, dlt: set) -> str:
    """
    Write a PDDL action block from raw Priority-1 frequency sets.

    signature: LearnerAction.signature  (OrderedDict param_name -> type_obj)
    pre: positive precondition strings
    add: positive add-effect strings
    dlt: positive delete-effect strings (will be wrapped in (not ...))
    """
    params_str = ' '.join(f'{var} - {typ}' for var, typ in signature.items())
    pre_str = ' '.join(sorted(pre))
    eff_parts = sorted(add) + [f'(not {d})' for d in sorted(dlt)]
    eff_str = ' '.join(eff_parts)
    return (
        f"(:action {aname}\n"
        f"\t:parameters ({params_str})\n"
        f"\t:precondition (and {pre_str})\n"
        f"\t:effect (and {eff_str}))\n"
    )


# ── Domain PDDL writer ────────────────────────────────────────────────────────

def write_freq_domain_pddl(learned_domain, freq_model: dict) -> str:
    """
    Produce the final domain PDDL string.

    For actions where freq_model assigns priority=1: use the frequency-learned
    pre/add/del sets.  For all others: use the LearnerDomain's own action PDDL.
    The domain header (name, requirements, types, predicates) is taken from
    learned_domain.to_pddl() unchanged.
    """
    full_pddl = learned_domain.to_pddl()
    header_end = full_pddl.find('(:action')
    header = full_pddl[:header_end]

    action_blocks = []
    for aname, action in learned_domain.actions.items():
        entry = freq_model.get(aname.lower())
        if entry and entry['priority'] == 1:
            action_blocks.append(
                _p1_action_pddl(aname, action.signature,
                                entry['pre'], entry['add'], entry['del'])
            )
        else:
            action_blocks.append(action.to_pddl())

    return header + '\n'.join(action_blocks) + '\n)'


# ── Main learning function ────────────────────────────────────────────────────

def learn_with_frequency(
    observations,
    ma_sam_result,
    freq_threshold: float,
    effect_threshold: float,
    min_observations: int,
) -> dict:
    """
    Two-layer action model learning.

    observations   : list[MultiAgentObservation]  (output of adapter.py)
    ma_sam_result  : LearnerDomain or dict  action_name -> LearnerAction
    freq_threshold : precondition frequency cutoff  (e.g. 0.9)
    effect_threshold: effect frequency cutoff       (e.g. 0.7)
    min_observations: minimum single-agent steps to trust Priority 1

    Returns dict  action_name -> {pre, add, del, n, priority}
      pre : set[str]  PDDL precondition strings
      add : set[str]  positive add-effect strings
      del : set[str]  positive delete-effect base strings
      n   : int       number of single-agent observations used
      priority : 1 or 2
    """
    pre_counts: dict = defaultdict(Counter)
    add_counts: dict = defaultdict(Counter)
    del_counts: dict = defaultdict(Counter)
    obs_count:  dict = defaultdict(int)

    # Get domain action signatures for lifting  (param_name -> type, ordered)
    ma_actions = (
        ma_sam_result.actions
        if hasattr(ma_sam_result, 'actions')
        else ma_sam_result
    )

    # ── Priority 1: frequency counting from single-agent steps ───────────────
    for ma_obs in observations:
        for component in ma_obs.components:
            ops  = component.grounded_joint_action.operational_actions
            real = [a for a in ops if a.name.lower() not in _NOP_NAMES]
            if len(real) != 1:
                continue

            action = real[0]
            aname  = action.name.lower()

            schema = ma_actions.get(action.name) or ma_actions.get(aname)
            if schema is None:
                continue

            # action.parameters is List[str] of grounded objects in parameter order
            param_names  = list(schema.signature.keys())   # ['?a', '?x', '?y']
            concrete_args = action.parameters               # ['a1', 'f', 'g']
            obj_to_var   = {obj: var for var, obj in zip(param_names, concrete_args)}
            action_objs  = set(concrete_args)

            pre = _lift_state(component.previous_state, obj_to_var, action_objs)
            nxt = _lift_state(component.next_state,      obj_to_var, action_objs)
            add = nxt - pre
            dlt = pre - nxt

            obs_count[aname] += 1
            for lit in pre: pre_counts[aname][lit] += 1
            for lit in add: add_counts[aname][lit] += 1
            for lit in dlt: del_counts[aname][lit] += 1

    p1: dict = {}
    for aname, n in obs_count.items():
        if n < min_observations:
            continue
        pre = {lit for lit, c in pre_counts[aname].items() if c / n >= freq_threshold}
        add = {lit for lit, c in add_counts[aname].items() if c / n >= effect_threshold}
        dlt = {lit for lit, c in del_counts[aname].items() if c / n >= effect_threshold}
        p1[aname] = {'pre': pre, 'add': add, 'del': dlt, 'n': n}

    # ── Priority 2: MA-SAM+ ───────────────────────────────────────────────────
    p2: dict = {}
    for aname, la in ma_actions.items():
        aname = aname.lower()
        pre, add, dlt = _ma_sam_action_data(la)
        p2[aname] = {'pre': pre, 'add': add, 'del': dlt, 'n': obs_count.get(aname, 0)}

    # ── Priority 3: merge ─────────────────────────────────────────────────────
    final: dict = {}
    for aname in set(p1) | set(p2):
        use_p1 = (
            aname in p1
            and obs_count.get(aname, 0) >= min_observations
            and bool(p1[aname]['pre'])
        )
        if use_p1:
            final[aname] = {**p1[aname], 'priority': 1}
        elif aname in p2:
            final[aname] = {**p2[aname], 'priority': 2}

    return final
