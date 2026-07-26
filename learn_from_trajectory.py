#!/usr/bin/env python3
"""
Standalone action model learner from PDDL trajectory files.

Usage:
    python learn_from_trajectory.py domain.pddl problem.pddl trajectory.trajectory

Implements:
  1. Parse trajectory -> (s_t, operators, s_{t+1}) triplets
  2. Filter to single-agent steps (exactly one non-NOP operator)
  3. Extract pre/add/del per step, restricted to predicates relevant to the action
  4. Lift grounded observations to schema variables
  5. Preconditions = intersection across observations; effects = union
  6. Print learned model in PDDL format
"""

import sys
import re
from collections import defaultdict
from pathlib import Path

NOP_NAMES = {"nop", "dummy-add-predicate-action", "dummy-del-predicate-action"}
IGNORE_PREDICATES = {"dummy-additional-predicate"}
_PAREN_RE = re.compile(r'\(([^()]+)\)')
_COMMENT_RE = re.compile(r';[^\n]*')


# ── S-expression parser (for domain / problem) ────────────────────────────────

def _tokenize(text):
    text = _COMMENT_RE.sub('', text)
    return re.findall(r'\(|\)|[^\s()]+', text)


def _parse_sexp(tokens, pos):
    if tokens[pos] == '(':
        pos += 1
        items = []
        while tokens[pos] != ')':
            item, pos = _parse_sexp(tokens, pos)
            items.append(item)
        return items, pos + 1
    return tokens[pos], pos + 1


def _parse_typed_list(items):
    """('?x', '-', 'block', '?y', '-', 'block') -> [('?x','block'), ('?y','block')]"""
    result, pending = [], []
    i = 0
    while i < len(items):
        tok = items[i] if isinstance(items[i], str) else str(items[i])
        if tok == '-':
            i += 1
            typ = (items[i] if isinstance(items[i], str) else str(items[i])).lower()
            for p in pending:
                result.append((p, typ))
            pending = []
        else:
            pending.append(tok)
        i += 1
    for p in pending:
        result.append((p, 'object'))
    return result


# ── Domain parser ─────────────────────────────────────────────────────────────

def parse_domain(text):
    """
    Returns:
      actions: dict  action_name -> {'params': [('?var', 'type'), ...]}
    """
    tokens = _tokenize(text)
    top, _ = _parse_sexp(tokens, 0)   # ['define', ['domain', ...], [...], ...]
    actions = {}

    for item in top:
        if not isinstance(item, list) or not item:
            continue
        kw = item[0].lower() if isinstance(item[0], str) else ''
        if kw == ':action':
            aname = item[1].lower()
            params = []
            i = 2
            while i < len(item):
                if isinstance(item[i], str) and item[i] == ':parameters':
                    params = _parse_typed_list(item[i + 1])
                    i += 2
                else:
                    i += 1
            actions[aname] = {'params': params}

    return actions


# ── Problem parser (object types, for richer output) ─────────────────────────

def parse_problem(text):
    """Returns dict  object_name -> type_name"""
    tokens = _tokenize(text)
    top, _ = _parse_sexp(tokens, 0)
    obj_types = {}
    for item in top:
        if not isinstance(item, list) or not item:
            continue
        if isinstance(item[0], str) and item[0].lower() == ':objects':
            for name, typ in _parse_typed_list(item[1:]):
                obj_types[name.lower()] = typ
    return obj_types


# ── Trajectory parser ─────────────────────────────────────────────────────────

def _extract_preds(line):
    """All (name arg...) tokens on a line; skip keywords and ignored predicates."""
    preds = set()
    for m in _PAREN_RE.finditer(line):
        parts = m.group(1).strip().split()
        if not parts:
            continue
        name = parts[0].lower()
        if ':' in name or name in IGNORE_PREDICATES:
            continue
        preds.add((name,) + tuple(p.lower() for p in parts[1:]))
    return preds


def _extract_ops(line):
    """All (name arg...) tokens from an operators line; skip the 'operators:' token."""
    ops = []
    for m in _PAREN_RE.finditer(line):
        parts = m.group(1).strip().split()
        if not parts:
            continue
        name = parts[0].lower()
        if ':' in name:          # skip 'operators:' token
            continue
        ops.append((name,) + tuple(p.lower() for p in parts[1:]))
    return ops


def parse_trajectory(text):
    """
    Returns list of (s_t, operators, s_{t+1}) where
      s_t, s_{t+1} are sets of predicate tuples: {('on','f','g'), ...}
      operators    is a list of action tuples:    [('unstack','a1','f','g'), ...]
    """
    init_state = None
    states = []
    operator_blocks = []

    for raw_line in text.splitlines():
        line = raw_line.strip().lower()
        if '(:init' in line or line.startswith('((:init'):
            init_state = _extract_preds(raw_line)
        elif line.startswith('(:state'):
            states.append(_extract_preds(raw_line))
        elif 'operators:' in line:
            operator_blocks.append(_extract_ops(raw_line))

    if init_state is None:
        raise ValueError("No :init section found in trajectory file.")

    all_states = [init_state] + states
    triplets = []
    for i, ops in enumerate(operator_blocks):
        if i + 1 < len(all_states):
            triplets.append((all_states[i], ops, all_states[i + 1]))
    return triplets


# ── Single-agent filtering ────────────────────────────────────────────────────

def filter_single_agent(triplets):
    """Keep steps where exactly one operator is not a NOP."""
    result = []
    for s_t, ops, s_next in triplets:
        real = [op for op in ops if op[0] not in NOP_NAMES]
        if len(real) == 1:
            result.append((s_t, real[0], s_next))
    return result


# ── Lifting ───────────────────────────────────────────────────────────────────

def _lift_step(s_t, operator, s_next, schema):
    """
    Lift one grounded step to schema variables.

    Returns (preconditions, add_effects, delete_effects) as frozensets of
    lifted predicate tuples, e.g. frozenset({('on','?x','?y'), ...}).
    Only predicates whose every argument belongs to the action's objects
    are considered (i.e. "relevant" predicates).
    """
    params = schema['params']                  # [('?a','agent'), ...]
    concrete_args = list(operator[1:])
    action_objs = set(concrete_args)

    # object -> variable mapping  (position in operator matches position in params)
    obj_to_var = {obj: params[i][0] for i, obj in enumerate(concrete_args)
                  if i < len(params)}

    def lift(pred):
        name, *args = pred
        if not all(a in action_objs for a in args):
            return None
        return (name,) + tuple(obj_to_var[a] for a in args)

    pre  = frozenset(filter(None, (lift(p) for p in s_t)))
    nxt  = frozenset(filter(None, (lift(p) for p in s_next)))
    return pre, nxt - pre, pre - nxt   # preconditions, add_effects, delete_effects


# ── Learning ──────────────────────────────────────────────────────────────────

def learn(sa_triplets, actions):
    """
    Returns dict  action_name -> {'pre': set, 'add': set, 'del': set, 'n': int}

    Preconditions: intersection across all observations (conservative).
    Effects:       union across all observations.
    """
    buckets = defaultdict(list)   # aname -> [(pre, add, del)]

    for s_t, operator, s_next in sa_triplets:
        aname = operator[0]
        if aname not in actions:
            continue
        pre, add, dlt = _lift_step(s_t, operator, s_next, actions[aname])
        buckets[aname].append((pre, add, dlt))

    result = {}
    for aname, obs in buckets.items():
        pre_inter = set(obs[0][0])
        for p, _, _ in obs[1:]:
            pre_inter &= p
        add_union = set().union(*(a for _, a, _ in obs))
        del_union = set().union(*(d for _, _, d in obs))
        result[aname] = {
            'pre': pre_inter,
            'add': add_union,
            'del': del_union,
            'n':   len(obs),
        }

    return result


# ── PDDL output ───────────────────────────────────────────────────────────────

def _pred_str(tup):
    return '(' + ' '.join(tup) + ')'


def _action_pddl(aname, schema, learned):
    data = learned[aname]
    params_str = ' '.join(f"{v} - {t}" for v, t in schema['params'])
    pre  = sorted(data['pre'])
    adds = sorted(data['add'])
    dels = sorted(data['del'])

    conds  = '  '.join(_pred_str(p) for p in pre)
    effects = ' '.join(_pred_str(p) for p in adds)
    if dels:
        effects += (' ' if effects else '') + ' '.join(f'(not {_pred_str(p)})' for p in dels)

    return (
        f"  (:action {aname}\n"
        f"      :parameters ({params_str})\n"
        f"      :precondition (and {conds})\n"
        f"      :effect (and {effects}))"
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) not in (4, 5):
        print("Usage: python learn_from_trajectory.py domain.pddl problem.pddl trajectory.trajectory [output.pddl]")
        sys.exit(1)

    domain_path, problem_path, traj_path = sys.argv[1:4]
    out_path = Path(sys.argv[4]) if len(sys.argv) == 5 else (
        Path("output") / ("learned_" + Path(traj_path).stem + ".pddl")
    )

    actions     = parse_domain(Path(domain_path).read_text())
    _obj_types  = parse_problem(Path(problem_path).read_text())
    triplets    = parse_trajectory(Path(traj_path).read_text())

    print(f"; Trajectory steps:    {len(triplets)}", file=sys.stderr)

    sa = filter_single_agent(triplets)
    print(f"; Single-agent steps:  {len(sa)} / {len(triplets)}", file=sys.stderr)

    learned = learn(sa, actions)

    print(f"; Actions with data:   {sorted(learned.keys())}", file=sys.stderr)
    for aname, d in sorted(learned.items()):
        print(f";   {aname}: {d['n']} observations", file=sys.stderr)

    lines = ["(define (domain learned)",
             "  (:requirements :typing :negative-preconditions :equality)", ""]
    for aname in sorted(actions):
        if aname in NOP_NAMES:
            continue
        if aname not in learned:
            print(f"  ; {aname}: no observations", file=sys.stderr)
            continue
        lines.append(_action_pddl(aname, actions[aname], learned))
        lines.append("")
    lines.append(")")

    pddl = "\n".join(lines)
    print(pddl)

    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(pddl)
    print(f"\nWrote {out_path}", file=sys.stderr)


if __name__ == '__main__':
    main()
