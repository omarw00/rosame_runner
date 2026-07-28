# MA-ROSAME Algorithm

## Input
- PDDL domain file
- N multi-agent trajectories (possibly noisy)
- `noise_threshold`, `freq_threshold`, `effect_threshold`, `epochs`, `min_obs`

## Output
- Learned action model **M**: Pre(α), Add(α), Del(α) per action schema α

---

## Phase 1 — Parse

```
for each trajectory t:
    parse problem  → grounded objects
    parse trajectory → (s_t, â_t, s_{t+1}) triplets
    encode states as binary vectors over grounded propositions
```

---

## Phase 2 — Train ROSAME (Two Passes)

### Pass 1 — Train on ALL steps

```
for each step (s_t, â_t, s_{t+1}):
    active = agents with non-NOP action in â_t

    for each agent i in active:
        add_i, del_i ← PAM_Network(action_i)

    joint_add[p] = 1 - Π_i (1 - add_i[p])
    joint_del[p] = 1 - Π_i (1 - del_i[p])

    predicted = s_t × (1 - joint_del)
              + (1 - s_t) × joint_add

    Loss = MSE(predicted, s_{t+1})
    backpropagate → update PAM Networks
```

### Intermediate Scoring

```
for each step:
    score(step) = MSE(predicted_s_{t+1}, actual s_{t+1})
    flag step if score > noise_threshold
```

### Pass 2 — Retrain on CLEAN steps only

```
same loop as Pass 1, skipping flagged steps
```

---

## Phase 3 — Filter

```
for each step:
    if score(step) > noise_threshold → drop
    else                             → keep

kept_steps ← clean observations
```

---

## Phase 4 — Frequency Counting (Single-Agent Steps)

```
for each kept step where |active| == 1:
    lift grounded predicates → schema variables

    pre = { p | p true in s_t,       all args ∈ action objects }
    add = { p | p true in s_{t+1},   p not true in s_t         }
    del = { p | p true in s_t,       p not true in s_{t+1}     }

    accumulate counters per action schema α

for each action schema α:
    n = number of single-agent observations of α
    if n >= min_obs:
        Pre(α) = { p | count_pre(p) / n >= freq_threshold  }
        Add(α) = { p | count_add(p) / n >= effect_threshold }
        Del(α) = { p | count_del(p) / n >= effect_threshold }
```

---

## Phase 5 — MA-SAM+ Fallback

```
run MA-SAM+ on kept_steps (multi-agent observations)
→ produces Pre_2(α), Add_2(α), Del_2(α) for all α
→ handles actions with too few single-agent observations
```

---

## Phase 6 — Merge

```
for each action schema α:
    if observations(α) >= min_obs AND Pre(α) ≠ ∅:
        M(α) ← frequency result      ← Priority 1
    else:
        M(α) ← MA-SAM+ result        ← Priority 2

output PDDL domain with M
```

---

## Complexity Notes

| Phase | Dominant cost |
|-------|--------------|
| 1 | O(N · T) — N trajectories, T steps each |
| 2 | O(epochs · N · T · P) — P propositions per problem |
| 3 | O(N · T) |
| 4 | O(N · T · |state|) |
| 5 | O(MA-SAM+ runtime) |
| 6 | O(|actions|) |

## Key Design Decisions

| Decision | Reason |
|----------|--------|
| Two-pass training | Pass 1 bootstraps a rough model; Pass 2 avoids fitting noisy dynamics |
| Per-trajectory grounding | Each problem has its own object set; union space is too large to converge |
| Product-of-complements for joint steps | Principled composition of independent agent effects |
| Single-agent steps only for frequency counting | Joint steps cannot unambiguously attribute pre/add/del to individual agents |
| Priority merge | Frequency counting is more reliable when observations are sufficient; MA-SAM+ handles the rest |
