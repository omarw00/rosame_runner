# Mid-Progress Report: MA-ROSAME
### Multi-Agent Robust Safe Action Model Estimation

**Student:** Royi Aizen
**Date:** June 2026

---

## 1. Overview

This project develops **MA-ROSAME** (Multi-Agent Robust Safe Action Model Estimation), a hybrid learning framework that induces safe, multi-agent PDDL action models from noisy execution traces. The system combines two existing algorithms — ROSAME and MA-SAM+ — through a novel bridge that uses neural predictions to filter noisy observations before symbolic learning.

---

## 2. Background

### 2.1 ROSAME — Neural Action Model Learner
ROSAME (neuRO-Symbolic Action Model lEarner) is a differentiable framework for learning lifted action models from execution traces. It trains a small MLP per action schema to predict soft precondition, add-effect, and delete-effect tensors over a grounded proposition space.

The training loss for a single-agent step `(s1, action, s2)` is:

```
predicted_s2 = s1 ⊙ (1 − deleff) + (1 − s1) ⊙ addeff

loss = MSE(predicted_s2, s2)
     + MSE((1 − s1) ⊙ precon, 0)      ← false props cannot be preconditions
     + 0.2 · MSE(precon, 1)           ← encourage confident preconditions
```

ROSAME operates on **single-agent** observations only.

### 2.2 MA-SAM+ — Multi-Agent Symbolic Learner
MA-SAM+ is a SAT-based algorithm that learns action models from multi-agent concurrent observations. It builds a CNF (Conjunctive Normal Form) matrix from observed state transitions and resolves agent-effect ambiguities using macro-actions. MA-SAM+ produces exact symbolic PDDL output but is brittle to noisy input — a single flipped predicate can corrupt the learned model.

---

## 3. Proposed Solution: MA-ROSAME

The key insight is to use ROSAME's learned transition model as a **noise oracle**: steps that ROSAME cannot explain (high MSE) are likely noisy and should be discarded before MA-SAM+ sees them.

### 3.1 Algorithm

The MA-ROSAME pipeline proceeds in six phases:

**Phase 1 — Parse**
Multi-agent PDDL trajectory files are parsed into `MultiAgentObservation` objects using `pddl_plus_parser`.

**Phase 2 — Train ROSAME (surrogate)**
Since ROSAME is a single-agent model, pseudo-single-agent steps are extracted from each joint step by selecting the first non-NOP agent action. All trajectories are merged into one dataset and trained jointly over a union proposition space (objects from all problem instances combined).

**Phase 3 — Score Steps (Novel Contribution)**
Each multi-agent step is scored using a **joint multi-agent loss function**:

For a step with concurrent agent actions `{a₁, a₂, ..., aₙ}`:
```
complement_add = ⊙ᵢ (1 − addeff_i)
complement_del = ⊙ᵢ (1 − deleff_i)

joint_addeff = 1 − complement_add          ← product-of-complements
joint_deleff = 1 − complement_del

predicted_s2 = s1 ⊙ (1 − joint_deleff) + (1 − s1) ⊙ joint_addeff

score = MSE(predicted_s2, actual_s2)
```

The product-of-complements formula computes the probability that *at least one* agent causes each effect, under an independence assumption. Unlike a clamped sum, this stays in `[0, 1]` for any number of agents and reduces to the original ROSAME formula when `n = 1`.

**Phase 4 — Filter Noisy Steps**
Steps with `score > threshold` are dropped. The cleaned `MultiAgentObservation` retains only steps the model can explain.

**Phase 5 — Symbolic Learning**
The filtered observations are passed to MA-SAM+, which learns a symbolic PDDL action model.

**Phase 6 — Export**
The learned domain is serialised to PDDL.

### 3.2 System Architecture

```
PDDL trajectories + domain
        │
        ▼
  Rosame_Runner.learn_rosame()            ← Phase 2: train ROSAME
        │
        ▼
  noise_filter.score_steps()              ← Phase 3: joint MSE per step
        │  drop steps above threshold
        ▼
  adapter.to_multi_agent_observation()    ← Phase 4: keep clean steps
        │
        ▼
  MASAMPlus.learn_combined_action_model_with_macro_actions()
        │
        ▼
  LearnerDomain → PDDL file
```

### 3.3 Implementation

The bridge is implemented as a self-contained Python module `ma_rosame_module/` with three components:

| File | Responsibility |
|---|---|
| `noise_filter.py` | `score_steps()` and `filter_observation()` |
| `adapter.py` | Converts between single-agent and multi-agent observation formats |
| `learner.py` | `MARosame` class orchestrating the full pipeline |

The existing `rosame.py`, `rosame_runner.py`, and `libs/ma-sam/` are not modified.

---

## 4. Experiments

### 4.1 Domain
Experiments use the **Blocksworld** domain with 4 concurrent agents (`a1`–`a4`), 20 problem instances, and 1643 total trajectory steps. Noise is injected via random predicate flipping at a configurable rate (default 10%), writing noisy copies to a temp directory without touching originals.

### 4.2 Baseline
`run_ma_sam_plus_noisy.py` runs MA-SAM+ directly on the noisy trajectories (no ROSAME filter), using the same random seed, for direct comparison.

### 4.3 Results

**Clean data (`NOISE_RATE = 0.0`, `threshold = 0.1`):**
- 100% of steps pass the filter — the model correctly identifies all clean steps as clean
- Learned domain is identical to the MA-SAM+ baseline on clean data ✓

**Noisy data (`NOISE_RATE = 0.10`, `threshold = 0.1`):**
- Filter drops 5–17% of steps across trajectories
- MA-ROSAME recovers at least one effect per action (e.g. `unstack` recovers `(not (on ?y ?x))`)
- MA-SAM+ baseline on the same noisy data recovers **zero** effects for most actions

---

## 5. Known Limitations and Open Problems

### 5.1 ROSAME Convergence
The union proposition space (all objects from all 20 problem instances) is large. After 100 training epochs, the model loss barely moves (149 → 144). As a result, clean-step MSE scores sit in the `0.05–0.15` range, overlapping with noisy-step scores — making threshold selection unreliable.

### 5.2 Threshold Sensitivity
- `threshold = 0.05`: drops ~85–95% of steps including clean ones (too aggressive)
- `threshold = 0.1`: lets some noisy steps through

There is no safe threshold until the model converges.

### 5.3 Incomplete Effect Recovery
At 10% noise, MA-ROSAME recovers at most 1–2 effects per action where the clean domain has up to 5. This is a downstream consequence of poor convergence rather than a fundamental algorithm problem.

---

## 6. Next Steps

1. **Increase training epochs (500+)** — the union proposition space requires significantly more training. Currently blocked by CPU runtime (~3 min per 100 epochs).

2. **Per-problem grounding during training** — instead of one large union space, train each trajectory separately while sharing optimizer state across calls. This reduces the proposition space per training call and should accelerate convergence.

3. **Learning rate tuning** — the Adam optimizer uses `lr = 1e-3` hardcoded in `rosame_runner.py`. A higher rate (e.g. `5e-3`) may accelerate convergence on the larger space.

4. **GPU acceleration** — moving the PyTorch training to GPU would allow significantly more epochs in the same wall-clock time.

---

## 7. Summary

| Component | Status |
|---|---|
| ROSAME integration (single-agent surrogate training) | ✅ Working |
| Joint multi-agent loss function (product-of-complements) | ✅ Implemented |
| Step filtering and noise detection | ✅ Working |
| Full pipeline on clean data | ✅ Correct output |
| Noise injection + baseline comparison | ✅ Working |
| Convergence on noisy data | ❌ Insufficient epochs |
| Reliable threshold selection under noise | ❌ Blocked by convergence |
