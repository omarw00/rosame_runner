"""
Baseline: run MA-SAM+ on the same noisy trajectories that run_ma_rosame.py uses,
but WITHOUT the ROSAME denoising filter. Lets us compare learned domains directly.
"""
import re
import random
import tempfile
from pathlib import Path

import ma_sam_path  # noqa: F401
from pddl_plus_parser.lisp_parsers import DomainParser, ProblemParser, TrajectoryParser
from sam_learning.learners import MASAMPlus

DATA = Path(__file__).parent / "libs/ma-sam/experiments_dataset/blocks"
AGENTS = ["a1", "a2", "a3", "a4"]
NOISE_RATE = 0.10
RANDOM_SEED = 42

# ── same noise injector as run_ma_rosame.py ───────────────────────────────────

def inject_noise(trajectory_path: Path, noise_rate: float, rng: random.Random) -> str:
    text = trajectory_path.read_text()
    lines = text.splitlines()
    state_lines = [l for l in lines if l.strip().startswith("(:init") or l.strip().startswith("(:state")]
    state_text = "\n".join(state_lines)
    all_predicates = set(re.findall(r'\([a-z][a-z0-9\-]*(?:\s+[a-z][a-z0-9]*)*\)', state_text))
    keywords = {":init", ":state"}
    all_predicates = {p for p in all_predicates if not any(k in p for k in keywords)}

    noisy_lines = []
    for line in lines:
        if not line.strip().startswith("(:state"):
            noisy_lines.append(line)
            continue
        present = set(re.findall(r'\([a-z][a-z0-9\-]*(?:\s+[a-z][a-z0-9]*)*\)', line))
        present = {p for p in present if not any(k in p for k in keywords)}
        absent = all_predicates - present
        to_remove = {p for p in present if rng.random() < noise_rate}
        to_add = {p for p in absent if rng.random() < noise_rate}
        new_state = (present - to_remove) | to_add
        noisy_lines.append(f"(:state  {'  '.join(sorted(new_state))} )")
    return "\n".join(noisy_lines)


# ── main ──────────────────────────────────────────────────────────────────────

pairs = sorted(
    (t, t.with_suffix(".pddl"))
    for t in DATA.glob("probBLOCKS-*.trajectory")
    if t.with_suffix(".pddl").exists()
)
trajectory_paths = [t for t, _ in pairs]
problem_paths    = [p for _, p in pairs]

print(f"Trajectories: {len(trajectory_paths)}")
print(f"Noise rate:   {NOISE_RATE * 100:.0f}%  (seed={RANDOM_SEED})")
print("No denoising filter — raw noisy input to MA-SAM+")

domain = DomainParser(DATA / "blocks_combined_domain.pddl", partial_parsing=True).parse_domain()

with tempfile.TemporaryDirectory() as tmp_dir:
    rng = random.Random(RANDOM_SEED)
    observations = []
    for traj_path, prob_path in zip(trajectory_paths, problem_paths):
        noisy_text = inject_noise(traj_path, NOISE_RATE, rng)
        noisy_file = Path(tmp_dir) / traj_path.name
        noisy_file.write_text(noisy_text)

        problem = ProblemParser(problem_path=prob_path, domain=domain).parse_problem()
        obs = TrajectoryParser(domain, problem).parse_trajectory(
            noisy_file, executing_agents=AGENTS
        )
        observations.append(obs)

    learner = MASAMPlus(domain)
    learned_domain, report, macro_mapping = (
        learner.learn_combined_action_model_with_macro_actions(observations)
    )

output_path = Path("output/learned_blocks_domain_noisy_baseline.pddl")
output_path.parent.mkdir(exist_ok=True)
output_path.write_text(learned_domain.to_pddl())

print(f"Safe actions:  {sorted(report.keys())}")
print(f"Macro mapping: {list(macro_mapping.keys())}")
print(f"Wrote {output_path}")
