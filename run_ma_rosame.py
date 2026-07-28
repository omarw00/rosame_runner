import re
import random
import tempfile
from pathlib import Path

from ma_rosame import MARosame

DATA = Path(__file__).parent / "libs/ma-sam/experiments_dataset/blocks"
AGENTS = ["a1", "a2", "a3", "a4"]
NOISE_THRESHOLD   = 0.1
NOISE_RATE        = 0.0    # set to 0.10 to simulate noise
EPOCHS            = 300
MAX_TRAJECTORIES  = 8     # limit trajectory count; None = use all (blocks is large/slow at 20)
RANDOM_SEED       = 42


def _extract_predicates(line: str, keywords: set) -> set:
    """Extract predicate tokens from a single trajectory line, excluding keyword tags."""
    tokens = set(re.findall(r'\([a-z][a-z0-9\-]*(?:\s+[a-z][a-z0-9]*)*\)', line))
    return {p for p in tokens if not any(k in p for k in keywords)}


def inject_noise(trajectory_path: Path, noise_rate: float, rng: random.Random) -> str:
    """Return trajectory text with noise injected only into dynamic predicates.

    A predicate is "dynamic" if it is observed to change (be added or removed)
    at least once between any two consecutive states in this trajectory. Static
    predicates (path, link, etc.) never change across a transition, so flipping
    them is invisible to a transition-based detector like ROSAME — only dynamic
    predicates are eligible for corruption.

    :init (s_0) is never touched; only :state lines are corrupted.
    Each eligible predicate has a `noise_rate` chance of being flipped:
    - present predicates may be removed
    - absent predicates (drawn from the dynamic set) may be added
    """
    text = trajectory_path.read_text()
    lines = text.splitlines()
    keywords = {":init", ":state"}

    # Collect predicate vocabulary and per-state snapshots from :init/:state lines
    state_lines = [l for l in lines if l.strip().startswith("(:init") or l.strip().startswith("(:state")]
    state_sets = [_extract_predicates(l, keywords) for l in state_lines]
    all_predicates = set().union(*state_sets) if state_sets else set()

    # Dynamic predicates: those that change between any two consecutive states
    dynamic_predicates = set()
    for a, b in zip(state_sets, state_sets[1:]):
        dynamic_predicates |= (a ^ b)

    noisy_lines = []
    for line in lines:
        if not line.strip().startswith("(:state"):
            noisy_lines.append(line)
            continue

        present = _extract_predicates(line, keywords)
        absent  = all_predicates - present
        to_remove = {p for p in (present & dynamic_predicates) if rng.random() < noise_rate}
        to_add    = {p for p in (absent  & dynamic_predicates) if rng.random() < noise_rate}
        new_state = (present - to_remove) | to_add
        noisy_lines.append(f"(:state  {'  '.join(sorted(new_state))} )")

    return "\n".join(noisy_lines)


def write_noisy_trajectories(trajectory_paths, noise_rate, seed, tmp_dir):
    """Write noise-injected copies to tmp_dir, return new path list."""
    rng = random.Random(seed)
    noisy_paths = []
    for tp in trajectory_paths:
        noisy_text = inject_noise(tp, noise_rate, rng)
        out = Path(tmp_dir) / tp.name
        out.write_text(noisy_text)
        noisy_paths.append(out)
    return noisy_paths


# ── main ──────────────────────────────────────────────────────────────────────

pairs = sorted(
    (t, t.with_suffix(".pddl"))
    for t in DATA.glob("*.trajectory")
    if t.with_suffix(".pddl").exists()
)
if MAX_TRAJECTORIES is not None:
    pairs = pairs[:MAX_TRAJECTORIES]
trajectory_paths = [t for t, _ in pairs]
problem_paths = [p for _, p in pairs]

print(f"Trajectories:    {len(trajectory_paths)}")
print(f"Noise rate:      {NOISE_RATE * 100:.0f}%")
print(f"Noise threshold: {NOISE_THRESHOLD}")
print(f"Epochs:          {EPOCHS}")

domain_candidates = list(DATA.glob("*_combined_domain.pddl"))
if not domain_candidates:
    # fall back: any .pddl file that isn't a per-trajectory problem file
    problem_stems = {t.stem for t in DATA.glob("*.trajectory")}
    domain_candidates = [p for p in DATA.glob("*.pddl") if p.stem not in problem_stems]
domain_file = domain_candidates[0]

ma_rosame = MARosame(
    domain_path=domain_file,
    agents=AGENTS,
    noise_threshold=NOISE_THRESHOLD,
    epochs=EPOCHS,
)

if NOISE_RATE > 0:
    with tempfile.TemporaryDirectory() as tmp_dir:
        noisy_paths = write_noisy_trajectories(trajectory_paths, NOISE_RATE, RANDOM_SEED, tmp_dir)
        learned_domain, report, macro_mapping = ma_rosame.fit(
            trajectory_paths=noisy_paths,
            problem_paths=problem_paths,
        )
else:
    learned_domain, report, macro_mapping = ma_rosame.fit(
        trajectory_paths=trajectory_paths,
        problem_paths=problem_paths,
    )

domain_name = DATA.name
output_path = Path(f"output/learned_{domain_name}_domain_ma_rosame.pddl")
output_path.parent.mkdir(exist_ok=True)
ma_rosame.export(learned_domain, output_path)

print(f"\nSafe actions:    {sorted(report.keys())}")
print(f"Macro mapping:   {list(macro_mapping.keys())}")
print(f"Wrote {output_path}")
