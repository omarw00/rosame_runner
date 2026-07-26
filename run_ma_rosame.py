import re
import random
import tempfile
from pathlib import Path

from ma_rosame_module.learner import MARosame

DATA = Path(__file__).parent / "libs/ma-sam/experiments_dataset/blocks"
AGENTS = ["a1", "a2", "a3", "a4"]
NOISE_THRESHOLD = 0.1
NOISE_RATE = 0.01
EPOCHS = 300
RANDOM_SEED = 42


def inject_noise(trajectory_path: Path, noise_rate: float, rng: random.Random) -> str:
    """Return trajectory text with random-predicate-flip noise applied to every :state line.

    Each predicate in a state has a `noise_rate` chance of being flipped:
    - present predicates may be removed
    - absent predicates (seen elsewhere in the file) may be added
    """
    text = trajectory_path.read_text()
    lines = text.splitlines()

    # Collect predicate tokens only from :init and :state lines (not operators lines)
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

        # Extract predicates currently in this state
        present = set(re.findall(r'\([a-z][a-z0-9\-]*(?:\s+[a-z][a-z0-9]*)*\)', line))
        present = {p for p in present if not any(k in p for k in keywords)}
        absent = all_predicates - present

        # Flip a random subset of present predicates (remove them)
        to_remove = {p for p in present if rng.random() < noise_rate}
        # Flip a random subset of absent predicates (add them)
        to_add = {p for p in absent if rng.random() < noise_rate}

        new_state = (present - to_remove) | to_add
        state_str = "  ".join(sorted(new_state))
        noisy_lines.append(f"(:state  {state_str} )")

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
    for t in DATA.glob("probBLOCKS-*.trajectory")
    if t.with_suffix(".pddl").exists()
)
trajectory_paths = [t for t, _ in pairs]
problem_paths = [p for _, p in pairs]

print(f"Trajectories:    {len(trajectory_paths)}")
print(f"Noise rate:      {NOISE_RATE * 100:.0f}%")
print(f"Noise threshold: {NOISE_THRESHOLD}")
print(f"Epochs:          {EPOCHS}")

ma_rosame = MARosame(
    domain_path=DATA / "blocks_combined_domain.pddl",
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

output_path = Path("output/learned_blocks_domain_ma_rosame.pddl")
output_path.parent.mkdir(exist_ok=True)
ma_rosame.export(learned_domain, output_path)

print(f"Safe actions:    {sorted(report.keys())}")
print(f"Macro mapping:   {list(macro_mapping.keys())}")
print(f"Wrote {output_path}")
