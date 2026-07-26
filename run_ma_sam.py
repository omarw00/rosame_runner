import ma_sam_path  # noqa: F401 – adds libs/ma-sam to sys.path
from pathlib import Path
from pddl_plus_parser.lisp_parsers import DomainParser, ProblemParser, TrajectoryParser
from sam_learning.learners import MultiAgentSAM

DATA = Path(__file__).parent / "libs/ma-sam/experiments_dataset/blocks"
AGENTS = ["a1", "a2", "a3", "a4"]

domain = DomainParser(DATA / "blocks_combined_domain.pddl", partial_parsing=True).parse_domain()

observations = []
for problem_path in sorted(DATA.glob("probBLOCKS-*.pddl")):
    traj = problem_path.with_suffix(".trajectory")
    if not traj.exists():
        continue
    problem = ProblemParser(problem_path=problem_path, domain=domain).parse_problem()
    observations.append(
        TrajectoryParser(domain, problem).parse_trajectory(traj, executing_agents=AGENTS)
    )

print(f"Loaded {len(observations)} trajectories")

learner = MultiAgentSAM(domain)
learned_domain, report = learner.learn_combined_action_model(observations)

out = Path("output/learned_blocks_domain.pddl")
out.parent.mkdir(exist_ok=True)
out.write_text(learned_domain.to_pddl())
print(f"Safe actions: {learner.safe_actions}")
print(f"Unsafe actions: {sorted(learner._unsafe_actions)}")
print(f"Wrote {out}")
