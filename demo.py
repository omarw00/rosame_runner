import os
from pathlib import Path

from pddl_plus_parser.lisp_parsers import ProblemParser, DomainParser, TrajectoryParser

from rosame_runner import Rosame_Runner


train_set_dir_path = Path(os.environ.get("ROSAME_DATA_DIR", Path(__file__).parent / "problems"))
domain_file_name = os.environ.get("ROSAME_DOMAIN_FILE", "blocksworld.pddl")
problem_file_name = os.environ.get("ROSAME_PROBLEM_FILE", "0_blocksworld_prob.pddl")

sorted_trajectory_paths = sorted(train_set_dir_path.glob("*_traj.txt"))  # for consistency

rosame = Rosame_Runner(train_set_dir_path/domain_file_name)
problem_path = train_set_dir_path / problem_file_name

for index, trajectory_file_path in enumerate(sorted_trajectory_paths):
    problem = ProblemParser(problem_path, rosame.domain).parse_problem()
    complete_observation = TrajectoryParser(rosame.domain, problem).parse_trajectory(
        trajectory_file_path
    )
    rosame.add_problem(problem)
    rosame.ground_new_trajectory()
    rosame.learn_rosame(complete_observation)

print(rosame.rosame_to_pddl())