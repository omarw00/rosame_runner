from pathlib import Path

from pddl_plus_parser.lisp_parsers import TrajectoryParser
from pddl_plus_parser.models import MultiAgentObservation


def to_multi_agent_observation(
    domain, problem, trajectory_path: Path, agents: list[str], kept_indices: list[int]
) -> MultiAgentObservation:
    """Re-parses trajectory_path with executing_agents, keeps only steps at kept_indices."""
    full_obs = TrajectoryParser(domain, problem).parse_trajectory(
        trajectory_file_path=trajectory_path,
        executing_agents=agents,
    )

    filtered = MultiAgentObservation(executing_agents=agents)
    filtered.add_problem_objects(problem.objects)

    kept_set = set(kept_indices)
    for i, component in enumerate(full_obs.components):
        if i in kept_set:
            filtered.components.append(component)

    return filtered
