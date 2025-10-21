import json
import os
from pathlib import Path

from pydantic import TypeAdapter

from src.models import Challenge

arc_prize_data_path = Path(__file__).parent.parent / "arc-prize-2024"
arc_prize_v2_data_path = Path(__file__).parent.parent / "arc-agi-2"
SolutionAdapter = TypeAdapter(dict[str, list[list[list[int]]]])
ChallengeAdapter = TypeAdapter(dict[str, Challenge])


def build_dummy_solutions(challenges_j: dict) -> dict[str, list[list[list[int]]]]:
    solutions_j: dict[str, list[list[list[int]]]] = {}
    for challenge_id in challenges_j.keys():
        solutions_list: list[list[list[int]]] = []
        for _ in challenges_j[challenge_id]["test"]:
            solutions_list.append([[0], [0]])
        solutions_j[challenge_id] = solutions_list
    return solutions_j


def build_challenges(
    challenges_path: Path, solutions_path: Path | None
) -> dict[str, Challenge]:
    challenges_j = json.loads(open(challenges_path).read())
    if solutions_path:
        solutions_d: dict[str, list[list[list[int]]]] = SolutionAdapter.validate_json(
            open(solutions_path).read()
        )
    else:
        solutions_d = build_dummy_solutions(challenges_j)
    for k, v in challenges_j.items():
        for i, val in enumerate(v["test"]):
            val["output"] = solutions_d[k][i]
        v["id"] = k
        if int(os.environ.get("TOTAL_TASKS", "500")) == k:
            break
    return ChallengeAdapter.validate_python(challenges_j)

def build_challenges_v2(
    challenges_path: Path
) -> dict[str, Challenge]:
    # Read every file in the directory and use the file_path prefix as the key in the challenges_j dictionary
    challenges_j = {}
    for file_path in sorted(challenges_path.iterdir()):
        if file_path.is_file() and file_path.suffix == ".json":
            with open(file_path) as f:
                file_challenge = json.load(f)
                # Use the file name without suffix as the key
                key = file_path.stem
                challenges_j[key] = file_challenge
                challenges_j[key]["id"] = key
    return ChallengeAdapter.validate_python(challenges_j)

training_challenges = build_challenges(
    challenges_path=arc_prize_data_path / "arc-agi_training_challenges.json",
    solutions_path=arc_prize_data_path / "arc-agi_training_solutions.json",
)
eval_challenges = build_challenges(
    challenges_path=arc_prize_data_path / "arc-agi_evaluation_challenges.json",
    solutions_path=arc_prize_data_path / "arc-agi_evaluation_solutions.json",
)

v2_training_challenges = build_challenges_v2(
    challenges_path=arc_prize_v2_data_path / "training",
)
v2_eval_challenges = build_challenges_v2(
    challenges_path=arc_prize_v2_data_path / "evaluation",
)

"""
for example_id, val in training_challenges.items():
    # print(f"{example_id=}")
    # debug(len(val["train"]))
    # debug(len(val["test"]))
    # if len(val.test) > 1:
    #     print(example_id, len(val.test))
    ...
"""

__all__ = [
    "training_challenges",
    "eval_challenges",
]
