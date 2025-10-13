import asyncio
import random
import pickle
import os
import argparse
import wandb
import omegaconf
import hydra
from flax.serialization import from_bytes
import jax
import optax
from flax.training.train_state import TrainState
from collections import defaultdict

from devtools import debug

from pathlib import Path

from src.data import eval_challenges, training_challenges, v2_training_challenges, v2_eval_challenges, build_challenges_v2, build_challenges
from src.logic import solve_challenge_with_accuracy
from src.models import GRID
from src import logfire
from lpn.src.models.transformer import EncoderTransformer, DecoderTransformer
from lpn.src.models.lpn import LPN
from lpn.src.evaluator import Evaluator

from pydantic import BaseModel, TypeAdapter

class ChallengeSolutionWithAccuracy(BaseModel):
    attempt_1: list[GRID]
    attempt_2: list[GRID]
    accuracy_1: float
    accuracy_2: float

class ChallengeSolution(BaseModel):
    attempt_1: GRID
    attempt_2: GRID

def load_lpn_model(artifact_path: str) -> tuple[LPN, Evaluator]:
    run = wandb.init()
    artifact = run.use_artifact(artifact_path, type='model')
    run.finish()
    cfg = omegaconf.OmegaConf.create(artifact.metadata)
    artifact_dir = artifact.download()
    omegaconf.OmegaConf.save(config=cfg, f=os.path.join(artifact_dir, "config.yaml"))

    cfg.encoder_transformer['_target_'] = 'lpn.src.models.utils.EncoderTransformerConfig'
    cfg.encoder_transformer.transformer_layer['_target_'] = 'lpn.src.models.utils.TransformerLayerConfig'

    cfg.decoder_transformer['_target_'] = 'lpn.src.models.utils.DecoderTransformerConfig'
    cfg.decoder_transformer.transformer_layer['_target_'] = 'lpn.src.models.utils.TransformerLayerConfig'

    encoder = EncoderTransformer(hydra.utils.instantiate(cfg.encoder_transformer))
    decoder = DecoderTransformer(hydra.utils.instantiate(cfg.decoder_transformer))

    lpn_model = LPN(encoder=encoder, decoder=decoder)

    key = jax.random.PRNGKey(0)
    grids = jax.random.randint(
        key, (1, 3, decoder.config.max_rows, decoder.config.max_cols, 2), minval=0, maxval=decoder.config.vocab_size,
    )
    shapes = jax.random.randint(
        key, (1, 3, 2, 2), minval=1, maxval=min(decoder.config.max_rows, decoder.config.max_cols) + 1,
    )
    variables = lpn_model.init(key, grids, shapes, dropout_eval=False, prior_kl_coeff=0.0, pairwise_kl_coeff=0.0, mode="mean")
    evaluator = Evaluator(
        lpn_model,
        inference_mode="gradient_ascent",
        inference_mode_kwargs={
            "num_steps": 200,
            "lr": 1.0,
            "lr_schedule": True,
            "optimizer": "adam",
            "optimizer_kwargs": {"b2": 0.9},
            "accumulate_gradients_decoder_pairs": True,
        },
    )

    learning_rate, linear_warmup_steps = 0, 0
    linear_warmup_scheduler = optax.warmup_exponential_decay_schedule(
        init_value=learning_rate / (linear_warmup_steps + 1),
        peak_value=learning_rate,
        warmup_steps=linear_warmup_steps,
        transition_steps=1,
        end_value=learning_rate,
        decay_rate=1.0,
    )
    optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(linear_warmup_scheduler))
    optimizer = optax.MultiSteps(optimizer, every_k_schedule=1)
    train_state = TrainState.create(apply_fn=lpn_model.apply, tx=optimizer, params=variables["params"])

    ckpt_path = "state.msgpack"

    with open(os.path.join(artifact_dir, ckpt_path), "rb") as data_file:
        byte_data = data_file.read()
    loaded_state = from_bytes(train_state, byte_data)

    # Explicitly bind the model to the parameters
    bound_lpn_model = lpn_model.bind({'params': loaded_state.params})
    return bound_lpn_model, evaluator, key


async def main() -> None:
    # Add argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--eval", action="store_true", help="Use saved library")
    parser.add_argument("-l", "--lpn", type=str, help="Use LPN model")
    parser.add_argument("-v1", "--version1", action="store_true", help="Test on ARC-AGI-1 public eval set")
    parser.add_argument("-p", "--path", type=str, help="Input dataset file path")
    args = parser.parse_args()

    if args.lpn:
        artifact_path = os.getenv("LPN_ARTIFACT_PATH")
        lpn_model, evaluator, key = load_lpn_model(artifact_path)
    else:
        lpn_model, evaluator, key = None, None, None

    num_correct: int = 0
    num_tested: int = 0
    total_cost_in_cents: list[float] = [ 0.0 ]

    if args.version1:
        challenges = eval_challenges
    elif args.path:
        print(f"Building challenges from {args.path}")
        challenges = build_challenges(
            challenges_path=Path(args.path),
            solutions_path=None,
        )
    else:
        challenges = v2_eval_challenges

    eval_ids_to_test = list(challenges.keys())
    
    print(f"eval set size: {len(eval_ids_to_test)}")
    logfire.debug(f"eval set size: {len(eval_ids_to_test)}")
    debug(eval_ids_to_test)

    from src.trees.experiments import grok_dreamcoder_tree
    from src.models import Library

    # Function to load library
    def load_library(filename="saved_library.pkl"):
        if os.path.exists(filename):
            with open(filename, "rb") as f:
                return pickle.load(f)
        else:
            print(f"No library file found at {filename}, creating new library")
            return Library(primitives=[])

    # Function to save library
    def save_library(library, filename="saved_library.pkl"):
        with open(filename, "wb") as f:
            pickle.dump(library, f)
        print(f"Library saved to {filename}")

    def load_challenge_primitive_accuracy_scores(filename="challenge_primitive_accuracy_scores.pkl"):
        if os.path.exists(filename):
            with open(filename, "rb") as f:
                return pickle.load(f)
        else:
            return defaultdict(dict)

    # Only use library if -e flag is provided
    library = None
    if args.eval:
        # library of primitives that have been obtained from arc-agi 2 train set (1000 challenges)
        library_path = "saved_library_1000.pkl"
        library = load_library(library_path)
    else:
        library = Library(primitives=[])

    print(f"library size: {len(library.primitives)}")
    logfire.debug(f"library size: {len(library.primitives)}")

    solved_challenges = set()
    # Dictionary to store primitive lpn scores for each challenge (scores don't change across runs)
    challenge_primitive_lpn_scores = defaultdict(dict)
    # Dictionary to store primitive naive accuracy scores (how many squares it gets correct)
    #challenge_primitive_accuracy_scores = load_challenge_primitive_accuracy_scores()
    challenge_primitive_accuracy_scores = defaultdict(dict)
    #print(f"challenge_primitive_accuracy_scores length: {len(challenge_primitive_accuracy_scores)}")
    intermediate_solutions_d: dict[str, ChallengeSolutionWithAccuracy] = {}
    solutions_d: dict[str, list[ChallengeSolution]] = {}

    async def try_solve_challenge(challenge_id: str, solved_challenges: set[str], total_cost_in_cents: float) -> bool:
        if challenge_id in solved_challenges:
            print(f"Challenge {challenge_id} already solved")
            logfire.debug(f"Challenge {challenge_id} already solved")
            return True
        debug(challenge_id)
        print(f"value length: {len(challenge_primitive_accuracy_scores[challenge_id])}")
        challenge = challenges[challenge_id]

        solutions_and_accuracies = await solve_challenge_with_accuracy(
            challenge=challenge,
            tree=grok_dreamcoder_tree,
            library=library,
            use_primitives_weighed_by_score=not lpn_model,
            lpn_model=lpn_model,
            evaluator=evaluator,
            key=key,
            challenge_primitive_lpn_scores=challenge_primitive_lpn_scores,
            challenge_primitive_accuracy_scores=challenge_primitive_accuracy_scores,
            aggregate_cost_in_cents=total_cost_in_cents,
        )

        logfire.debug(f"[{challenge.id}] solutions_and_accuracies: {solutions_and_accuracies}")
        print(f"[{challenge.id}] solutions_and_accuracies: {solutions_and_accuracies}")

        first_solutions_and_accuracy, second_solutions_and_accuracy = solutions_and_accuracies[0], solutions_and_accuracies[1]

        logfire.debug(f"[{challenge.id}] tuple 1: {first_solutions_and_accuracy}, tuple 2: {second_solutions_and_accuracy}")
        print(f"[{challenge.id}] tuple 1: {first_solutions_and_accuracy}, tuple 2: {second_solutions_and_accuracy}")

        first_solutions, first_accuracy, first_code = first_solutions_and_accuracy
        second_solutions, second_accuracy, second_code = second_solutions_and_accuracy

        logfire.debug(f"[{challenge.id}] first_solutions: {first_solutions}, first_accuracy: {first_accuracy}")
        logfire.debug(f"[{challenge.id}] second_solutions: {second_solutions}, second_accuracy: {second_accuracy}")
        print(f"[{challenge.id}] first_solutions: {first_solutions}, first_accuracy: {first_accuracy}")
        print(f"[{challenge.id}] second_solutions: {second_solutions}, second_accuracy: {second_accuracy}")

        first_solution_correct_length, second_solution_correct_length = True, True
        if len(challenge.test) != len(first_solutions):
            print(f"[{challenge.id}] first_solutions have len {len(first_solutions)} but challenge.test has len {len(challenge.test)}")
            first_solution_correct_length = False
        if len(challenge.test) != len(second_solutions):
            print(f"[{challenge.id}] second_solutions have len {len(second_solutions)} but challenge.test has len {len(challenge.test)}")
            second_solution_correct_length = False

        if challenge.id not in intermediate_solutions_d:
            if first_solution_correct_length and second_solution_correct_length:
                intermediate_solutions_d[challenge.id] = ChallengeSolutionWithAccuracy(
                    attempt_1=first_solutions,
                    attempt_2=second_solutions,
                    accuracy_1=first_accuracy,
                    accuracy_2=second_accuracy,
                )
            elif first_solution_correct_length:
                intermediate_solutions_d[challenge.id] = ChallengeSolutionWithAccuracy(
                    attempt_1=first_solutions,
                    attempt_2=first_solutions,
                    accuracy_1=first_accuracy,
                    accuracy_2=first_accuracy,
                )
            elif second_solution_correct_length:
                intermediate_solutions_d[challenge.id] = ChallengeSolutionWithAccuracy(
                    attempt_1=second_solutions,
                    attempt_2=second_solutions,
                    accuracy_1=second_accuracy,
                    accuracy_2=second_accuracy,
                )
            else:
                raise ValueError(f"[{challenge.id}] first_solution_correct_length: {first_solution_correct_length}, second_solution_correct_length: {second_solution_correct_length}. Challenge test length: {len(challenge.test)}")
        else:
            old_solutions = intermediate_solutions_d[challenge.id]
            lst = [ [old_solutions.accuracy_1, old_solutions.attempt_1], 
                   [old_solutions.accuracy_2, old_solutions.attempt_2] ]
            if first_solution_correct_length:
                lst.append([first_accuracy, first_solutions])
            if second_solution_correct_length:
                lst.append([second_accuracy, second_solutions])
            lst.sort(key=lambda x: x[0], reverse=True)
            intermediate_solutions_d[challenge.id] = ChallengeSolutionWithAccuracy(
                attempt_1=lst[0][1],
                attempt_2=lst[1][1],
                accuracy_1=lst[0][0],
                accuracy_2=lst[1][0],
            )

        solutions_d[challenge.id] = []
        for i in range(len(challenge.test)):
            solutions_d[challenge.id].append(
                ChallengeSolution(
                    attempt_1=intermediate_solutions_d[challenge.id].attempt_1[i],
                    attempt_2=intermediate_solutions_d[challenge.id].attempt_2[i],
                )
            )

        if first_accuracy == 1.0 and second_accuracy == 1.0:
            return True
        else:
            return False

    for i in range(2):
        batch_size = 60
        for j in range(0, len(eval_ids_to_test), batch_size):
            batch_eval_ids_to_test = eval_ids_to_test[j:j+batch_size]
            tasks = [
                try_solve_challenge(challenge_id, solved_challenges, total_cost_in_cents) 
                for challenge_id in batch_eval_ids_to_test
            ]

            # Execute all tasks in parallel
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for solved, challenge_id in zip(results, batch_eval_ids_to_test):
                if isinstance(solved, Exception):
                    print(f"Error solving challenge {challenge_id}: {solved}")
                    logfire.debug(f"Error solving challenge {challenge_id}: {solved}")
                elif solved:
                    solved_challenges.add(challenge_id)
                else:
                    print(f"Challenge {challenge_id} not solved")
                    logfire.debug(f"Challenge {challenge_id} not solved")
            print(f"Round {i+1} challenge {j+batch_size}, Correct SO FAR: {len(solved_challenges)} solved. {solved_challenges}")
            logfire.debug(f"Round {i+1} challenge {j+batch_size}, Correct SO FAR: {len(solved_challenges)} solved. {solved_challenges}")

        logfire.debug(f"After {i+1} rounds, Solved Challenges: {solved_challenges}")
        print(f"After {i+1} rounds, Solved Challenges: {solved_challenges}")
        logfire.debug(f"After {i+1} rounds, Correct Percent SO FAR: {len(solved_challenges) / len(eval_ids_to_test)}")
        print(f"After {i+1} rounds, Correct Percent SO FAR: {len(solved_challenges) / len(eval_ids_to_test)}")
        print("Saving library...")
        save_library(library, f"saved_library_eval_round_{i}.pkl")

        logfire.debug(f"After {i+1} rounds, Total cost in cents: {total_cost_in_cents[0]}")
        print(f"After {i+1} rounds, Total cost in cents: {total_cost_in_cents[0]}")


    print(f"FINAL: Solved Challenges: {solved_challenges}")
    print(f"FINAL: Correct Percent: {len(solved_challenges) / len(eval_ids_to_test)}")

    print(f"FINAL: Total cost in cents: {total_cost_in_cents[0]}")


    # finally, check if there are solutions to all challenges
    for challenge_id in eval_ids_to_test:
        if challenge_id not in solutions_d:
            print(f"Challenge {challenge_id} does not have a solution")
            logfire.debug(f"Challenge {challenge_id} does not have a solution")
            challenge = challenges[challenge_id]
            solutions_d[challenge_id] = []
            for _ in range(len(challenge.test)):
                solutions_d[challenge_id].append(
                    ChallengeSolution(
                        attempt_1=[[0]],
                        attempt_2=[[0]],
                    )
                )

    solutions_path = "submission.json"
    print(f"Saving solutions to {solutions_path}")
    open(solutions_path, "w").write(
        TypeAdapter(dict[str, list[ChallengeSolution]])
        .dump_json(solutions_d)
        .decode("utf-8")
    )


if __name__ == "__main__":
    asyncio.run(main())
