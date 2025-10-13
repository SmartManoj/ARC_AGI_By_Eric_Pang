import asyncio
import json
import time
import os
from pathlib import Path
from src.data import build_challenges
from src.logic import solve_challenge_with_accuracy
from src.trees.experiments import grok_dreamcoder_tree
from src.models import Library
from collections import defaultdict

async def test_single_task(task_id: str, enable_streaming: bool = False):
    """Test a single ARC task by ID
    
    Args:
        task_id: The ARC task ID to test
        enable_streaming: If True, enables streaming output from LLM calls
    """
    
    # Set streaming environment variable if enabled
    if enable_streaming:
        os.environ["STREAM_LLM"] = "1"
        print("✓ Streaming enabled for LLM responses\n")
    
    # Load the task from evaluation set
    task_path = Path(f"arc-agi-2/evaluation/{task_id}.json")
    if not task_path.exists():
        print(f"Task {task_id} not found in evaluation set")
        return
    
    # Read the task data
    with open(task_path, 'r') as f:
        task_data = json.load(f)
    
    print(f"\n{'='*60}")
    print(f"Testing Task: {task_id}")
    print(f"{'='*60}")
    print(f"Training examples: {len(task_data['train'])}")
    print(f"Test examples: {len(task_data['test'])}")
    
    # Print training example details
    for i, train in enumerate(task_data['train']):
        print(f"\nTraining Example {i+1}:")
        print(f"  Input shape: {len(train['input'])}x{len(train['input'][0])}")
        print(f"  Output shape: {len(train['output'])}x{len(train['output'][0]) if train['output'] else 0}")
    
    # Print test example details
    for i, test in enumerate(task_data['test']):
        print(f"\nTest Example {i+1}:")
        print(f"  Input shape: {len(test['input'])}x{len(test['input'][0])}")
        print(f"  Expected output shape: {len(test['output'])}x{len(test['output'][0]) if test['output'] else 0}")
    
    # Build challenge
    challenges = {}
    from src.models import Challenge, Example
    
    train_examples = [
        Example(input=train['input'], output=train['output']) 
        for train in task_data['train']
    ]
    test_examples = [
        Example(input=test['input'], output=test['output']) 
        for test in task_data['test']
    ]
    
    challenge = Challenge(
        id=task_id,
        train=train_examples,
        test=test_examples
    )
    
    # Initialize library
    library = Library(primitives=[])
    
    # Initialize tracking dictionaries
    challenge_primitive_lpn_scores = defaultdict(dict)
    challenge_primitive_accuracy_scores = defaultdict(dict)
    total_cost_in_cents = [0.0]
    
    print(f"\n{'='*60}")
    print("Running solver...")
    print(f"{'='*60}\n")
    
    start_time = time.time()
    
    try:
        solutions_and_accuracies = await solve_challenge_with_accuracy(
            challenge=challenge,
            tree=grok_dreamcoder_tree,
            library=library,
            use_primitives_weighed_by_score=True,
            lpn_model=None,
            evaluator=None,
            key=None,
            challenge_primitive_lpn_scores=challenge_primitive_lpn_scores,
            challenge_primitive_accuracy_scores=challenge_primitive_accuracy_scores,
            aggregate_cost_in_cents=total_cost_in_cents,
        )
        
        end_time = time.time()
        total_time = end_time - start_time
        
        print(f"\n{'='*60}")
        print("Results:")
        print(f"{'='*60}")
        
        first_solutions_and_accuracy, second_solutions_and_accuracy = solutions_and_accuracies[0], solutions_and_accuracies[1]
        first_solutions, first_accuracy, first_code = first_solutions_and_accuracy
        second_solutions, second_accuracy, second_code = second_solutions_and_accuracy
        
        print(f"\nAttempt 1 Accuracy: {first_accuracy:.2%}")
        print(f"Attempt 2 Accuracy: {second_accuracy:.2%}")
        
        if first_accuracy == 1.0:
            print(f"\n✓ Task SOLVED on attempt 1!")
        elif second_accuracy == 1.0:
            print(f"\n✓ Task SOLVED on attempt 2!")
        else:
            print(f"\n✗ Task NOT SOLVED (best accuracy: {max(first_accuracy, second_accuracy):.2%})")
        
        print(f"\nTotal execution time: {total_time:.2f} seconds")
        print(f"Total cost: ${total_cost_in_cents[0]/100:.4f}")
        
        # Show the first solution program
        if first_code:
            print(f"\n{'='*60}")
            print("First Solution Program:")
            print(f"{'='*60}")
            print(first_code)
        
        # Show the solutions
        if len(first_solutions) > 0:
            print(f"\n{'='*60}")
            print(f"First solution output shape: {len(first_solutions[0])}x{len(first_solutions[0][0]) if first_solutions[0] else 0}")
            if len(first_solutions[0]) <= 30:
                print("\nFirst solution output:")
                for row in first_solutions[0]:
                    print(row)
        
    except Exception as e:
        print(f"\n✗ Error solving task: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    import sys
    
    # Parse command line arguments
    task_id = "7b5033c1"
    enable_streaming = True  # Streaming is now default
    
    if len(sys.argv) >= 2:
        task_id = sys.argv[1]
    
    # Check for --no-stream flag to disable streaming
    if "--no-stream" in sys.argv:
        enable_streaming = False
    
    asyncio.run(test_single_task(task_id, enable_streaming))

