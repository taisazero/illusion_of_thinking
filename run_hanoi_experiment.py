#!/usr/bin/env python3
"""
Script to generate Tower of Hanoi solutions using LLMs for different disk counts.

This script:
1. Generates problems for different numbers of disks (3, 4, 5, 6, 7, 8, etc.)
2. Uses the Tower of Hanoi system prompt and user template
3. Generates solutions using LLMs via batch processing
4. Saves results as JSON files for evaluation

Usage:
    # Individual processing (default)
    python run_hanoi_experiment.py --llm anthropic --output-dir results/
    python run_hanoi_experiment.py --llm vllm --vllm-base-url http://localhost:8000/v1
    python run_hanoi_experiment.py --llm openai --openai-model gpt-4o
    python run_hanoi_experiment.py --llm gemini --gemini-model gemini-2.5-pro-preview-06-05
    
    # Batch processing (faster & cheaper for large datasets)
    python run_hanoi_experiment.py --llm anthropic --use-batch
    python run_hanoi_experiment.py --llm openai --use-batch
    python run_hanoi_experiment.py --llm vllm --use-batch
    # Note: Gemini automatically switches to individual mode even with --use-batch
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple
import re
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.llm_clients import OpenAIClient, VLLMClient, AnthropicClient, GeminiClient


# Tower of Hanoi System Prompt
SYSTEM_PROMPT = """You are a helpful assistant. Solve this puzzle for me.
There are three pegs and n disks of different sizes stacked on the first peg. The disks are numbered from 1 (smallest) to n (largest). Disk moves in this puzzle should follow:

1. Only one disk can be moved at a time.

2. Each move consists of taking the upper disk from one stack and placing it on top of another stack.

3. A larger disk may not be placed on top of a smaller disk.

The goal is to move the entire stack to the third peg.
Example: With 3 disks numbered 1 (smallest), 2, and 3 (largest), the initial state is [[3, 2, 1], [], []], and a solution might be:

moves = [[1, 0, 2], [2, 0, 1], [1, 2, 1], [3, 0, 2],
         [1, 1, 0], [2, 1, 2], [1, 0, 2]]

This means: Move disk 1 from peg 0 to peg 2, then move disk 2 from peg 0 to peg 1, and so on.
Requirements:

• When exploring potential solutions in your thinking process, always include the corresponding complete list of moves.

• The positions are 0-indexed (the leftmost peg is 0).

• Ensure your final answer includes the complete list of moves in the format:
  moves = [[disk_id, from_peg, to_peg], ...]"""


# User Prompt Template
USER_PROMPT_TEMPLATE = """I have a puzzle with {N} disks of different sizes with
Initial configuration:

• Peg 0: {N} (bottom), ... 2, 1 (top)

• Peg 1: (empty)

• Peg 2: (empty)

Goal configuration:

• Peg 0: (empty)

• Peg 1: (empty)

• Peg 2: {N} (bottom), ... 2, 1 (top)

Rules:

• Only one disk can be moved at a time.

• Only the top disk from any stack can be moved.

• A larger disk may not be placed on top of a smaller disk.

Find the sequence of moves to transform the initial configuration into the goal configuration."""


def create_hanoi_problems(min_disks: int = 3, max_disks: int = 8) -> List[Dict[str, Any]]:
    """Create Tower of Hanoi problems for different disk counts."""
    problems = []
    
    for n in range(min_disks, max_disks + 1):
        problem = {
            "id": f"hanoi_{n}",
            "disk_count": n,
            "title": f"Tower of Hanoi with {n} disks",
            "description": f"Solve Tower of Hanoi puzzle with {n} disks",
            "system_prompt": SYSTEM_PROMPT,
            "user_prompt": USER_PROMPT_TEMPLATE.format(N=n)
        }
        problems.append(problem)
    
    return problems


def create_prompt_messages(problem: Dict[str, Any]) -> List[Dict[str, str]]:
    """Create messages for LLM API call."""
    messages = [
        {"role": "system", "content": problem["system_prompt"]},
        {"role": "user", "content": problem["user_prompt"]}
    ]
    return messages


def parse_llm_response(response: str) -> Dict[str, Any]:
    """Parse LLM response and extract moves and reasoning."""
    result = {
        "reasoning": "",
        "moves": [],
        "moves_raw": "",
        "raw_response": response,
        "parse_success": False
    }
    
    try:
        # First, try to extract reasoning from XML tags (for Anthropic with thinking enabled)
        reasoning_match = re.search(r'<reasoning>\s*(.*?)\s*</reasoning>', response, re.DOTALL)
        if reasoning_match:
            result["reasoning"] = reasoning_match.group(1).strip()
            # Remove the reasoning section from response for move extraction
            response_for_moves = re.sub(r'<reasoning>.*?</reasoning>\s*', '', response, flags=re.DOTALL)
        else:
            response_for_moves = response
        
        # Try to extract moves using improved patterns for multi-line arrays
        moves_patterns = [
            # Pattern 1: moves = [ ... ] (handles multi-line arrays properly)
            r'moves\s*=\s*(\[(?:[^\[\]]+|\[[^\]]*\])*\])',
            # Pattern 2: Look for array starting with [[ and ending with ]] 
            r'(\[\s*\[(?:[^\[\]]+|\[[^\]]*\])*\]\s*\])',
            # Pattern 3: Fallback to simpler pattern
            r'moves\s*=\s*(\[.*?\])',
        ]
        
        moves_match = None
        for i, pattern in enumerate(moves_patterns):
            moves_match = re.search(pattern, response_for_moves, re.DOTALL)
            if moves_match:
                break
        
        if moves_match:
            moves_str = moves_match.group(1)
            result["moves_raw"] = moves_str
            
            # Try to parse as Python list
            try:
                import ast
                moves = ast.literal_eval(moves_str)
                result["moves"] = moves
                result["parse_success"] = True
            except Exception as ast_error:
                # Fallback: try to extract individual move patterns from the entire response
                move_pattern = r'\[(\d+),\s*(\d+),\s*(\d+)\]'
                individual_moves = re.findall(move_pattern, response_for_moves)  # Search entire response
                if individual_moves:
                    result["moves"] = [[int(disk), int(from_peg), int(to_peg)] 
                                     for disk, from_peg, to_peg in individual_moves]
                    result["parse_success"] = True
        else:
            # Last resort: extract all [x, y, z] patterns from response
            move_pattern = r'\[(\d+),\s*(\d+),\s*(\d+)\]'
            individual_moves = re.findall(move_pattern, response_for_moves)
            if individual_moves:
                result["moves"] = [[int(disk), int(from_peg), int(to_peg)] 
                                 for disk, from_peg, to_peg in individual_moves]
                result["parse_success"] = True
                result["moves_raw"] = f"Extracted {len(result['moves'])} individual moves"
        
        # If we haven't extracted reasoning from XML tags, extract from text before moves
        if not result["reasoning"]:
            if moves_match:
                reasoning_text = response_for_moves[:moves_match.start()].strip()
                # Remove common prefixes
                reasoning_text = re.sub(r'^(thinking|reasoning|solution):\s*', '', reasoning_text, flags=re.IGNORECASE)
                result["reasoning"] = reasoning_text
            else:
                result["reasoning"] = response_for_moves.strip()
            
    except Exception as e:
        result["reasoning"] = f"Parse error: {str(e)}"
        print(f"Warning: Failed to parse LLM response: {e}")
    
    return result


def get_model_name(args, llm_client=None) -> str:
    """Get the model name being used."""
    if args.llm == "anthropic":
        return args.anthropic_model
    elif args.llm == "openai":
        return args.openai_model
    elif args.llm == "gemini":
        return args.gemini_model
    else:  # vllm
        if llm_client and hasattr(llm_client, 'model_name') and llm_client.model_name:
            return llm_client.model_name
        elif llm_client and hasattr(llm_client, 'get_llm_server_modelname'):
            model_name = llm_client.get_llm_server_modelname()
            return model_name if model_name else "vllm-unknown-model"
        else:
            return "vllm-served-model"


def get_llm_kwargs(args) -> Dict[str, Any]:
    """Get LLM-specific kwargs based on the selected provider."""
    if args.llm == "anthropic":
        # Check if model supports reasoning
        reasoning_supported = args.anthropic_model in ["claude-3-7-sonnet-latest", "claude-4-sonnet-latest", "claude-4-opus-latest"]
        return {
            "model": args.anthropic_model,
            "temperature": 1.0 if reasoning_supported else 0.1,
            "max_tokens": 32_000, # 10_000, # 100_000,
            "reasoning": reasoning_supported,
            "budget_tokens": 32_000 if reasoning_supported else 1000 # 6_400 # 64_000
        }
    elif args.llm == "openai":
        return {
            "model": args.openai_model,
            "temperature": 0.1,
            "max_tokens": 4000
        }
    elif args.llm == "gemini":
        return {
            "model": args.gemini_model,
            "temperature": 0.1,
            "max_tokens": 4000
        }
    else:  # vllm
        return {
            "temperature": 0.1,
            "max_tokens": 4000
        }


def create_llm_client(args) -> Any:
    """Create the appropriate LLM client based on arguments."""
    if args.llm == "anthropic":
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable not set")
        return AnthropicClient(api_key=api_key)
    
    elif args.llm == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        return OpenAIClient(api_key=api_key)
    
    elif args.llm == "vllm":
        return VLLMClient(
            api_key=args.vllm_api_key or "NONE",
            base_url=args.vllm_base_url
        )
    
    elif args.llm == "gemini":
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set")
        return GeminiClient(api_key=api_key, model=args.gemini_model)
    
    else:
        raise ValueError(f"Unsupported LLM: {args.llm}")


def generate_hanoi_batches(problems: List[Dict[str, Any]], num_trials: int = 5) -> List[Tuple[Dict[str, Any], List[Dict[str, str]]]]:
    """Generate batches of prompts for all Hanoi problems with multiple trials."""
    
    batches = []
    
    print(f"Preparing prompts for batch processing ({num_trials} trials per problem)...")
    
    for problem in tqdm(problems, desc="Processing problems"):
        for trial in range(num_trials):
            # Create messages for this problem
            messages = create_prompt_messages(problem)
            
            # Store metadata about this request
            metadata = {
                "problem_id": problem["id"],
                "disk_count": problem["disk_count"],
                "trial": trial,
                "title": problem["title"]
            }
            
            batches.append((metadata, messages))
    
    print(f"Prepared {len(batches)} prompts for processing")
    return batches


def save_results(results: List[Dict[str, Any]], output_dir: str, llm_info: Dict[str, str]):
    """Save results to JSON files organized by disk count."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Group results by disk count
    grouped = {}
    
    for result in results:
        disk_count = result["metadata"]["disk_count"]
        problem_id = result["metadata"]["problem_id"]
        
        if problem_id not in grouped:
            grouped[problem_id] = {
                "problem_id": problem_id,
                "disk_count": disk_count,
                "title": result["metadata"]["title"],
                "trials": []
            }
        
        grouped[problem_id]["trials"].append({
            "trial": result["metadata"]["trial"],
            "moves": result["parsed_response"]["moves"],
            "moves_raw": result["parsed_response"]["moves_raw"],
            "reasoning": result["parsed_response"]["reasoning"],
            "raw_response": result["parsed_response"]["raw_response"],
            "parse_success": result["parsed_response"]["parse_success"]
        })
    
    # Save each group to a separate file
    for problem_id, data in grouped.items():
        output_file = os.path.join(output_dir, f"{problem_id}.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"Saved {len(grouped)} result files to {output_dir}")
    
    # Save summary statistics
    total_trials = len(results)
    successful_parses = sum(1 for r in results if r["parsed_response"]["parse_success"])
    
    summary = {
        "total_trials": total_trials,
        "total_problems": len(grouped),
        "parse_success_rate": successful_parses / total_trials if total_trials > 0 else 0,
        "llm_provider": llm_info["provider"],
        "llm_model": llm_info["model"],
        "processing_mode": llm_info["processing_mode"],
        "problems_summary": {
            problem_id: {
                "disk_count": data["disk_count"],
                "trials": len(data["trials"]),
                "successful_parses": sum(1 for t in data["trials"] if t["parse_success"])
            }
            for problem_id, data in grouped.items()
        }
    }
    
    with open(os.path.join(output_dir, "summary.json"), 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Summary: {summary['parse_success_rate']:.2%} parse success across {total_trials} trials")
    print(f"LLM: {llm_info['provider']} ({llm_info['model']}) - {llm_info['processing_mode']} mode")


def main():
    parser = argparse.ArgumentParser(description="Generate Tower of Hanoi solutions using LLMs")
    
    # LLM selection
    parser.add_argument("--llm", choices=["anthropic", "openai", "vllm", "gemini"], 
                       default="anthropic", help="LLM provider to use")
    
    # LLM-specific arguments
    parser.add_argument("--vllm-base-url", default="http://localhost:8000/v1",
                       help="Base URL for vLLM server")
    parser.add_argument("--vllm-api-key", default="NONE",
                       help="API key for vLLM server (if needed)")
    parser.add_argument("--openai-model", default="gpt-4o",
                        choices=["gpt-4o", "gpt-4o-mini", "gpt-4-turbo"],
                       help="OpenAI model name")
    parser.add_argument("--anthropic-model", default="claude-3-7-sonnet-latest",
                        choices=["claude-3-7-sonnet-latest", "claude-4-sonnet-latest", "claude-4-opus-latest", "claude-3-5-sonnet-20241022", "claude-3-opus-20240229", "claude-3-haiku-20240307"],
                       help="Anthropic model name (claude-3-7-sonnet-latest, claude-4-sonnet-latest, claude-4-opus-latest support reasoning)")
    parser.add_argument("--gemini-model", 
                       choices=["gemini-2.0-flash", "gemini-1.5-pro", "gemini-1.5-flash"],
                       default="gemini-2.0-flash",
                       help="Gemini model name")
    
    # Problem configuration
    parser.add_argument("--min-disks", type=int, default=3,
                       help="Minimum number of disks to test")
    parser.add_argument("--max-disks", type=int, default=8,
                       help="Maximum number of disks to test")
    parser.add_argument("--num-trials", type=int, default=5,
                       help="Number of trials per problem")
    
    # Output
    parser.add_argument("--output-dir", default="illusion_thinking_exp/hanoi_results",
                       help="Output directory for results")
    
    # Processing options
    parser.add_argument("--use-batch", action="store_true",
                       help="Enable batch processing (faster for large datasets)")
    parser.add_argument("--debug-prompt", action="store_true",
                       help="Save first generated prompt to debug.txt for inspection")
    
    args = parser.parse_args()
    
    # Create problems
    print(f"Creating Tower of Hanoi problems (disks: {args.min_disks}-{args.max_disks})...")
    problems = create_hanoi_problems(args.min_disks, args.max_disks)
    print(f"Created {len(problems)} problems")
    
    # Create LLM client
    print(f"Initializing {args.llm} LLM client...")
    try:
        llm_client = create_llm_client(args)
    except Exception as e:
        print(f"Error creating LLM client: {e}")
        return 1
    
    # Generate all prompts
    batches = generate_hanoi_batches(problems, args.num_trials)
    
    # Debug: save first prompt if requested
    if args.debug_prompt and batches:
        debug_prompt = f"System: {batches[0][1][0]['content']}\n\nUser: {batches[0][1][1]['content']}"
        with open("debug_hanoi_prompt.txt", 'w', encoding='utf-8') as f:
            f.write(debug_prompt)
        print(f"🔍 Debug: First prompt saved to debug_hanoi_prompt.txt")
    
    if not batches:
        print("No prompts generated. Check your configuration.")
        return 1
    
    # Process messages (batch or individual)
    all_results = []
    
    if args.use_batch:
        # Check if provider supports batch processing
        if args.llm == "gemini":
            print("⚠️ Gemini does not support batch processing. Switching to individual mode.")
            args.use_batch = False
        else:
            print(f"🔄 Processing {len(batches)} requests as one large batch")
            
            # Process entire dataset as one batch
            all_metadata = [item[0] for item in batches]
            all_messages = [item[1] for item in batches]
            
            try:
                kwargs = get_llm_kwargs(args)
                print(f"Submitting batch of {len(all_messages)} requests to {args.llm}...")
                
                # Extract reasoning parameters for Anthropic client
                if args.llm == "anthropic":
                    reasoning = kwargs.pop("reasoning", False)
                    budget_tokens = kwargs.pop("budget_tokens", 1000)
                    responses = llm_client.create_batch_messages(all_messages, reasoning=reasoning, budget_tokens=budget_tokens, **kwargs)
                else:
                    responses = llm_client.create_batch_messages(all_messages, **kwargs)
                print(f"✅ Batch processing completed, received {len(responses)} responses")
            except Exception as e:
                print(f"Error processing batch: {e}")
                return 1
            
            # Parse and store results
            for metadata, response in zip(all_metadata, responses):
                parsed_response = parse_llm_response(response)
                
                result = {
                    "metadata": metadata,
                    "parsed_response": parsed_response
                }
                all_results.append(result)
    
    if not args.use_batch:
        print(f"🔄 Processing {len(batches)} requests individually")
        
        # Process each request individually
        for i, (metadata, messages) in enumerate(tqdm(batches, desc="Processing requests")):
            try:
                kwargs = get_llm_kwargs(args)
                
                # Process single message
                if hasattr(llm_client, 'create_message'):
                    if args.llm == "anthropic":
                        reasoning = kwargs.pop("reasoning", False)
                        budget_tokens = kwargs.pop("budget_tokens", 1000)
                        response = llm_client.create_message(messages, kwargs=kwargs, reasoning=reasoning, budget_tokens=budget_tokens)
                    else:
                        response = llm_client.create_message(messages, kwargs=kwargs)
                else:
                    # Fallback to batch with single item
                    if args.llm == "anthropic":
                        reasoning = kwargs.pop("reasoning", False)
                        budget_tokens = kwargs.pop("budget_tokens", 1000)
                        responses = llm_client.create_batch_messages([messages], reasoning=reasoning, budget_tokens=budget_tokens, **kwargs)
                    else:
                        responses = llm_client.create_batch_messages([messages], **kwargs)
                    response = responses[0]
                    
            except Exception as e:
                print(f"Error processing request {i+1}: {e}")
                response = f"Error: {e}"
            
            parsed_response = parse_llm_response(response)
            
            result = {
                "metadata": metadata,
                "parsed_response": parsed_response
            }
            all_results.append(result)
    
    # Save results
    print("Saving results...")
    
    # Prepare LLM info for summary
    llm_info = {
        "provider": args.llm,
        "model": get_model_name(args, llm_client),
        "processing_mode": "batch" if args.use_batch else "individual"
    }
    
    save_results(all_results, args.output_dir, llm_info)
    
    print("✅ Tower of Hanoi experiment completed successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main()) 