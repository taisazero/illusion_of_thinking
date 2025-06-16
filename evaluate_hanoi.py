#!/usr/bin/env python3
"""
Evaluation script for Tower of Hanoi solutions.

This script evaluates LLM-generated solutions to Tower of Hanoi problems by:
1. Loading results from JSON files
2. Simulating each move sequence
3. Checking if the solution is valid and optimal
4. Computing success rates and statistics

Usage:
    python evaluate_hanoi.py --results-dir illusion_thinking_exp/hanoi_results
    python evaluate_hanoi.py --results-dir results/ --output-file evaluation_report.json
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import re
from collections import defaultdict


class HanoiSimulator:
    """Simulator for Tower of Hanoi puzzles."""
    
    def __init__(self, num_disks: int):
        self.num_disks = num_disks
        self.reset()
    
    def reset(self):
        """Reset to initial state: all disks on peg 0."""
        self.pegs = [
            list(range(self.num_disks, 0, -1)),  # Peg 0: [n, n-1, ..., 2, 1]
            [],  # Peg 1: empty
            []   # Peg 2: empty
        ]
        self.move_count = 0
        self.move_history = []
    
    def is_valid_move(self, disk: int, from_peg: int, to_peg: int) -> Tuple[bool, str]:
        """Check if a move is valid."""
        # Check peg indices
        if not all(0 <= peg <= 2 for peg in [from_peg, to_peg]):
            return False, f"Invalid peg index: from={from_peg}, to={to_peg}"
        
        if from_peg == to_peg:
            return False, f"Cannot move disk to same peg: {from_peg}"
        
        # Check if from_peg has disks
        if not self.pegs[from_peg]:
            return False, f"No disks on peg {from_peg}"
        
        # Check if disk is on top of from_peg
        top_disk = self.pegs[from_peg][-1]
        if top_disk != disk:
            return False, f"Disk {disk} is not on top of peg {from_peg} (top disk is {top_disk})"
        
        # Check if we can place disk on to_peg (not on smaller disk)
        if self.pegs[to_peg] and self.pegs[to_peg][-1] < disk:
            return False, f"Cannot place disk {disk} on smaller disk {self.pegs[to_peg][-1]}"
        
        return True, "Valid move"
    
    def make_move(self, disk: int, from_peg: int, to_peg: int) -> Tuple[bool, str]:
        """Make a move if valid."""
        is_valid, message = self.is_valid_move(disk, from_peg, to_peg)
        
        if not is_valid:
            return False, message
        
        # Make the move
        moved_disk = self.pegs[from_peg].pop()
        self.pegs[to_peg].append(moved_disk)
        self.move_count += 1
        self.move_history.append([disk, from_peg, to_peg])
        
        return True, f"Moved disk {disk} from peg {from_peg} to peg {to_peg}"
    
    def is_solved(self) -> bool:
        """Check if puzzle is solved (all disks on peg 2)."""
        return (len(self.pegs[2]) == self.num_disks and 
                self.pegs[2] == list(range(self.num_disks, 0, -1)))
    
    def get_state(self) -> List[List[int]]:
        """Get current state of pegs."""
        return [peg.copy() for peg in self.pegs]
    
    def optimal_moves(self) -> int:
        """Calculate optimal number of moves for n disks."""
        return 2 ** self.num_disks - 1


def evaluate_solution(moves: List[List[int]], num_disks: int) -> Dict[str, Any]:
    """Evaluate a single Tower of Hanoi solution."""
    
    result = {
        "valid": False,
        "solved": False,
        "optimal": False,
        "move_count": len(moves),
        "optimal_moves": 2 ** num_disks - 1,
        "error_message": "",
        "error_move_index": -1,
        "final_state": None,
        "efficiency_ratio": 0.0
    }
    
    if not moves:
        result["error_message"] = "No moves provided"
        return result
    
    # Create simulator
    simulator = HanoiSimulator(num_disks)
    
    try:
        # Execute each move
        for i, move in enumerate(moves):
            if len(move) != 3:
                result["error_message"] = f"Invalid move format at index {i}: {move}"
                result["error_move_index"] = i
                return result
            
            disk, from_peg, to_peg = move
            
            # Validate move
            success, message = simulator.make_move(disk, from_peg, to_peg)
            if not success:
                result["error_message"] = f"Invalid move at index {i}: {message}"
                result["error_move_index"] = i
                result["final_state"] = simulator.get_state()
                return result
        
        # Check if solved
        result["valid"] = True
        result["solved"] = simulator.is_solved()
        result["final_state"] = simulator.get_state()
        
        if result["solved"]:
            result["optimal"] = (len(moves) == result["optimal_moves"])
            result["efficiency_ratio"] = result["optimal_moves"] / len(moves) if len(moves) > 0 else 0
        
    except Exception as e:
        result["error_message"] = f"Simulation error: {str(e)}"
    
    return result


def load_results(results_dir: str) -> Dict[str, Any]:
    """Load all result files from directory."""
    results = {}
    
    if not os.path.exists(results_dir):
        print(f"Results directory does not exist: {results_dir}")
        return results
    
    for filename in os.listdir(results_dir):
        if filename.endswith('.json') and filename.startswith('hanoi_'):
            filepath = os.path.join(results_dir, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    results[filename] = data
            except Exception as e:
                print(f"Error loading {filename}: {e}")
    
    return results


def evaluate_all_results(results: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluate all loaded results."""
    
    evaluation_results = {
        "summary": {
            "total_problems": 0,
            "total_trials": 0,
            "valid_solutions": 0,
            "solved_problems": 0,
            "optimal_solutions": 0,
            "parse_success": 0,
            "by_disk_count": {}
        },
        "detailed_results": {}
    }
    
    for filename, data in results.items():
        problem_id = data["problem_id"]
        disk_count = data["disk_count"]
        trials = data["trials"]
        
        evaluation_results["summary"]["total_problems"] += 1
        evaluation_results["summary"]["total_trials"] += len(trials)
        
        # Initialize disk count stats if not exists
        if disk_count not in evaluation_results["summary"]["by_disk_count"]:
            evaluation_results["summary"]["by_disk_count"][disk_count] = {
                "total_trials": 0,
                "parse_success": 0,
                "valid_solutions": 0,
                "solved_problems": 0,
                "optimal_solutions": 0,
                "avg_efficiency": 0.0
            }
        
        disk_stats = evaluation_results["summary"]["by_disk_count"][disk_count]
        
        # Evaluate each trial
        trial_evaluations = []
        efficiencies = []
        
        for trial in trials:
            disk_stats["total_trials"] += 1
            
            if trial["parse_success"]:
                evaluation_results["summary"]["parse_success"] += 1
                disk_stats["parse_success"] += 1
                
                # Evaluate the solution
                eval_result = evaluate_solution(trial["moves"], disk_count)
                trial_evaluations.append(eval_result)
                
                if eval_result["valid"]:
                    evaluation_results["summary"]["valid_solutions"] += 1
                    disk_stats["valid_solutions"] += 1
                    
                    if eval_result["solved"]:
                        evaluation_results["summary"]["solved_problems"] += 1
                        disk_stats["solved_problems"] += 1
                        efficiencies.append(eval_result["efficiency_ratio"])
                        
                        if eval_result["optimal"]:
                            evaluation_results["summary"]["optimal_solutions"] += 1
                            disk_stats["optimal_solutions"] += 1
            else:
                # No valid moves to evaluate
                trial_evaluations.append({
                    "valid": False,
                    "solved": False,
                    "optimal": False,
                    "error_message": "Failed to parse moves from response",
                    "move_count": 0,
                    "optimal_moves": 2 ** disk_count - 1
                })
        
        # Calculate average efficiency for this disk count
        if efficiencies:
            disk_stats["avg_efficiency"] = sum(efficiencies) / len(efficiencies)
        
        # Store detailed results
        evaluation_results["detailed_results"][problem_id] = {
            "disk_count": disk_count,
            "trials": trial_evaluations,
            "original_data": data
        }
    
    return evaluation_results


def generate_report(evaluation_results: Dict[str, Any]) -> str:
    """Generate a human-readable evaluation report."""
    
    summary = evaluation_results["summary"]
    
    report = []
    report.append("=" * 60)
    report.append("TOWER OF HANOI EVALUATION REPORT")
    report.append("=" * 60)
    report.append("")
    
    # Overall summary
    report.append("OVERALL SUMMARY:")
    report.append(f"Total Problems: {summary['total_problems']}")
    report.append(f"Total Trials: {summary['total_trials']}")
    report.append(f"Parse Success Rate: {summary['parse_success']}/{summary['total_trials']} ({summary['parse_success']/summary['total_trials']*100:.1f}%)")
    report.append(f"Valid Solutions: {summary['valid_solutions']}/{summary['total_trials']} ({summary['valid_solutions']/summary['total_trials']*100:.1f}%)")
    report.append(f"Solved Problems: {summary['solved_problems']}/{summary['total_trials']} ({summary['solved_problems']/summary['total_trials']*100:.1f}%)")
    report.append(f"Optimal Solutions: {summary['optimal_solutions']}/{summary['total_trials']} ({summary['optimal_solutions']/summary['total_trials']*100:.1f}%)")
    report.append("")
    
    # By disk count
    report.append("BY DISK COUNT:")
    report.append("-" * 40)
    
    for disk_count in sorted(summary["by_disk_count"].keys()):
        stats = summary["by_disk_count"][disk_count]
        report.append(f"Disks: {disk_count}")
        report.append(f"  Trials: {stats['total_trials']}")
        report.append(f"  Parse Success: {stats['parse_success']}/{stats['total_trials']} ({stats['parse_success']/stats['total_trials']*100:.1f}%)")
        report.append(f"  Valid: {stats['valid_solutions']}/{stats['total_trials']} ({stats['valid_solutions']/stats['total_trials']*100:.1f}%)")
        report.append(f"  Solved: {stats['solved_problems']}/{stats['total_trials']} ({stats['solved_problems']/stats['total_trials']*100:.1f}%)")
        report.append(f"  Optimal: {stats['optimal_solutions']}/{stats['total_trials']} ({stats['optimal_solutions']/stats['total_trials']*100:.1f}%)")
        report.append(f"  Avg Efficiency: {stats['avg_efficiency']:.3f}")
        report.append(f"  Optimal Moves: {2**disk_count - 1}")
        report.append("")
    
    # Detailed analysis
    report.append("DETAILED ANALYSIS:")
    report.append("-" * 40)
    
    for problem_id in sorted(evaluation_results["detailed_results"].keys()):
        details = evaluation_results["detailed_results"][problem_id]
        disk_count = details["disk_count"]
        trials = details["trials"]
        
        report.append(f"Problem: {problem_id} ({disk_count} disks)")
        
        valid_count = sum(1 for t in trials if t["valid"])
        solved_count = sum(1 for t in trials if t["solved"])
        optimal_count = sum(1 for t in trials if t["optimal"])
        
        report.append(f"  Valid: {valid_count}/{len(trials)}")
        report.append(f"  Solved: {solved_count}/{len(trials)}")
        report.append(f"  Optimal: {optimal_count}/{len(trials)}")
        
        # Show common errors
        errors = [t["error_message"] for t in trials if t["error_message"]]
        if errors:
            error_counts = defaultdict(int)
            for error in errors:
                error_counts[error] += 1
            report.append("  Common errors:")
            for error, count in sorted(error_counts.items(), key=lambda x: x[1], reverse=True)[:3]:
                report.append(f"    {error} ({count}x)")
        
        report.append("")
    
    return "\n".join(report)


def main():
    parser = argparse.ArgumentParser(description="Evaluate Tower of Hanoi LLM solutions")
    
    parser.add_argument("--results-dir", required=True,
                       help="Directory containing result JSON files")
    parser.add_argument("--output-file", default=None,
                       help="Output file for detailed evaluation results (JSON)")
    parser.add_argument("--report-file", default=None,
                       help="Output file for human-readable report (TXT)")
    parser.add_argument("--verbose", action="store_true",
                       help="Show detailed output")
    
    args = parser.parse_args()
    
    # Load results
    print(f"Loading results from {args.results_dir}...")
    results = load_results(args.results_dir)
    
    if not results:
        print("No results found!")
        return 1
    
    print(f"Loaded {len(results)} result files")
    
    # Evaluate all results
    print("Evaluating solutions...")
    evaluation_results = evaluate_all_results(results)
    
    # Generate report
    report = generate_report(evaluation_results)
    
    # Output results
    if args.output_file:
        print(f"Saving detailed results to {args.output_file}...")
        with open(args.output_file, 'w', encoding='utf-8') as f:
            json.dump(evaluation_results, f, indent=2, ensure_ascii=False)
    
    if args.report_file:
        print(f"Saving report to {args.report_file}...")
        with open(args.report_file, 'w', encoding='utf-8') as f:
            f.write(report)
    
    # Always print report to console
    print("\n" + report)
    
    return 0


if __name__ == "__main__":
    sys.exit(main()) 