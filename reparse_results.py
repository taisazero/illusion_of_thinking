#!/usr/bin/env python3
"""
Script to re-parse existing Tower of Hanoi results with improved parsing logic.

This script:
1. Loads existing result files
2. Re-parses the raw responses with improved regex patterns
3. Updates the results with correctly parsed moves
4. Saves the corrected results

Usage:
    python reparse_results.py --results-dir results/claude_hanoi
    python reparse_results.py --results-dir results/claude_hanoi --backup
"""

import argparse
import json
import os
import sys
import shutil
from pathlib import Path
from typing import Dict, List, Any
import re

# Import the improved parse function
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from run_hanoi_experiment import parse_llm_response


def reparse_result_file(filepath: str, backup: bool = False) -> Dict[str, Any]:
    """Re-parse a single result file."""
    
    # Load original data
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Create backup if requested
    if backup:
        backup_path = filepath + '.backup'
        shutil.copy2(filepath, backup_path)
        print(f"Created backup: {backup_path}")
    
    # Track changes
    total_trials = len(data["trials"])
    improved_count = 0
    
    # Re-parse each trial
    for trial in data["trials"]:
        if "raw_response" in trial:
            # Get original parsing results
            original_moves_count = len(trial.get("moves", []))
            original_parse_success = trial.get("parse_success", False)
            
            # Re-parse with improved logic
            print(f"\nRe-parsing trial {trial['trial']} for {data['problem_id']}...")
            new_result = parse_llm_response(trial["raw_response"])
            
            # Update the trial with new results
            trial["moves"] = new_result["moves"]
            trial["moves_raw"] = new_result["moves_raw"]
            trial["reasoning"] = new_result["reasoning"]
            trial["parse_success"] = new_result["parse_success"]
            
            # Check if we improved the parsing
            new_moves_count = len(new_result["moves"])
            new_parse_success = new_result["parse_success"]
            
            if (new_moves_count > original_moves_count or 
                (new_parse_success and not original_parse_success)):
                improved_count += 1
                print(f"  ✅ Improved: {original_moves_count} -> {new_moves_count} moves")
            else:
                print(f"  ➡️  No change: {original_moves_count} moves")
    
    # Save updated data
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"\nUpdated {filepath}: {improved_count}/{total_trials} trials improved")
    
    return {
        "file": filepath,
        "total_trials": total_trials,
        "improved_trials": improved_count,
        "improvement_rate": improved_count / total_trials if total_trials > 0 else 0
    }


def main():
    parser = argparse.ArgumentParser(description="Re-parse Tower of Hanoi results with improved parsing")
    
    parser.add_argument("--results-dir", required=True,
                       help="Directory containing result JSON files")
    parser.add_argument("--backup", action="store_true",
                       help="Create backup files before modifying")
    parser.add_argument("--pattern", default="hanoi_*.json",
                       help="File pattern to match (default: hanoi_*.json)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.results_dir):
        print(f"Results directory does not exist: {args.results_dir}")
        return 1
    
    # Find result files
    from glob import glob
    pattern = os.path.join(args.results_dir, args.pattern)
    result_files = glob(pattern)
    
    if not result_files:
        print(f"No result files found matching pattern: {pattern}")
        return 1
    
    print(f"Found {len(result_files)} result files to re-parse")
    
    # Re-parse each file
    file_stats = []
    total_improved = 0
    total_trials = 0
    
    for filepath in sorted(result_files):
        try:
            stats = reparse_result_file(filepath, args.backup)
            file_stats.append(stats)
            total_improved += stats["improved_trials"]
            total_trials += stats["total_trials"]
        except Exception as e:
            print(f"Error processing {filepath}: {e}")
    
    # Print summary
    print("\n" + "="*60)
    print("RE-PARSING SUMMARY")
    print("="*60)
    
    for stats in file_stats:
        filename = os.path.basename(stats["file"])
        print(f"{filename}: {stats['improved_trials']}/{stats['total_trials']} "
              f"({stats['improvement_rate']:.1%})")
    
    print(f"\nTotal: {total_improved}/{total_trials} trials improved "
          f"({total_improved/total_trials:.1%})")
    
    if args.backup:
        print(f"\nBackup files created with .backup extension")
    
    print("\n✅ Re-parsing completed!")
    return 0


if __name__ == "__main__":
    sys.exit(main()) 