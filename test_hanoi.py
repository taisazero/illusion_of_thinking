#!/usr/bin/env python3
"""
Test script for the Tower of Hanoi experiment and evaluation functions.

This script tests:
1. The HanoiSimulator class
2. Solution evaluation functions
3. Example correct and incorrect solutions

Usage:
    python test_hanoi.py
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from evaluate_hanoi import HanoiSimulator, evaluate_solution


def test_hanoi_simulator():
    """Test the HanoiSimulator class."""
    print("Testing HanoiSimulator...")
    
    # Test 3-disk puzzle
    simulator = HanoiSimulator(3)
    
    # Initial state should be [[3, 2, 1], [], []]
    assert simulator.get_state() == [[3, 2, 1], [], []]
    assert not simulator.is_solved()
    
    # Test some valid moves
    success, msg = simulator.make_move(1, 0, 2)
    assert success, f"Move should be valid: {msg}"
    assert simulator.get_state() == [[3, 2], [], [1]]
    
    success, msg = simulator.make_move(2, 0, 1)
    assert success, f"Move should be valid: {msg}"
    assert simulator.get_state() == [[3], [2], [1]]
    
    # Test invalid move (placing larger disk on smaller)
    success, msg = simulator.make_move(2, 1, 2)
    assert not success, f"Move should be invalid: {msg}"
    
    print("✅ HanoiSimulator tests passed!")


def test_correct_solution():
    """Test evaluation of a correct 3-disk solution."""
    print("Testing correct solution evaluation...")
    
    # Correct solution for 3 disks (optimal: 7 moves)
    correct_moves = [
        [1, 0, 2],  # Move disk 1 from peg 0 to peg 2
        [2, 0, 1],  # Move disk 2 from peg 0 to peg 1
        [1, 2, 1],  # Move disk 1 from peg 2 to peg 1
        [3, 0, 2],  # Move disk 3 from peg 0 to peg 2
        [1, 1, 0],  # Move disk 1 from peg 1 to peg 0
        [2, 1, 2],  # Move disk 2 from peg 1 to peg 2
        [1, 0, 2]   # Move disk 1 from peg 0 to peg 2
    ]
    
    result = evaluate_solution(correct_moves, 3)
    
    assert result["valid"], "Solution should be valid"
    assert result["solved"], "Solution should solve the puzzle"
    assert result["optimal"], "Solution should be optimal"
    assert result["move_count"] == 7, f"Should have 7 moves, got {result['move_count']}"
    assert result["efficiency_ratio"] == 1.0, f"Efficiency should be 1.0, got {result['efficiency_ratio']}"
    
    print("✅ Correct solution evaluation passed!")


def test_incorrect_solution():
    """Test evaluation of an incorrect solution."""
    print("Testing incorrect solution evaluation...")
    
    # Invalid move sequence (trying to move disk that's not on top)
    invalid_moves = [
        [2, 0, 1],  # This should fail - disk 2 is not on top of peg 0
    ]
    
    result = evaluate_solution(invalid_moves, 3)
    
    assert not result["valid"], "Solution should be invalid"
    assert not result["solved"], "Solution should not solve the puzzle"
    assert result["error_move_index"] == 0, "Error should be at move index 0"
    
    print("✅ Incorrect solution evaluation passed!")


def test_suboptimal_solution():
    """Test evaluation of a suboptimal but correct solution."""
    print("Testing suboptimal solution evaluation...")
    
    # Suboptimal solution for 3 disks (more than 7 moves)
    suboptimal_moves = [
        [1, 0, 1],  # Move disk 1 to peg 1 (unnecessary)
        [1, 1, 2],  # Move disk 1 to peg 2
        [2, 0, 1],  # Move disk 2 to peg 1
        [1, 2, 1],  # Move disk 1 back to peg 1
        [3, 0, 2],  # Move disk 3 to peg 2
        [1, 1, 0],  # Move disk 1 to peg 0
        [2, 1, 2],  # Move disk 2 to peg 2
        [1, 0, 2]   # Move disk 1 to peg 2
    ]
    
    result = evaluate_solution(suboptimal_moves, 3)
    
    assert result["valid"], "Solution should be valid"
    assert result["solved"], "Solution should solve the puzzle"
    assert not result["optimal"], "Solution should not be optimal"
    assert result["move_count"] == 8, f"Should have 8 moves, got {result['move_count']}"
    assert result["efficiency_ratio"] == 7/8, f"Efficiency should be 7/8, got {result['efficiency_ratio']}"
    
    print("✅ Suboptimal solution evaluation passed!")


def print_example_solutions():
    """Print example solutions for different disk counts."""
    print("\nExample optimal solutions:")
    print("-" * 40)
    
    for n in range(3, 6):
        optimal_moves = 2**n - 1
        print(f"{n} disks: {optimal_moves} moves required")
    
    # Show the 3-disk solution step by step
    print("\n3-disk solution step by step:")
    simulator = HanoiSimulator(3)
    correct_moves = [
        [1, 0, 2], [2, 0, 1], [1, 2, 1], [3, 0, 2],
        [1, 1, 0], [2, 1, 2], [1, 0, 2]
    ]
    
    print(f"Initial: {simulator.get_state()}")
    for i, move in enumerate(correct_moves):
        disk, from_peg, to_peg = move
        simulator.make_move(disk, from_peg, to_peg)
        print(f"Move {i+1}: Disk {disk} from peg {from_peg} to peg {to_peg} -> {simulator.get_state()}")
    
    print(f"Solved: {simulator.is_solved()}")


def main():
    """Run all tests."""
    print("Running Tower of Hanoi tests...")
    print("=" * 50)
    
    try:
        test_hanoi_simulator()
        test_correct_solution()
        test_incorrect_solution()
        test_suboptimal_solution()
        
        print("\n🎉 All tests passed!")
        
        print_example_solutions()
        
    except AssertionError as e:
        print(f"❌ Test failed: {e}")
        return 1
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main()) 