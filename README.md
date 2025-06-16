# Tower of Hanoi LLM Experiment

This directory contains scripts for running Tower of Hanoi experiments with Large Language Models (LLMs).

## Overview

The experiment tests how well different LLMs can solve the Tower of Hanoi puzzle for various numbers of disks (3-8 by default). Each LLM is given:

1. A **system prompt** explaining the Tower of Hanoi rules and expected output format
2. A **user prompt** specifying the initial and goal configurations for N disks

The LLM must respond with a sequence of moves in the format:
```
moves = [[disk_id, from_peg, to_peg], ...]
```

## Files

- `run_hanoi_experiment.py` - Main script to generate solutions using LLMs
- `evaluate_hanoi.py` - Script to evaluate and analyze the generated solutions
- `test_hanoi.py` - Test script to verify the evaluation functions work correctly
- `README.md` - This documentation file

## Quick Start

### 1. Test the evaluation system
```bash
cd illusion_thinking_exp
python test_hanoi.py
```

### 2. Run a small experiment (3-5 disks, 3 trials each)
```bash
# Using Anthropic Claude
python run_hanoi_experiment.py --llm anthropic --max-disks 5 --num-trials 3

# Using OpenAI GPT-4
python run_hanoi_experiment.py --llm openai --openai-model gpt-4o --max-disks 5 --num-trials 3

# Using local vLLM server
python run_hanoi_experiment.py --llm vllm --vllm-base-url http://localhost:8000/v1 --max-disks 5 --num-trials 3
```

### 3. Evaluate the results
```bash
python evaluate_hanoi.py --results-dir hanoi_results --report-file evaluation_report.txt
```

## Detailed Usage

### Running Experiments

#### Basic Usage
```bash
python run_hanoi_experiment.py --llm PROVIDER [OPTIONS]
```

#### LLM Providers
- `--llm anthropic` - Uses Anthropic Claude (requires `ANTHROPIC_API_KEY`)
- `--llm openai` - Uses OpenAI GPT models (requires `OPENAI_API_KEY`)
- `--llm gemini` - Uses Google Gemini (requires `GEMINI_API_KEY`)
- `--llm vllm` - Uses local vLLM server

#### Key Options
- `--min-disks N` - Minimum number of disks to test (default: 3)
- `--max-disks N` - Maximum number of disks to test (default: 8)
- `--num-trials N` - Number of trials per problem (default: 5)
- `--output-dir DIR` - Output directory for results (default: `hanoi_results`)
- `--use-batch` - Enable batch processing (faster for large experiments)
- `--debug-prompt` - Save first prompt to file for inspection

#### Model-Specific Options
```bash
# OpenAI models
--openai-model gpt-4o|gpt-4o-mini|gpt-4-turbo

# Anthropic models  
--anthropic-model claude-3-5-sonnet-20241022|claude-3-opus-20240229|claude-3-haiku-20240307

# Gemini models
--gemini-model gemini-2.0-flash|gemini-1.5-pro|gemini-1.5-flash

# vLLM server
--vllm-base-url http://localhost:8000/v1
--vllm-api-key YOUR_KEY_IF_NEEDED
```

### Evaluating Results

#### Basic Evaluation
```bash
python evaluate_hanoi.py --results-dir hanoi_results
```

#### Advanced Options
```bash
python evaluate_hanoi.py \
    --results-dir hanoi_results \
    --output-file detailed_evaluation.json \
    --report-file human_readable_report.txt \
    --verbose
```

## Output Format

### Experiment Results
Results are saved as JSON files, one per problem:
- `hanoi_3.json` - Results for 3-disk problem
- `hanoi_4.json` - Results for 4-disk problem
- etc.

Each file contains:
```json
{
  "problem_id": "hanoi_3",
  "disk_count": 3,
  "title": "Tower of Hanoi with 3 disks",
  "trials": [
    {
      "trial": 0,
      "moves": [[1, 0, 2], [2, 0, 1], ...],
      "moves_raw": "[[1, 0, 2], [2, 0, 1], ...]",
      "reasoning": "First I need to move disk 1...",
      "raw_response": "Full LLM response...",
      "parse_success": true
    }
  ]
}
```

### Evaluation Results
The evaluation produces:
1. **Console output** - Summary statistics and detailed breakdown
2. **JSON file** (optional) - Machine-readable detailed results
3. **Text file** (optional) - Human-readable report

## Problem Complexity

The Tower of Hanoi problem has exponential complexity:

| Disks | Optimal Moves | Complexity |
|-------|--------------|------------|
| 3     | 7            | Easy       |
| 4     | 15           | Medium     |
| 5     | 31           | Medium     |
| 6     | 63           | Hard       |
| 7     | 127          | Very Hard  |
| 8     | 255          | Extreme    |

## Evaluation Metrics

The evaluation script computes:

- **Parse Success Rate** - How often the LLM output could be parsed
- **Valid Solution Rate** - How often parsed moves were valid
- **Solve Rate** - How often valid moves actually solved the puzzle  
- **Optimal Rate** - How often solutions used the minimum number of moves
- **Efficiency Ratio** - Average (optimal_moves / actual_moves) for solved puzzles

## Example Commands

### Full experiment with batch processing
```bash
# Run complete experiment (3-8 disks, 5 trials each) using Claude
python run_hanoi_experiment.py \
    --llm anthropic \
    --min-disks 3 \
    --max-disks 8 \
    --num-trials 5 \
    --use-batch \
    --output-dir results/claude_hanoi

# Evaluate results
python evaluate_hanoi.py \
    --results-dir results/claude_hanoi \
    --output-file results/claude_evaluation.json \
    --report-file results/claude_report.txt
```

### Quick test with fewer problems
```bash
# Small test (3-4 disks, 2 trials each)
python run_hanoi_experiment.py \
    --llm openai \
    --openai-model gpt-4o-mini \
    --max-disks 4 \
    --num-trials 2 \
    --debug-prompt

# Evaluate
python evaluate_hanoi.py --results-dir hanoi_results
```

### Compare multiple models
```bash
# Run experiments for different models
for model in anthropic openai gemini; do
    python run_hanoi_experiment.py \
        --llm $model \
        --output-dir results/${model}_hanoi \
        --max-disks 6 \
        --num-trials 3
    
    python evaluate_hanoi.py \
        --results-dir results/${model}_hanoi \
        --report-file results/${model}_report.txt
done
```

## Environment Setup

### Required Environment Variables
```bash
# For Anthropic
export ANTHROPIC_API_KEY=your_key_here

# For OpenAI  
export OPENAI_API_KEY=your_key_here

# For Gemini
export GEMINI_API_KEY=your_key_here
```

### Dependencies
The scripts use the shared LLM client library (`../shared/llm_clients.py`) which handles:
- API authentication and requests
- Batch processing (where supported)
- Error handling and retries
- Response parsing

## Troubleshooting

### Common Issues

1. **API Key not set**
   ```
   Error: ANTHROPIC_API_KEY environment variable not set
   ```
   Solution: Export the required API key for your chosen provider

2. **vLLM server not running**
   ```
   No vLLM server running at the specified URL
   ```
   Solution: Start your vLLM server or check the `--vllm-base-url`

3. **Parse failures**
   - Check `--debug-prompt` to see what prompt was sent
   - Examine raw responses in the output JSON files
   - Some models may need different prompting strategies

4. **Batch processing fails**
   - Falls back to individual processing automatically
   - Individual mode is slower but more reliable
   - Gemini doesn't support batch processing

### Performance Tips

- Use `--use-batch` for large experiments (much faster)
- Start with small disk counts (3-5) to test your setup
- Monitor API costs - larger experiments can be expensive
- Use `--debug-prompt` to verify prompt formatting 