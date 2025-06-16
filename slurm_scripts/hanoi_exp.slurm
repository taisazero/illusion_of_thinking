#!/bin/bash
#SBATCH --job-name=hanoi_exp
#SBATCH --partition=Orion
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=12
#SBATCH --time=05:00:00
#SBATCH --mem=32GB
#SBATCH --output=hanoi-%j.out
#SBATCH --error=hanoi-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ealhossa@charlotte.edu

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Partition: $SLURM_JOB_PARTITION"
echo "Nodes: $SLURM_JOB_NUM_NODES"
echo "Tasks per node: $SLURM_NTASKS_PER_NODE"
echo "Start time: $(date)"
echo "Working directory: $PWD"
echo "========================================="

# Change to the project directory (adjust if needed)
cd $SLURM_SUBMIT_DIR


mkdir -p job_logs

# Activate virtual environment if you have one (uncomment and modify as needed)
source activate socratic_env

python3 illusion_thinking_exp/run_hanoi_experiment.py --llm anthropic --min-disks 9 --max-disks 12 --num-trials 1  --output-dir results/claude_hanoi

echo "========================================="
echo "Job completed at: $(date)"
echo "Total runtime: $SECONDS seconds" 

# move the logs to the job_logs directory
mv hanoi-*.out job_logs/
mv hanoi-*.err job_logs/