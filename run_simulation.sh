#!/bin/bash
#SBATCH --job-name=t1d-sim
#SBATCH --output=logs/sim_%j.log
#SBATCH --error=logs/sim_%j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --partition=general         # replace with the correct partition name (run: sinfo)

mkdir -p logs

source /home/furla/desktop/dtu-mt-patient-generator/.venv/bin/activate

cd /home/furla/desktop/dtu-mt-patient-generator
python library_generator.py
