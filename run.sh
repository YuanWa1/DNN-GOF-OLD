#!/bin/bash
#SBATCH -J test
#SBATCH -p shared
#SBATCH -N 1 -c 4
#SBATCH --time=02:00:00
#SBATCH --mem=1024M

#SBATCH --array=1-1000

module load python39
module load cuda11.0/toolkit/11.0.3
source myenv/bin/activate
python3 test.py $1

