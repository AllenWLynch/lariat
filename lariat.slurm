#!/bin/bash
#SBATCH --job-name=lariat
#SBATCH --output=/n/data1/hms/dbmi/park/charlotte/lariat/logs/lariat%j.out    # Use this to monitor your job's progress 
#SBATCH --error=/n/data1/hms/dbmi/park/charlotte/lariat/logs/lariat%j.err
#SBATCH --time=2:00:00        # 2 hours
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=200M # 200 MB
#SBATCH --partition=park 

# ****************************************************************
# Remember to activate conda lariat environment before running
# ****************************************************************

lariat-compose --cluster --partition park /n/data1/hms/dbmi/park/charlotte/lariat/example_config.yaml