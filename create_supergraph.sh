#SBATCH --job-name=my_job
#SBATCH --output=my_job_output_%j.txt
#SBATCH --partition=tue.default.q
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2G

# Load modules or software if needed optimized for GPU use if available
# In the example PyTorch is made available for import in to my_sript.py


# Execute the script or command
python experiments.create_save_NYC.py