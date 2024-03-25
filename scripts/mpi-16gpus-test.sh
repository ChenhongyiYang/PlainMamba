#!/bin/bash
##----------------------- Start job description -----------------------
#SBATCH --nodes=4                # Select the number of nodes
#SBATCH --ntasks-per-node=4      # Number of MPI slots per node, MPI processes per node SLURM_NTASKS_PER_NODE
#SBATCH --cpus-per-task=1        # The num of CPUs per each MPI process
#SBATCH --mem=400GB              # total memory per node (4 GB per cpu-core is default)
#SBATCH --gres=gpu:4             # number of gpus per node
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x.%j.out
#SBATCH --partition=orchid
#SBATCH --account=orchid
#SBATCH --job-name=mpi-test
##------------------------ End job description ------------------------

module purge
module load eb/OpenMPI/gcc/4.0.0
source /gws/nopw/j04/sensecdt/users/mespi/virtualenvs/v_mamba/bin/activate

echo "GPUS_ON_NODE="$SLURM_GPUS_ON_NODE
echo "SLURM_NNODES="$SLURM_NNODES

export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_GPUS_ON_NODE))
echo "WORLD_SIZE="$WORLD_SIZE

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

# Loop through all environment variables and print those starting with "SLURM_"
for var in $(env | grep '^SLURM_' | awk -F= '{print $1}'); do
    echo "$var=${!var}"
done

mpirun \
    --allow-run-as-root \
    --np $WORLD_SIZE \
    --npernode $SLURM_GPUS_ON_NODE \
    python tools/train.py work_configs/vmamba_test.py \
            --launcher mpi \
            --cfg-options world_size=$WORLD_SIZE