#!/bin/bash -l
#SBATCH --job-name="swarm_ffn"
#SBATCH --output=swarm_ffn_%j.out
#SBATCH --error=swarm_ffn_err_%j.out
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --ntasks-per-core=1
#SBATCH --partition=normal
#SBATCH --constraint=gpu
#SBATCH --account=s1160
#SBATCH --mail-user=wadaniel@ethz.ch
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL


module purge
module load daint-gpu gcc GSL cray-hdf5-parallel cray-python cdt-cuda craype-accel-nvidia60 VTK Boost/1.78.0-CrayGNU-21.09-python3

cd ..
#python3 run-tf-ffn.py --file _trajectories/observations_0o_25N_7NN_1000NT_50num_3d.json --NN 7 --obj 0 --epochs 500 --learningRate 0.00005
#python3 run-tf-ffn.py --file _trajectories/observations_1o_25N_7NN_1000NT_50num_3d.json --NN 7 --obj 1 --epochs 500 --learningRate 0.00005
python3 run-tf-ffn.py --file _trajectories/observations_2o_25N_7NN_1000NT_50num_3d.json --NN 7 --obj 2 --epochs 500 --learningRate 0.00005
