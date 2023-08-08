#!/bin/bash -l
#SBATCH --job-name="swarm_ffn"
#SBATCH --output=swarm_ffn_%j.out
#SBATCH --error=swarm_ffn_err_%j.out
#SBATCH --time=4:00:00
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
export HDF5_USE_FILE_LOCKING='FALSE'

cd ..

#python run-tf-ffn.py --file _trajectories/observations_0o_25N_7NN_1000NT_50num_3d.json --obj 0 --NN 7 --learningRate 0.0001 --steps 2000 --train
#python run-tf-ffn.py --file _trajectories/observations_1o_25N_7NN_1000NT_50num_3d.json --obj 1 --NN 7 --learningRate 0.0001 --steps 2000 --train
#python run-tf-ffn.py --file _trajectories/observations_2o_25N_7NN_1000NT_50num_3d.json --obj 2 --NN 7 --learningRate 0.0001 --steps 2000 --train

python plotFFNTrajectory.py --modelfile ./training_ffn_o0/final_o0.h5 --seed 1336 --obj 0
python plotFFNTrajectory.py --modelfile ./training_ffn_o1/final_o1.h5 --seed 1336 --obj 1
python plotFFNTrajectory.py --modelfile ./training_ffn_o2/final_o2.h5 --seed 1336 --obj 2

python plotFFNTrajectory.py --modelfile ./training_ffn_o0/final_o0.h5 --seed 1338 --obj 0
python plotFFNTrajectory.py --modelfile ./training_ffn_o1/final_o1.h5 --seed 1338 --obj 1
python plotFFNTrajectory.py --modelfile ./training_ffn_o2/final_o2.h5 --seed 1338 --obj 2

python plotFFNTrajectory.py --modelfile ./training_ffn_o0/final_o0.h5 --seed 1339 --obj 0
python plotFFNTrajectory.py --modelfile ./training_ffn_o1/final_o1.h5 --seed 1339 --obj 1
python plotFFNTrajectory.py --modelfile ./training_ffn_o2/final_o2.h5 --seed 1339 --obj 2

#python plotFFNReward.py --modelfile ./training_ffn_o0/final_o0.h5 --seed 1337 --reps 25 --obj 0
#python plotFFNReward.py --modelfile ./training_ffn_o1/final_o1.h5 --seed 1337 --reps 25 --obj 1
#python plotFFNReward.py --modelfile ./training_ffn_o2/final_o2.h5 --seed 1337 --reps 25 --obj 2
