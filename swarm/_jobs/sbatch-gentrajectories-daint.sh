#!/bin/bash -l
#SBATCH --job-name="swarm_gent"
#SBATCH --output=./output/swarm_gent_%j.out
#SBATCH --error=./output/swarm_gent_err_%j.out
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
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
python generateTrajectories.py --N 25 --NT 1000 --NN 7 --D 2 --num 50 --obj 2
python generateTrajectories.py --N 25 --NT 1000 --NN 7 --D 3 --num 50 --obj 2

code=$?
exit $code
