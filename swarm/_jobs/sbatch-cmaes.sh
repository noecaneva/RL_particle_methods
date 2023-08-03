#!/bin/bash -l
#SBATCH --job-name="swarm_cmaes"
#SBATCH --output=swarm_cmaes_${RUN}_%j.out
#SBATCH --error=swarm_cmaes_err_${RUN}_%j.out
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

DIM=3
N=100
OBJ=1       #0: milling, 1: schooling, 2: swarming
RUN=0

DIR="./_result_cmaes_${OBJ}_${RUN}/"
cd ..

mkdir $DIR
touch "${DIR}/run.config"

echo DIM ${DIM} >> "${DIR}/run.config"
echo N ${N}     >> "${DIR}/run.config"
echo OBJ ${OBJ} >> "${DIR}/run.config"
echo RUN ${RUB} >> "${DIR}/run.config"


cp run-cames.py $DIR/run-cmaes.py

mpirun -n 9 python3 run-cmaes.py --dim ${DIM} --obj ${OBJ} --N $N --run ${RUN} 

python3 -m korali.plot --dir $DIR --out "cmaes_${RUN}.png"

code=$?
exit $code
