#!/bin/bash
#SBATCH -n 16
#SBATCH --time=24:00
#SBATCH --job-name=cmaes
#SBATCH --output=./output/cmaes-%j.out 
#SBATCH --error=./output/cmaes-%j.err
#SBATCH --mail-user=wadaniel@ethz.ch
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

source /cluster/apps/local/env2lmod.sh
module purge
module load gcc/8.2.0 python/3.9.9 openmpi/4.0.2 gsl/2.6
module load git-lfs

DIM=3
N=25
OBJ=2       #0: milling, 1: schooling, 2: swarming
RUN=1

DIR="./_result_cmaes_${OBJ}_${DIM}_${RUN}/"
cd ..

mkdir -p ${DIR}
touch "${DIR}/run.config"

echo DIM ${DIM} >> "${DIR}/run.config"
echo N ${N}     >> "${DIR}/run.config"
echo OBJ ${OBJ} >> "${DIR}/run.config"
echo RUN ${RUN} >> "${DIR}/run.config"

cp run-cames.py $DIR/run-cmaes.py

python3 run-cmaes.py --dim ${DIM} --obj ${OBJ} --N $N --run ${RUN} 

python3 -m korali.plot --dir ${DIR} --out "cmaes_${RUN}.png"

code=$?
exit $code
