#!/bin/bash -l

source settings_irl.sh

cat > run.sh <<EOF
#!/bin/bash -l
#SBATCH --job-name="swarm_irl"
#SBATCH --output=swarm_irl_${RUN}_%j.out
#SBATCH --error=swarm_irl_err_${RUN}_%j.out
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

export OMP_NUM_THREADS=12

cd ..
BASE="${SCRATCH}/swarm_marl_${DIM}d/${RUN}"
DIR="\${BASE}/_result_vracer_irl_${RUN}/"

mkdir \${DIR} -p
touch "\${DIR}/run.config"
echo DIM    ${DIM} >> "\${DIR}/run.config"
echo EBRU   ${EBRU} >> "\${DIR}/run.config"
echo DBS    ${DBS} >> "\${DIR}/run.config"
echo BBS    ${BBS} >> "\${DIR}/run.config"
echo BSS    ${BSS} >> "\${DIR}/run.config"
echo EXP    ${EXP} >> "\${DIR}/run.config"
echo RNN    ${RNN} >> "\${DIR}/run.config"
echo POL    ${POL} >> "\${DIR}/run.config"
echo RUN    ${RUN} >> "\${DIR}/run.config"

cp -r _model \${BASE}
cp -r run-vracer-irl.py \${BASE}
cp -r _trajectories \${BASE}
cp -r _jobs/settings_irl.sh \${BASE}

pushd .
cd \${BASE}

python3 run-vracer-irl.py \
    --dim ${DIM} --ebru ${EBRU} --dbs ${DBS} --bbs ${BBS} \
    --bss ${BSS} --exp ${EXP} --rnn ${RNN} --pol ${POL} --run ${RUN} \
    --N ${N} --NN ${NN} --NT ${NT}

popd

python3 -m korali.plot --dir \$DIR --out "swarm_${RUN}.png"
python3 -m korali.rlview --dir \$DIR --out "irl-swarm_${RUN}.png" --showObservations --showCI 0.8 --minReward 0. --maxReward 1.0
python3 -m korali.rlview --dir \$DIR --out "firl-swarm_${RUN}.png" --featureReward --showObservations --showCI 0.8

mkdir _irl_figures_march -p
mv swarm*.png _irl_figures_march/
mv irl*.png _irl_figures_march/
mv firl*.png  _irl_figures_march/

sstat --format=AveCPU,AvePages,AveRSS,AveVMSize,JobID -j %j --allsteps
date
code=$?
exit $code
EOF

sbatch run.sh

