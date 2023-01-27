run=200

N=10
NN=6
L=500

cd ..

for S in 10 30 70 120 130 170 190 220 230 240 250 260 270 280 290 300
do
    python ./main.py --numIndividuals $N --numNearestNeighbours $NN --seed $S --record
done
