run=400

export POL="Linear"
#export POL="Quadratic"
export EXP=2000000

export N=25
export NN=6

for EU in 1000 5000 15000
do
    for R in 8 32
    do

    for D in 1 4 8
    do 
        for B in 4 16 64
        do 
            run=$(($run+1))
            export RUN=$run
            export RNN=$R
            export DBS=$D
            export BBS=$B
            export EBRU=$EU
            bsub < bsub-vracer-irl.lsf
        done
    done

    done
done