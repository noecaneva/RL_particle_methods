run=501

export POL="Linear"
#export POL="Quadratic"
export EXP=2000000

export DIM=3
export N=10
export NN=3
#export DIM=2
#export N=25
#export NN=9
export NT=1000

for EU in 5000 #15000
do
    for R in 8 #32
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
