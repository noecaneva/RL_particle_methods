run=0

export POL="Linear"
#export POL="Quadratic"
export EXP=3000000

export DIM=3
export N=25
export NN=9
#export DIM=2
#export N=25
#export NN=9
export NT=1000

for EU in 3000 5000 10000
do
    #for R in 8 32
    for R in 128
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
                echo $run
                #bsub < bsub-vracer-irl.lsf
                bash sbatch-irl.sh
                sleep 0.1
                exit
            done
        done

    done
done
