run=1

export POL="Linear"
#export POL="Quadratic"
export EXP=3000000

export DIM=3
#export DIM=2
export N=25
#export NN=9
export NN=7
#export NT=500
export NT=1000
export DAT=50
export OBJ=1

for EU in 8000 #12000 16000
#for EU in 8000 16000
do
    for R in 32 #64 128
    do
        for D in 4 #16 64
        do 
            for B in 4 #16 64
            do 
                export RUN=$run
                export RNN=$R
                export DBS=$D
                export BBS=$B
                export EBRU=$EU
                echo $run
                #bsub < bsub-vracer-irl.lsf
                bash sbatch-irl.sh
                sleep 0.1
                run=$(($run+1))
            done
        done

    done
done
