run=10000

export POL="Linear"
#export POL="Quadratic"
export EXP=5000000

for EU in 2000 5000
do
    for D in 1 2 4
    do 
        for B in 1 4 16 64
        do 
            run=$(($run+1))
            export RUN=$run
            export DBS=$D
            export BBS=$B
            export EBRU=$EU
            bsub < bsub-vracer-irl.lsf
        done
    done
done
