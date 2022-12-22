JOBID=10
for N in 40
do 
    for NN in 5 9; 
    do
	for NT in 1000;
	do
	   for NL in 128 256;
	   do
	      for reward in "local" "global";
	      do
	         for exp in 2000000;
		 do
	             for rep in 1 2;
		     do

		      echo $JOBID
		      export N=$N
		      export NN=$NN
		      export NT=$NT
		      export NL=$NL
		      export REWARD="$reward" 
		      export EXP=$exp
		      export JID=$JOBID
			  
		      export BASE="${SCRATCH}/RLSwimmers/JID_withangletovel_${JID}/"
		      export DIR="${BASE}_result_vracer_${JID}/"

		      mkdir ${DIR} -p

		      export configfile=${DIR}run_${JID}.config
		      touch "${configfile}"
		      
		      cp ./bsub-vracer.lsf ${BASE}
		      cp ./settings.sh ${BASE}
		      cp ../run-vracer.py ${BASE}
		      cp ../eval-vracer.py ${BASE}
		      cp ../profile.py ${BASE}
		      cp -r ../_model ${BASE}

		      pushd .
		      cd ${BASE}
		      bsub < ./bsub-vracer.lsf
		      popd
		      JOBID=$(($JOBID+1))
		      
		      done;
		 done
	      done;
	   done;
	done;
    done;	
done
