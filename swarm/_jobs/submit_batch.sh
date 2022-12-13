JOBID=100
for N in 10 20
do 
    for NN in 3 5 9; 
    do
	for NT in 1000;
	do
	   for NL in 64 128 256;
	   do
	      for reward in "local" "global";
	      do
	         for exp in 10000000;
		 do
		   echo $JOBID
		   export N=$N
        	   export NN=$NN
		   export NT=$NT
		   export NL=$NL
		   export REWARD="$reward" 
        	   export exp=$reward
		   export JID=$JOBID 
        	   bsub < ./bsub-vracer.lsf
		   JOBID=$(($JOBID+1))
	         done;
	      done;
	   done;
	done;
    done;	
done
