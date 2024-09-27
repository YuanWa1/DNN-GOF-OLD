for var in 200 500 1000
do
	echo "current nN is: $var"
	sbatch run.sh $var
done
