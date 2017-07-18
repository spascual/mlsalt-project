id_list='4' #' 4 5'
number_pts='750' # 1000 1250'
for item in $id_list; do 
	for N_train in $number_pts; do
		echo 'sample_'$item', N_train='$N_train
		python thesis_work/regression_exp.py $N_train $item &
	done
done