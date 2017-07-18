id_list='1 2 3 4 5'
number_pts='500 750 1000 1250'
for item in $id_list; do 
	for N_train in $number_pts; do
		echo 'sample_'$item', N_train='$N_train
		python thesis_work/regression_exp.py $N_train $item &
		# python thesis_work/fixed_zu_reg.py $N_train $item &
		# python thesis_work/default_reg.py $N_train $item &
		# python thesis_work/baseline_reg.py $N_train $item &
	done
done