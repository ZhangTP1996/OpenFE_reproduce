python3 FE_first_order.py --data 'telecom' --fold 1 --task_type 'classification' --n_saved_features 10
python3 eval.py --data 'telecom' --model 'lightgbm' --model_type 'tuned' --task_type 'classification' --algorithm 'OpenFE' --n_saved_features 10
