python3 FE_first_order.py --data 'vehicleNorm' --fold 16 --task_type 'classification' --n_saved_features 10
python3 eval.py --data 'vehicleNorm' --model 'lightgbm' --model_type 'tuned' --task_type 'classification' --algorithm 'OpenFE' --n_saved_features 10
