python3 FE_first_order.py --data 'medical' --fold 1 --task_type 'regression' --n_saved_features 10
python3 eval.py --data 'medical' --model 'lightgbm' --model_type 'tuned' --task_type 'regression' --algorithm 'OpenFE' --n_saved_features 10
