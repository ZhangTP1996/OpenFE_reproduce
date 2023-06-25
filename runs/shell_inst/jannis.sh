python3 FE_first_order.py --data 'jannis' --fold 1 --task_type 'classification' --n_saved_features 50
python3 eval.py --data 'jannis' --model 'lightgbm' --model_type 'tuned' --task_type 'classification' --n_saved_features 50  --algorithm 'OpenFE'
