python3 FE_first_order.py --data 'microsoft' --fold 64 --task_type 'regression' --n_saved_features 10
python3 eval.py --data 'microsoft' --model 'lightgbm' --model_type 'tuned' --task_type 'regression' --n_saved_features 10  --algorithm 'OpenFE'
