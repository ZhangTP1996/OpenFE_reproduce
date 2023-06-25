python3 FE_first_order.py --data 'california_housing' --fold 1 --task_type 'regression' --n_saved_features 10 --remain_for_stage2 100
python3 eval.py --data 'california_housing' --model 'lightgbm' --model_type 'tuned' --task_type 'regression' --algorithm 'OpenFE' --n_saved_features 10
