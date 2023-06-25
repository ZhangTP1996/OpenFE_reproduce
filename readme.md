# Guide (for linux)

### Environment Setup

- Install anaconda
- ```shell
  export PROJECT_DIR=<ABSOLUTE path to the repository root>
  conda create -n OpenFE python=3.8.12
  conda activate OpenFE
  conda env config vars set PYTHONPATH=${PYTHONPATH}:${PROJECT_DIR}
  conda env config vars set PROJECT_DIR=${PROJECT_DIR}
  conda env config vars set LD_LIBRARY_PATH=${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}
  conda deactivate
  conda activate OpenFE
  python -m pip install -r requirements.txt --no-deps
  ```

### Data Download

- Part 1: Kaggle data

  - Prepare data of IEEE

    - Download link: [IEEE-CIS Fraud Detection | Kaggle](https://www.kaggle.com/competitions/ieee-fraud-detection/data) (There is a `Download All` button)
    - unzip and make sure there exists
      - `./data/IEEE/train_identity.csv`
      - `./data/IEEE/train_transaction.csv`
      - `./data/IEEE/test_identity.csv`
      - `./data/IEEE/test_transaction.csv`
      - `./data/IEEE/sample_submission.csv`

  - Prepare data of BNP

    - Download link: [BNP Paribas Cardif Claims Management | Kaggle](https://www.kaggle.com/competitions/bnp-paribas-cardif-claims-management/data) (There is a `Download All` button)
    - unzip and make sure there exists
      - `./data/BNP/train.csv.zip`
      - `./data/BNP/test.csv.zip`
      - `./data/BNP/sample_submission.csv.zip`

- Part 2: other data
  - Download link: https://www.dropbox.com/s/8tj5ln7wz1r9arc/data.zip?dl=0
  - Unzip and move the files so that there exists
    - `./data/{dataset}/*.npy`

### Experiment

- Part 1: Kaggle experiment **(consistent with Table 5 in our paper)**
  - IEEE Experiment
    - Make sure you are in the folder `run_IEEE`.
    - `bash IEEE.sh`
    - Output is the file `run_IEEE/results/sub_xgb_OpenFE_*_order.csv`.
    - Submit link: [IEEE-CIS Fraud Detection | Kaggle](https://www.kaggle.com/competitions/ieee-fraud-detection/submit)
  - BNP Experiment
    - Make sure you are in the folder `run_BNP`.
    - `bash BNP.sh`
    - Outputs are in the folder `run_BNP/result/`. To evaluate them, submit them to the link below.
    - Submit link: [BNP Paribas Cardif Claims Management | Kaggle](https://www.kaggle.com/competitions/bnp-paribas-cardif-claims-management/submit)
- Part 2: other experiment **(consistent with Table 3 in our paper)**

[//]: # (  - Run all of them)

[//]: # (    - In the folder `runs`.)

[//]: # (    - `python3 run_all.py`)
  - Reproduce results of OpenFE
    - Run a single dataset (e.g. california_housing)
      - `bash shell_inst/california_housing.sh`
    - Results are in the folder `runs/output/{dataset}/lightgbm/tuned`
      - There are two files in the folder.
        - `result` shows the test value under corresponding metric.
        - `stats.json` shows more details
  - Reproduce results of baseline methods
    - We run FCTree on the california_housing dataset as an example. Running other methods on other datasets only require changing the arguments.
    - `python baseline/run_methods.py --method fctree --data california_housing --task regression --n_new_features 10 --n_jobs 8`
    - `python eval.py --data 'california_housing' --model 'lightgbm' --model_type 'tuned' --task_type 'regression' --algorithm 'fctree' --n_saved_features 10`



## Appendix

```
root:[demo]
+--data                           The folder of data.
|      +--BNP
|      ...
+--FeatureGenerator.py            Imported by OpenFE for calculating features.
+--OpenFE.py                      This is a bit different from the open-sourced package.
+--readme.md                      Guide.
+--requirements.txt
+--runs                           This folder is for other experiment.
|      +--bin
|      +--clear.sh                Remove all output files. (Including results.)
|      +--eval.py                 Train models to evaluate new features.
|      +--FE_first_order.py       Generate first order features.
|      +--FE_high_order.py        Generate second order features.
|      +--lib
|      +--nn_utils.py
|      +--run_all.py              Automatically run all experiments.
|      +--shell_inst              Experiment for a specific dataset.
|      |      +--nomao.sh
|      |      ...
|      +--tuned_parameters        This folder contains the tuned parameters.
|      +--tune_parameter.py
|      +--baseline                     This folder contains all the baseline methods we reproduce.
+--run_BNP
|      +--BNP.sh                  Automatically run BNP experiments.
|      +--eval_first_order.py     
|      +--eval_high_order.py
|      +--FE_first_order.py
|      +--FE_high_order.py
|      +--result
+--run_IEEE
|      +--IEEE.sh                 Automatically run IEEE experiments.
|      +--IEEE_utils.py
|      +--main.py
|      +--results
+--utils.py                       Utils imported by OpenFE. 
```

