import numpy as np
import pandas as pd
import os
import json
import scipy.special


def load_prediction(path, task):
    train_pred = np.load(os.path.join(path, 'p_train.npy'))
    val_pred = np.load(os.path.join(path, 'p_val.npy'))
    if 'class' in task:
        train_pred = scipy.special.expit(train_pred)
        val_pred = scipy.special.expit(val_pred)
    train_pred = pd.DataFrame(train_pred)
    val_pred = pd.DataFrame(val_pred)
    return train_pred, val_pred

def load_data(path):
    train_x = np.load(os.path.join(path, 'N_train.npy'))
    val_x = np.load(os.path.join(path, 'N_val.npy'))
    test_x = np.load(os.path.join(path, 'N_test.npy'))
    train_y = np.load(os.path.join(path, 'y_train.npy'))
    val_y = np.load(os.path.join(path, 'y_val.npy'))
    test_y = np.load(os.path.join(path, 'y_test.npy'))
    train_x = pd.DataFrame(train_x, columns=['f%d' % i for i in range(train_x.shape[1])])
    val_x = pd.DataFrame(val_x, columns=['f%d' % i for i in range(train_x.shape[1])])
    test_x = pd.DataFrame(test_x, columns=['f%d' % i for i in range(train_x.shape[1])])
    train_y = pd.DataFrame(train_y, columns=['label'])
    val_y = pd.DataFrame(val_y, columns=['label'])
    test_y = pd.DataFrame(test_y, columns=['label'])

    if 'C_train.npy' in os.listdir(path):
        train_c = np.load(os.path.join(path, 'C_train.npy'), allow_pickle=True)
        val_c = np.load(os.path.join(path, 'C_val.npy'), allow_pickle=True)
        test_c = np.load(os.path.join(path, 'C_test.npy'), allow_pickle=True)
        train_c = pd.DataFrame(train_c, columns=['c%d' % i for i in range(train_c.shape[1])])
        val_c = pd.DataFrame(val_c, columns=['c%d' % i for i in range(train_c.shape[1])])
        test_c = pd.DataFrame(test_c, columns=['c%d' % i for i in range(train_c.shape[1])])
    else:
        train_c, val_c, test_c = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    train_x = train_x.replace([np.inf, -np.inf], np.nan)
    val_x = val_x.replace([np.inf, -np.inf], np.nan)
    test_x = test_x.replace([np.inf, -np.inf], np.nan)

    return train_x, val_x, test_x, train_c, val_c, test_c, train_y, val_y, test_y


def save_to_rtdl_format(file_name, file_path, task_type, train_x, val_x, test_x,
                        train_c, val_c, test_c, train_y, val_y, test_y):
    os.makedirs(file_path, exist_ok=True)
    np.save(os.path.join(file_path, 'N_train.npy'), train_x.values.astype(np.float32))
    np.save(os.path.join(file_path, 'N_val.npy'), val_x.values.astype(np.float32))
    np.save(os.path.join(file_path, 'N_test.npy'), test_x.values.astype(np.float32))
    if len(train_c.columns) > 0:
        np.save(os.path.join(file_path, 'C_train.npy'), train_c.values.astype(str))
        np.save(os.path.join(file_path, 'C_val.npy'), val_c.values.astype(str))
        np.save(os.path.join(file_path, 'C_test.npy'), test_c.values.astype(str))
    if 'class' in task_type:
        save_format = np.int64
    else:
        save_format = np.float32
    np.save(os.path.join(file_path, 'y_train.npy'), train_y.values.ravel().astype(save_format))
    np.save(os.path.join(file_path, 'y_val.npy'), val_y.values.ravel().astype(save_format))
    np.save(os.path.join(file_path, 'y_test.npy'), test_y.values.ravel().astype(save_format))

    info = {
        "name": "%s___0" % file_name,
        "basename": "%s" % file_name,
        "split": 0,
        "task_type": task_type,
        "n_num_features": len(train_x.columns),
        "n_cat_features": 0,
        "train_size": len(train_x),
        "val_size": len(val_x),
        "test_size": len(test_x),
        "n_classes": len(train_y[train_y.columns[0]].unique())
    }
    json_str = json.dumps(info, indent=4)
    with open(os.path.join(file_path, 'info.json'), 'w') as json_file:
        json_file.write(json_str)