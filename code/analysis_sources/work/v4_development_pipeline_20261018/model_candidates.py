from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from itertools import product
import numpy as np

def model_candidates(seed=42, threads=1, gpu=True):
    candidates = {}
    
    # LR
    lr_params = {
        'solver': ['liblinear'],
        'penalty': ['l2'],
        'C': [.01, .1, 1, 10],
        'max_iter': [5000],
        'random_state': [seed]
    }
    lr_keys = ['LR_C_0.01', 'LR_C_0.1', 'LR_C_1.0', 'LR_C_10.0']
    lr_models = [LogisticRegression(**dict(zip(lr_params.keys(), vals))) 
                 for vals in product(*lr_params.values())]
    candidates['LR'] = list(zip(lr_keys, lr_models))
    
    # DT
    dt_params = {
        'max_depth': [3, 5, None],
        'min_samples_leaf': [5, 20],
        'random_state': [seed]
    }
    dt_keys = ['DT_md3_ml5', 'DT_md3_ml20', 'DT_md5_ml5', 'DT_md5_ml20', 'DT_mdNone_ml5', 'DT_mdNone_ml20']
    dt_models = [DecisionTreeClassifier(**dict(zip(dt_params.keys(), vals))) 
                 for vals in product(*dt_params.values())]
    candidates['DT'] = list(zip(dt_keys, dt_models))
    
    # RF
    rf_params = {
        'n_estimators': [500],
        'max_depth': [6, None],
        'min_samples_leaf': [1, 5],
        'max_features': ['sqrt'],
        'n_jobs': [threads],
        'random_state': [seed]
    }
    rf_keys = ['RF_md6_ml1', 'RF_md6_ml5', 'RF_mdNone_ml1', 'RF_mdNone_ml5']
    rf_models = [RandomForestClassifier(**dict(zip(rf_params.keys(), vals))) 
                 for vals in product(*rf_params.values())]
    candidates['RF'] = list(zip(rf_keys, rf_models))
    
    # KNN
    knn_params = {
        'n_neighbors': [5, 15, 31],
        'weights': ['uniform', 'distance'],
        'n_jobs': [threads]
    }
    knn_keys = ['KNN_k5_wuniform', 'KNN_k5_wdistance', 'KNN_k15_wuniform', 'KNN_k15_wdistance', 'KNN_k31_wuniform', 'KNN_k31_wdistance']
    knn_models = [KNeighborsClassifier(**dict(zip(knn_params.keys(), vals))) 
                  for vals in product(*knn_params.values())]
    candidates['KNN'] = list(zip(knn_keys, knn_models))
    
    # SVM
    svm_params = {
        'kernel': ['rbf'],
        'C': [.1, 1, 10],
        'gamma': ['scale', .01],
        'probability': [True],
        'random_state': [seed]
    }
    svm_keys = ['SVM_C0.1_gscale', 'SVM_C0.1_g0.01', 'SVM_C1.0_gscale', 'SVM_C1.0_g0.01', 'SVM_C10.0_gscale', 'SVM_C10.0_g0.01']
    svm_models = [SVC(**dict(zip(svm_params.keys(), vals))) 
                  for vals in product(*svm_params.values())]
    candidates['SVM'] = list(zip(svm_keys, svm_models))
    
    # NB
    nb_params = {
        'var_smoothing': [1e-9, 1e-7, 1e-5]
    }
    nb_keys = ['NB_vs1e-09', 'NB_vs1e-07', 'NB_vs1e-05']
    nb_models = [GaussianNB(**dict(zip(nb_params.keys(), vals))) 
                 for vals in product(*nb_params.values())]
    candidates['NB'] = list(zip(nb_keys, nb_models))
    
    # XGB
    xgb_params = {
        'n_estimators': [500],
        'max_depth': [2, 4],
        'learning_rate': [.03, .1],
        'min_child_weight': [3],
        'subsample': [.9],
        'colsample_bytree': [.8],
        'eval_metric': ['logloss'],
        'tree_method': ['hist'],
        'device': ['cuda' if gpu else 'cpu'],
        'n_jobs': [threads],
        'random_state': [seed]
    }
    xgb_keys = ['XGB_md2_lr0.03', 'XGB_md2_lr0.1', 'XGB_md4_lr0.03', 'XGB_md4_lr0.1']
    xgb_models = [XGBClassifier(**dict(zip(xgb_params.keys(), vals))) 
                  for vals in product(*xgb_params.values())]
    candidates['XGB'] = list(zip(xgb_keys, xgb_models))
    
    # SGBT
    sgbt_params = {
        'n_estimators': [500],
        'max_depth': [1, 2],
        'learning_rate': [.03, .1],
        'subsample': [.8],
        'random_state': [seed]
    }
    sgbt_keys = ['SGBT_md1_lr0.03', 'SGBT_md1_lr0.1', 'SGBT_md2_lr0.03', 'SGBT_md2_lr0.1']
    sgbt_models = [GradientBoostingClassifier(**dict(zip(sgbt_params.keys(), vals))) 
                   for vals in product(*sgbt_params.values())]
    candidates['SGBT'] = list(zip(sgbt_keys, sgbt_models))
    
    # NNET
    nnet_params = {
        'hidden_layer_sizes': [(25,), (50,)],
        'alpha': [.001, .1],
        'solver': ['adam'],
        'activation': ['relu'],
        'max_iter': [1000],
        'early_stopping': [False],
        'random_state': [seed]
    }
    nnet_keys = ['NNET_h25_a0.001', 'NNET_h25_a0.1', 'NNET_h50_a0.001', 'NNET_h50_a0.1']
    nnet_models = [MLPClassifier(**dict(zip(nnet_params.keys(), vals))) 
                   for vals in product(*nnet_params.values())]
    candidates['NNET'] = list(zip(nnet_keys, nnet_models))
    
    return candidates
