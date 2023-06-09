import os
import hdf5storage as hds
from file_and_folder_operations import *
import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def calculate_nri(y_true, y_pred_base, y_pred_new, threshold):
    y_base = (y_pred_base >= threshold).astype(int)
    y_new = (y_pred_new >= threshold).astype(int)

    tp_base = ((y_true == 1) & (y_base == 1)).sum()
    tn_base = ((y_true == 0) & (y_base == 0)).sum()
    fp_base = ((y_true == 0) & (y_base == 1)).sum()
    fn_base = ((y_true == 1) & (y_base == 0)).sum()
    
    tp_new = ((y_true == 1) & (y_new == 1)).sum()
    tn_new = ((y_true == 0) & (y_new == 0)).sum()
    fp_new = ((y_true == 0) & (y_new == 1)).sum()
    fn_new = ((y_true == 1) & (y_new == 0)).sum()

    nri = (tp_new / (tp_new + fp_new) - tp_base / (tp_base + fp_base)
           + tn_new / (tn_new + fn_new) - tn_base / (tn_base + fn_base))
    
    return nri


exp_root = './exps'
exp_path = exp_root
mat_path = os.path.join(exp_path, 'mat_out')
mat_path_folders = subfolders(mat_path, join=False)
for mat_folder in mat_path_folders:
    if mat_folder in ['NPH', 'RIHtr', 'RIHts',  'TMH', 'TMH_NPH']:
        print(mat_folder)
    # if mat_folder in ['RIHts', ]:
        set_mat_folder = join(mat_path, mat_folder)
        deep_mat = join(set_mat_folder, 'Deep_results_for_risk_groups.mat')
        rsf_mat = join(set_mat_folder, 'RSF_results_for_risk_groups.mat')
        label_mat = join(set_mat_folder, 'labels_all.mat')
        deep_mat_data = hds.loadmat(deep_mat)
        rsf_mat_data = hds.loadmat(rsf_mat)
        label_mat_data = hds.loadmat(label_mat)
        cdx_PESI = deep_mat_data['scores_PESI']
        PESI_list = deep_mat_data['PESI']
        PESI_list = deep_mat_data['PESI']
        label_event = np.array(label_mat_data['event_test'])[:,0]
        risk_img = np.array(deep_mat_data['scores_img'])[:,0]
        risk_clin = np.array(deep_mat_data['scores_clin'])[:,0]
        risk_fuse = np.array(deep_mat_data['scores_fuse'])
        risk_fuse = sigmoid(risk_fuse / risk_fuse.max() )
        risk_fusePESI = np.array(deep_mat_data['scores_fusePESI'])
        risk_fusePESI = sigmoid(risk_fusePESI / risk_fusePESI.max())
        
        print(calculate_nri(label_event, risk_img, risk_fuse, threshold=0.7))
        print(calculate_nri(label_event, risk_clin, risk_fuse, threshold=0.7))
        print(calculate_nri(label_event, risk_fuse, risk_fusePESI, threshold=0.66))
        

