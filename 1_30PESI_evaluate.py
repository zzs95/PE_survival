import os
import hdf5storage as hds
from file_and_folder_operations import *
import numpy as np
from sklearn.metrics import accuracy_score
from sksurv.metrics import concordance_index_censored
import copy
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
    
exp_path = './exps'
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
        label_event = np.array(label_mat_data['event_test'])[:,0]
        label_time = np.array(label_mat_data['time_test'])[:,0]
        risk_img = np.array(deep_mat_data['scores_img'])[:,0]
        risk_clin = np.array(deep_mat_data['scores_clin'])[:,0]
        risk_fuse = np.array(deep_mat_data['scores_fuse'])
        risk_fuse = sigmoid(risk_fuse / risk_fuse.max() )
        risk_fusePESI = np.array(deep_mat_data['scores_fusePESI'])
        risk_fusePESI = sigmoid(risk_fusePESI / risk_fusePESI.max())
        
        label_time_30day = copy.deepcopy(label_time)
        label_event_30day = copy.deepcopy(label_event)
        time_gt30 = label_time_30day > 30
        label_time_30day[time_gt30] = 30
        label_event_30day[time_gt30] = 0

        cidx = concordance_index_censored(label_event.astype(bool), label_time, PESI_list)[0]
        cidx30 = concordance_index_censored(label_event_30day.astype(bool), label_time_30day, PESI_list)[0]
        print('PESI,', cidx, ',', cidx30)
        
        cidx = concordance_index_censored(label_event.astype(bool), label_time, risk_img)[0]
        cidx30 = concordance_index_censored(label_event_30day.astype(bool), label_time_30day, risk_img)[0]
        print('img', cidx, cidx30)
        cidx = concordance_index_censored(label_event.astype(bool), label_time, risk_clin)[0]
        cidx30 = concordance_index_censored(label_event_30day.astype(bool), label_time_30day, risk_clin)[0]
        print('clin', cidx, cidx30)
        cidx = concordance_index_censored(label_event.astype(bool), label_time, risk_fuse)[0]
        cidx30 = concordance_index_censored(label_event_30day.astype(bool), label_time_30day, risk_fuse)[0]
        print('fuse', cidx, cidx30)
        cidx = concordance_index_censored(label_event.astype(bool), label_time, risk_fusePESI)[0]
        cidx30 = concordance_index_censored(label_event_30day.astype(bool), label_time_30day, risk_fusePESI)[0]
        print('fusePESI', cidx, cidx30)
        
        