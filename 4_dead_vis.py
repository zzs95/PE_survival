import os
import hdf5storage as hds
from file_and_folder_operations import *
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

label_path = './PE_data/pe_label.xls'
labels_df_s = pd.read_excel(label_path)

exp_path = './exps'
mat_path = os.path.join(exp_path, 'mat_out')
mat_path_folders = subfolders(mat_path, join=False)
for mat_folder in mat_path_folders:
    if mat_folder in ['TMH_NPH', ]:
        set_mat_folder = join(mat_path, mat_folder)
        deep_mat = join(set_mat_folder, 'Deep_results_for_risk_groups.mat')
        rsf_mat = join(set_mat_folder, 'RSF_results_for_risk_groups.mat')
        label_mat = join(set_mat_folder, 'labels_all.mat')
        deep_mat_data = hds.loadmat(deep_mat)
        rsf_mat_data = hds.loadmat(rsf_mat)
        label_mat_data = hds.loadmat(label_mat)
        cidx_PESI = deep_mat_data['scores_PESI']
        PESI_list = deep_mat_data['PESI']
        label_event = np.array(label_mat_data['event_test'])[:,0]
        label_time = np.array(label_mat_data['time_test'])[:,0]
        risk_img = np.array(deep_mat_data['scores_img'])[:,0]
        risk_clin = np.array(deep_mat_data['scores_clin'])[:,0]
        risk_fuse = np.array(deep_mat_data['scores_fuse'])
        risk_fuse = sigmoid(risk_fuse / risk_fuse.max() )
        risk_fusePESI = np.array(deep_mat_data['scores_fusePESI'])
        risk_fusePESI = sigmoid(risk_fusePESI / risk_fusePESI.max())

        for mod in ['fuse', 'fusePESI']:
            if mod == 'img':
                risk = risk_img
            elif mod == 'clin':
                risk = risk_clin
            elif mod == 'fuse':
                risk = risk_fuse
            elif mod == 'fusePESI':
                risk = risk_fusePESI
            risk_median = np.median(risk)

            risk_order = np.argsort(risk)
            risk_x = np.arange(len(risk))
            risk_sorted = risk[risk_order]

            dead_idx = np.array(label_event)
            dead_idx_sorted = dead_idx[risk_order]
            dead_order_sorted = np.where(dead_idx_sorted)
            dead_risk_sorted = risk_sorted[dead_order_sorted]
            dead_x_sorted = risk_x[dead_order_sorted]
            print('all num:', len(risk), 'dead num:', np.sum(dead_idx), 'high risk dead:', np.sum(dead_risk_sorted >= risk_median))
            fig, ax = plt.subplots(figsize=(6,4), dpi=300)
            plt.bar(risk_x[risk_sorted<risk_median], risk_sorted[risk_sorted<risk_median],  width=1, color='#BFE2F4', label='Low Risk')
            plt.bar(risk_x[risk_sorted>=risk_median], risk_sorted[risk_sorted>=risk_median],  width=1, color='#FED2cd', label='High Risk')
            
            plt.plot(dead_x_sorted, dead_risk_sorted, '^' , color='orange', label='Mortality Patient')
            plt.bar(dead_x_sorted, dead_risk_sorted,  width=1, color='orange')

            ax.set(
                title='Mortality Risk Distribution',
                xlabel='Number of PE Patients',
                ylabel='Survival Risk Prediction',
            )
            ax.legend(loc='upper right', fontsize=10)

            ax.set_ylim(0.4,1)
            
            ax.set_xlim(-0.02*len(risk), 1.02*len(risk))
            plt.savefig(mod+'dead.jpg')
            