import os
import sys
sys.path.append(os.getcwd())
import numpy as np
from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import concordance_index_censored
from sksurv.linear_model import CoxPHSurvivalAnalysis
import pandas as pd
import hdf5storage as hds
from copy import deepcopy
from file_and_folder_operations import *
import torchtuples as tt # Some useful functions
from loss_surv import *

def process():
    random_state_i = 123
    global df_dict0
    df_dict = deepcopy(df_dict0)
    RIH_df = df_dict['RIH']
    RIH_df_D = RIH_df.loc[RIH_df['Death'] == 1].drop('DeathDate', axis=1)
    RIH_df_noD = RIH_df.loc[RIH_df['Death'] == 0].drop('DeathDate', axis=1)
    RIHtr_D, RIHval_D, RIHts_D = \
        np.split(RIH_df_D.sample(frac=1, random_state=random_state_i),
                 [int(.7 * len(RIH_df_D)), int(.8 * len(RIH_df_D))])
    RIHtr_noD, RIHval_noD, RIHts_noD = \
        np.split(RIH_df_noD.sample(frac=1, random_state=random_state_i),
                 [int(.7 * len(RIH_df_noD)), int(.8 * len(RIH_df_noD))])

    RIHtr_df = pd.merge(RIHtr_D, RIHtr_noD, how='outer').sample(frac=1, random_state=random_state_i)
    RIHval_df = pd.merge(RIHval_D, RIHval_noD, how='outer').sample(frac=1, random_state=random_state_i)
    RIHts_df = pd.merge(RIHts_D, RIHts_noD, how='outer').sample(frac=1, random_state=random_state_i)
    TMH_NPH_df = pd.merge(df_dict['TMH'], df_dict['NPH'], how='outer').sample(frac=1, random_state=random_state_i)

    df_dict = {
               "RIHtr":RIHtr_df,
               "RIHval":RIHval_df,
               "RIHts":RIHts_df,
               'RIH': df_dict['RIH'],
               "TMH": df_dict['TMH'],
                "NPH": df_dict['NPH'],
                "TMH_NPH": TMH_NPH_df,
               }

    feat_dict = {}
    event_dict = {}
    time_dict = {}
    AHA_dict = {}
    PESI_dict = {}
    PESI_caseid_dict = {}
    PESI_prams_dict = {}
    race_dict = {}
    ethnicity_dict = {}
    img_feat_root = './penet_feat_avgpool_out'
    for trts in list(df_dict.keys()):
        feat_dict[trts] = []
        event_dict[trts] = []
        time_dict[trts] = []
        AHA_dict[trts] = []
        PESI_caseid_dict[trts] = []
        PESI_dict[trts] = []
        PESI_prams_dict[trts] = []
        race_dict[trts] = []
        ethnicity_dict[trts] = []
        for i_0, i in enumerate(df_dict[trts].index):
            d = df_dict[trts].loc[i]
            c_name = d['AccessionNumber_md5']
            img_modal = d['img_modal']
            img_name = c_name + '_' + img_modal + '_.npy'
            img_npy = np.load(join(img_feat_root, img_name))
            feat_dict[trts].append(deepcopy(img_npy))
            event_dict[trts].append(d['Death'])
            time_dict[trts].append(d['follow_up_day']) 
            if d['follow_up_day']* follow_up_max  == 1517:
                print()
            AHA_dict[trts].append(d['AHA_PE_severity'])
            PESI_caseid_dict[trts].append(i_0)
            PESI_dict[trts].append(d['PESI'])
            prams_list = [d[k] for k in ['Age', 'PatientSex', 'CA', 'CHF', 'COPD' , 'HR_gte110', 'SBP_lt100', 'RR', 'Temp', 'AMS', 'SpO2_lt90',]]
            PESI_prams_dict[trts].append(prams_list)
            race_dict[trts].append(d['Race'])
            ethnicity_dict[trts].append(d['Ethnicity'])
        feat_dict[trts] = deepcopy(np.array(feat_dict[trts]).squeeze())
        event_dict[trts] = deepcopy(np.array(event_dict[trts]).squeeze())
        time_dict[trts] = deepcopy(np.array(time_dict[trts]).squeeze())
        AHA_dict[trts] = deepcopy(np.array(AHA_dict[trts]).squeeze())
        PESI_dict[trts] = deepcopy(np.array(PESI_dict[trts]).squeeze())
        PESI_prams_dict[trts] = np.array(PESI_prams_dict[trts]).astype(np.float32)

    labels_dict = {}
    # convert the train labels
    for trts in list(df_dict.keys()):
        labels_dict[trts] = np.ndarray(shape=(event_dict[trts].shape[0],), dtype=[('status', '?'), ('survival_in_days', '<f8')])

        for ind_train in range(event_dict[trts].shape[0]):
            if event_dict[trts][ind_train] == 1:
                labels_dict[trts][ind_train] = (True, time_dict[trts][ind_train])
            else:
                labels_dict[trts][ind_train] = (False, time_dict[trts][ind_train])

    train_set = 'RIHtr'
    val_set = 'RIHval'
    y_dict = {}
    # convert the train labels
    y_dict[train_set] = (time_dict[train_set], event_dict[train_set])
    y_dict[val_set] = (time_dict[val_set], event_dict[val_set])
    # We don't need to transform the test labels
    test_keys = list(df_dict.keys())
    test_keys.remove(train_set)
    test_keys.remove(val_set)
    for trts in test_keys:
        y_dict[trts] = (time_dict[trts], event_dict[trts])
    # ---------RSF
    rsf_model_img = RandomSurvivalForest( random_state=100, n_jobs=1)
    rsf_model_img.fit(feat_dict[train_set], labels_dict[train_set])
    rsf_model_clin = RandomSurvivalForest( random_state=100, n_jobs=1)
    rsf_model_clin.fit(PESI_prams_dict[train_set], labels_dict[train_set])
    rsf_cph = CoxPHSurvivalAnalysis()
    rsf_fuse_PESI_cph = CoxPHSurvivalAnalysis()

    # ---------deep
    def train_model(feat_dict, num_nodes=[32, 32], device='cuda:0'):
        torch.manual_seed(0)
        val = (feat_dict[val_set], y_dict[val_set])
        in_features = feat_dict[train_set].shape[1]
        out_features = 1  # cox
        batch_norm = True
        dropout = 0.1
        output_bias = False
        criterion = CoxPHLoss()

        net = tt.practical.MLPVanilla(in_features, num_nodes, out_features, batch_norm,
                                      dropout, output_bias=output_bias, output_activation=torch.nn.Sigmoid())
        optimizer = tt.optim.AdamWR(0.01)
        model = tt.Model(net, criterion, optimizer, device=device,)

        batch_size = feat_dict[train_set].shape[0]
        # model.optimizer.set(0.01)
        epochs = 10000
        callbacks = [tt.cb.EarlyStopping(patience=100)]
        log = model.fit(feat_dict[train_set], y_dict[train_set], batch_size, epochs,
                        callbacks,
                        val_data=val, verbose=False)
        return model

    device = 'cuda:'+str(random_state_i%2)
    model_img = train_model(feat_dict, num_nodes=[1024, 512, 128], device=device)
    model_clin = train_model(PESI_prams_dict, num_nodes=[512, 128], device=device)
    pth_file = join(checkpoint_out_path, 'clin_model.pth')
    torch.save(model_clin.net, pth_file)
    deep_cph = CoxPHSurvivalAnalysis()

    deep_fuse_PESI_cph = CoxPHSurvivalAnalysis()

    out_c_ind_list = [random_state_i]
    # out_dicts
    PESI_c_ind_dict = {}
    rsf_img_dict = {}
    rsf_clin_dict = {}
    rsf_fuse_dict = {}
    rsf_fuse_PESI_dict = {}

    img_curves_dict = {}
    img_scores_dict = {}
    clin_curves_dict = {}
    clin_scores_dict = {}
    fuse_feat_dict = {}
    fuse_curves_dict = {}
    fuse_scores_dict = {}
    fusePESI_feat_dict = {}
    fusePESI_curves_dict = {}
    fusePESI_scores_dict = {}

    deep_img_dict = {}
    deep_clin_dict = {}
    deep_fuse_dict = {}
    deep_fuse_PESI_dict = {}

    deep_img_scores_dict = {}
    deep_clin_scores_dict = {}
    deep_fuse_feat_dict = {}
    deep_fuse_curves_dict = {}
    deep_fuse_scores_dict = {}
    deep_fusePESI_feat_dict = {}
    deep_fusePESI_curves_dict = {}
    deep_fusePESI_scores_dict = {}

    for trts in list(df_dict.keys()):
        # ----------PESI c index
        '''
        model.score ---> c index
        model.predict ----> risk score
        '''
        PESI_c_ind_dict[trts] = concordance_index_censored(event_dict[trts].astype(bool), time_dict[trts], PESI_dict[trts])[0]
        out_c_ind_list += [PESI_c_ind_dict[trts],]
        # ----------RSF c index
        rsf_img_dict[trts] = rsf_model_img.score(feat_dict[trts], labels_dict[trts])
        rsf_clin_dict[trts] = rsf_model_clin.score(PESI_prams_dict[trts], labels_dict[trts])

        risk_img = rsf_model_img.predict(feat_dict[trts])
        risk_clin = rsf_model_clin.predict(PESI_prams_dict[trts])
        feat_fuse = np.concatenate((np.array(risk_img)[None], np.array(risk_clin)[None]), axis=0)
        feat_fuse = np.transpose(feat_fuse)

        if trts == train_set:
            rsf_cph = rsf_cph.fit(feat_fuse, labels_dict[trts])
        rsf_fuse_dict[trts] = rsf_cph.score(feat_fuse, labels_dict[trts])
        risk_fuse = rsf_cph.predict(feat_fuse)

        feat_fusePESI = np.concatenate([risk_fuse[:,None], PESI_dict[trts][:,None]], axis=1)
        if trts == train_set:
            rsf_fuse_PESI_cph = rsf_fuse_PESI_cph.fit(feat_fusePESI, labels_dict[trts])

        rsf_fuse_PESI_dict[trts] = rsf_fuse_PESI_cph.score(feat_fusePESI, labels_dict[trts])
        risk_fusePESI = rsf_fuse_PESI_cph.predict(feat_fusePESI)

        out_c_ind_list += [rsf_img_dict[trts], rsf_clin_dict[trts],rsf_fuse_dict[trts], rsf_fuse_PESI_dict[trts]]
        # draw curve
        fuse_feat_dict[trts] = feat_fuse
        fusePESI_feat_dict[trts] = feat_fusePESI
        img_scores_dict[trts] = risk_img
        clin_scores_dict[trts] = risk_clin
        fuse_scores_dict[trts] = risk_fuse
        fusePESI_scores_dict[trts] = risk_fusePESI

        img_curves_dict[trts] = rsf_model_img.predict_survival_function(feat_dict[trts], return_array=True)
        clin_curves_dict[trts] = rsf_model_clin.predict_survival_function(PESI_prams_dict[trts], return_array=True)
        fuse_curves_dict[trts] = rsf_cph.predict_survival_function(feat_fuse)
        fuse_curves_dict[trts] = np.array([fuse_curves_dict[trts][i].y for i in range(len(fuse_curves_dict[trts]))])
        fusePESI_curves_dict[trts] = rsf_fuse_PESI_cph.predict_survival_function(feat_fusePESI )
        fusePESI_curves_dict[trts] = np.array([fusePESI_curves_dict[trts][i].y for i in range(len(fusePESI_curves_dict[trts]))])

        # ----------deep c index
        risk_img = model_img.predict(feat_dict[trts])
        risk_clin = model_clin.predict(PESI_prams_dict[trts])
        deep_img_dict[trts] = concordance_index_censored(event_dict[trts].astype(bool), time_dict[trts], risk_img[:,0])[0]
        deep_clin_dict[trts] = concordance_index_censored(event_dict[trts].astype(bool), time_dict[trts], risk_clin[:,0])[0]
        feat_fuse = np.concatenate((risk_img, risk_clin), axis=1)
        if trts == train_set:
            deep_cph = deep_cph.fit(feat_fuse, labels_dict[trts])
        deep_fuse_dict[trts] = deep_cph.score(feat_fuse, labels_dict[trts])
        risk_fused = deep_cph.predict(feat_fuse, )

        feat_fusePESI = np.concatenate([risk_fused[:,None], PESI_dict[trts][:,None]], axis=1)
        if trts == train_set:
            deep_fuse_PESI_cph = deep_fuse_PESI_cph.fit(feat_fusePESI, labels_dict[trts])
        deep_fuse_PESI_dict[trts] = deep_fuse_PESI_cph.score(feat_fusePESI, labels_dict[trts])
        risk_fusePESI = deep_fuse_PESI_cph.predict(feat_fusePESI)

        out_c_ind_list += [deep_img_dict[trts], deep_clin_dict[trts], deep_fuse_dict[trts], deep_fuse_PESI_dict[trts]]
        # draw curve
        deep_fuse_feat_dict[trts] = feat_fuse
        deep_fusePESI_feat_dict[trts] = feat_fusePESI
        deep_img_scores_dict[trts] = risk_img
        deep_clin_scores_dict[trts] = risk_clin
        deep_fuse_scores_dict[trts] = risk_fused
        deep_fusePESI_scores_dict[trts] = risk_fusePESI

        deep_fuse_curves_dict[trts] = deep_cph.predict_survival_function(feat_fuse)
        deep_fuse_curves_dict[trts] = np.array([deep_fuse_curves_dict[trts][i].y for i in range(len(deep_fuse_curves_dict[trts]))])
        deep_fusePESI_curves_dict[trts] = deep_fuse_PESI_cph.predict_survival_function(feat_fusePESI)
        deep_fusePESI_curves_dict[trts] = np.array([deep_fusePESI_curves_dict[trts][i].y for i in range(len(deep_fusePESI_curves_dict[trts]))])

    print(out_c_ind_list)

    """
    save the results for risk groups analysis
    """
    for trts in list(df_dict.keys()):
        path_curr = join(mat_out_path, str(random_state_i), trts)
        maybe_mkdir_p(path_curr)

        hds.savemat(join(path_curr, 'labels_all.mat'),
                    {
                        'event_train': event_dict['RIHtr'][:, None], 'time_train': time_dict['RIHtr'][:, None]*follow_up_max,
                        'event_test': event_dict[trts][:, None], 'time_test': time_dict[trts][:, None]*follow_up_max,
                    })

        hds.savemat(join(path_curr, 'RSF_results_for_risk_groups.mat'),
                    {
                    'PESI': PESI_dict[trts],
                    'feat_img': feat_dict[trts],
                    'scores_img': img_scores_dict[trts],
                    'curves_img': img_curves_dict[trts],
                    'feat_clin': PESI_prams_dict[trts],
                    'scores_clin': clin_scores_dict[trts],
                    'curves_clin': clin_curves_dict[trts],
                    'feat_fuse':  fuse_feat_dict[trts],
                    'scores_fuse': fuse_scores_dict[trts],
                    'curves_fuse': fuse_curves_dict[trts],
                    'feat_fusePESI':  fusePESI_feat_dict[trts],
                    'scores_fusePESI': fusePESI_scores_dict[trts],
                    'curves_fusePESI': fusePESI_curves_dict[trts],
                    'race': race_dict[trts],
                    'ethnicity': ethnicity_dict[trts],
                    })
        hds.savemat(join(path_curr, 'Deep_results_for_risk_groups.mat'),
                    {
                        'PESI': PESI_dict[trts],
                        'feat_img': feat_dict[trts],
                        'scores_img': deep_img_scores_dict[trts],
                        # 'curves_img': deep_img_curves_dict[trts],
                        'feat_clin': PESI_prams_dict[trts],
                        'scores_clin': deep_clin_scores_dict[trts],
                        # 'curves_clin': deep_clin_curves_dict[trts],
                        'feat_fuse': deep_fuse_feat_dict[trts],
                        'scores_fuse': deep_fuse_scores_dict[trts],
                        'curves_fuse': deep_fuse_curves_dict[trts],
                        'feat_fusePESI': deep_fusePESI_feat_dict[trts],
                        'scores_fusePESI': deep_fusePESI_scores_dict[trts],
                        'curves_fusePESI': deep_fusePESI_curves_dict[trts],
                        'race': race_dict[trts],
                        'ethnicity': ethnicity_dict[trts],
                    })

    '''statistics 
    '''

    df_dict_not_norm = deepcopy(df_dict)
    data_stat_dict = {}
    for trts in list(df_dict_not_norm.keys()):

        df_dict_not_norm[trts]['Age'] = (df_dict_not_norm[trts]['Age'] * age_max)
        df_dict_not_norm[trts]['follow_up_day'] = df_dict_not_norm[trts]['follow_up_day'] * follow_up_max

        data_stat_dict[trts] = {}
        data_stat_dict[trts]['PESI_c_ind'] = PESI_c_ind_dict[trts]
        data_stat_dict[trts]['RSF_img_c_ind'] = rsf_img_dict[trts]
        data_stat_dict[trts]['RSF_clin_c_ind'] = rsf_clin_dict[trts]
        data_stat_dict[trts]['RSF_fuse_c_ind'] = rsf_fuse_dict[trts]
        data_stat_dict[trts]['RSF_fusePESI_c_ind'] = rsf_fuse_PESI_dict[trts]
        data_stat_dict[trts]['DEEP_img_c_ind'] = deep_img_dict[trts]
        data_stat_dict[trts]['DEEP_clin_c_ind'] = deep_clin_dict[trts]
        data_stat_dict[trts]['DEEP_fuse_c_ind'] = deep_fuse_dict[trts]
        data_stat_dict[trts]['DEEP_fusePESI_c_ind'] = deep_fuse_PESI_dict[trts]
        for k in ['Num', 'Male', 'White or Caucasian', 'Not Hispanic or Latino', 'Death', 'follow_up_day_mean', 'follow_up_day_lt30', 'Age_gte80', 'CA', 'CHF', 'COPD',
                  'HR_gte110', 'SBP_lt100', 'RR_gte30', 'Temp_lt96.8°F', 'AMS', 'SpO2_lt90',]:
            if k =='Num':
                data_stat_dict[trts][k] = len(df_dict_not_norm[trts])
            if k == 'Male':
                data_stat_dict[trts][k] = df_dict_not_norm[trts]['PatientSex'].sum()
            if k =='follow_up_day_mean':
                data_stat_dict[trts][k] = df_dict_not_norm[trts]['follow_up_day'].mean()
            if k =='follow_up_day_lt30':
                data_stat_dict[trts][k] = df_dict_not_norm[trts].loc[df_dict_not_norm[trts]['follow_up_day']<30].__len__()
            if k =='Age_gte80':
                data_stat_dict[trts][k] = df_dict_not_norm[trts].loc[df_dict_not_norm[trts]['Age']>=80].__len__()
            if k =='Temp_lt96':
                data_stat_dict[trts][k] = df_dict_not_norm[trts]['Temp'].sum()
            if k =='RR_gte30':
                data_stat_dict[trts][k] = df_dict_not_norm[trts]['RR'].sum()
            if k =='White or Caucasian':
                data_stat_dict[trts][k] = df_dict_not_norm[trts].loc[df_dict_not_norm[trts]['Race']=='White or Caucasian'].__len__()
            if k =='Not Hispanic or Latino':
                data_stat_dict[trts][k] = df_dict_not_norm[trts].loc[df_dict_not_norm[trts]['Ethnicity']=='Not Hispanic or Latino'].__len__()
            if k in list(df_dict_not_norm[trts].keys()):
                data_stat_dict[trts][k] = df_dict_not_norm[trts][k].sum()

    return out_c_ind_list, data_stat_dict, df_dict_not_norm


if __name__ == '__main__':
    label_path = './PE_data/pe_label.xls'
    labels_df_s = pd.read_excel(label_path)
    age_max = labels_df_s['Age'].max()
    labels_df_s['Age'] = (labels_df_s['Age'] / age_max)
    follow_up_max = labels_df_s['follow_up_day'].max()
    labels_df_s['follow_up_day'] = labels_df_s['follow_up_day'] / follow_up_max

    RIH_df = labels_df_s.loc[labels_df_s['Var22'] == 'RIH']
    TMH_df = labels_df_s.loc[labels_df_s['Var22'] == 'TMH']
    NPH_df = labels_df_s.loc[labels_df_s['Var22'] == 'NPH']
    df_dict0 = {'RIH': RIH_df,
                "TMH": TMH_df,
                "NPH": NPH_df,
                }

    out_dict = []
    mat_out_path = './exps/mat_out/'
    maybe_mkdir_p(mat_out_path)
    stat_out_path = './exps/stat_out/'
    maybe_mkdir_p(stat_out_path)
    checkpoint_out_path = './exps/clinical_checkpoint_out/'
    maybe_mkdir_p(checkpoint_out_path)

    def save_data_sheets(data_dict, data_stat_dict):
        writer = pd.ExcelWriter(join(stat_out_path, 'clin_best_data.xlsx', ))
        values = {}
        values['set_name'] = list(data_stat_dict.keys())
        for k in list(data_stat_dict['NPH'].keys()):
            values[k] = []
            for s in values['set_name']:
                values[k].append(data_stat_dict[s][k])
        value_df = pd.DataFrame.from_dict(values)
        value_df.to_excel(excel_writer=writer, sheet_name='statistics', encoding="GBK")
        for k in data_dict.keys():
            data_dict[k].to_excel(excel_writer=writer, sheet_name=k, encoding="GBK")
        writer.save()
        writer.close()

    out_c_ind_list, data_stat_dict, data_dict = process()
    save_data_sheets(data_dict, data_stat_dict)




