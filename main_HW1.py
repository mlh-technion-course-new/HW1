# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 12:49:31 2019

@author: smorandv
"""

import pandas as pd
import pickle
import clean_data
from clean_data import rm_ext_and_nan as rm
from clean_data import sum_stat as sst
from clean_data import rm_outlier
from clean_data import norm_standard as nsd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from sklearn import linear_model
from lin_classifier import *
import scipy.stats as stats
from sklearn.ensemble import RandomForestClassifier as rfc
from pathlib import Path

if __name__ == '__main__':
    file = Path.cwd().joinpath('messed_CTG.xls') # concatenates messed_CTG.xls to the current folder that should be the extracted zip folder 
    CTG_dataset = pd.read_excel(file, sheet_name='Raw Data')
    CTG_features = CTG_dataset[['LB', 'AC', 'FM', 'UC', 'DL', 'DS', 'DR', 'DP', 'ASTV', 'MSTV', 'ALTV', 'MLTV',
     'Width', 'Min', 'Max', 'Nmax', 'Nzeros', 'Mode', 'Mean', 'Median', 'Variance', 'Tendency']]
    CTG_morph = CTG_dataset[['CLASS']]
    fetal_state = CTG_dataset[['NSP']]
    #####################
    extra_feature = 'DR' 
    c_ctg = rm(CTG_features, extra_feature)
    ####################
    feat = 'Width'
    Q = pd.DataFrame(CTG_features[feat])
    idx_na = Q.index[Q[feat].isna() == True].tolist()
    for i in idx_na:
        Q.loc[i] = 1000
    Q.hist(bins = 100)
    plt.xlabel('Histogram Width')
    plt.ylabel('Count')
    plt.show()
    ######################
    feat = 'Width'
    Q_clean = pd.DataFrame(c_ctg[feat])
    Q_clean.hist(bins=100)
    plt.xlabel('Histogram Width')
    plt.ylabel('Count')
    plt.show()
    #####################
    extra_feature = 'DR' 
    c_cdf = nan2num_cdf(CTG_features, extra_feature)
    #####################
    feat = 'MSTV'
    print(CTG_features[feat].iloc[0:5]) # print first 5 values
    print(c_cdf[feat].iloc[0:5])
    ######################
    # Boxplots
    c_cdf.boxplot(column=['Median','Mean','Mode'])
    plt.ylabel('Fetal Heart Rate [bpm]')
    # Histograms
    xlbl = ['a','b','c','d']
    axarr = c_cdf.hist(column=['LB', 'AC', 'UC','ASTV'], bins=100,layout = (2, 2),figsize=(20, 10))
    for i,ax in enumerate(axarr.flatten()):
        ax.set_xlabel(xlbl[i])
        ax.set_ylabel("Count")
    # Barplots (error bars)
    df = pd.DataFrame.from_dict({'lab':['Min','Max'], 'val':[np.mean(c_cdf['Min']), np.mean(c_cdf['Max'])]})
    errors = [np.std(c_cdf['Min']), np.std(c_cdf['Max'])] 
    ax = df.plot.bar(x='lab', y='val', yerr=errors, rot=0)
    ax.set_ylabel('Average value')
    plt.show()
    #########################
    d_summary = sst(c_cdf)
    ########################
    c_no_outlier = rm_outlier(c_cdf, d_summary)
    ###########################
    c_cdf.boxplot(column=['Median','Mean','Mode'])
    plt.ylabel('Fetal Heart Rate [bpm]')
    plt.show()
    ###########################
    c_no_outlier.boxplot(column=['Median','Mean','Mode'])
    plt.ylabel('Fetal Heart Rate [bpm]')
    plt.show()
    ###########################
    feature = 'LB' # change this feature
    thresh = 500 # FHR cannot be more then 500 bpm. Change this threshold accordingly to the feature
    filt_feature = phpr(c_cdf, feature, thresh)
    ###########################
    with open('objs.pkl', 'rb') as f:
        CTG_features, CTG_morph, fetal_state = pickle.load(f)
    orig_feat = CTG_features.columns.values
    ##############################
    selected_feat = ('LB','ASTV')
    orig = nsd(CTG_features,selected_feat,flag=True)
    nsd_std = nsd(CTG_features,selected_feat,mode='standard',flag=True)
    nsd_norm = nsd(CTG_features,selected_feat,mode='MinMax',flag=True)
    nsd_norm_mean = nsd(CTG_features,selected_feat,mode='mean',flag=True)
    ################################
    g = sns.countplot(x = 'NSP', data = fetal_state)
    g.set(xticklabels=['Normal','Suspect','Pathology'])
    plt.show()
    idx_1 = (fetal_state == 1).index[(fetal_state == 1)['NSP'] == True].tolist()
    idx_2 = (fetal_state == 2).index[(fetal_state == 2)['NSP'] == True].tolist()
    idx_3 = (fetal_state == 3).index[(fetal_state == 3)['NSP'] == True].tolist()
    print("Normal samples account for " + str("{0:.2f}".format(100*len(idx_1)/len(fetal_state))) + "% of the data.")
    print("Suspect samples account for " + str("{0:.2f}".format(100*len(idx_2)/len(fetal_state))) + "% of the data.")
    print("Pathology samples account for " + str("{0:.2f}".format(100*len(idx_3)/len(fetal_state))) + "% of the data.")
    ##################################
    bins = 100
    feat = 'Width'
    plt.hist(CTG_features[feat].loc[idx_1], bins, density=True, alpha=0.5, label='Normal')
    plt.hist(CTG_features[feat].loc[idx_2], bins, density=True, alpha=0.5, label='Suspect')
    plt.hist(CTG_features[feat].loc[idx_3], bins, density=True, alpha=0.5, label='Pathology')
    plt.xlabel('Histigram Width')
    plt.ylabel('Probabilty')
    plt.legend(loc='upper right')
    plt.show()
    ####################################
    pd.plotting.scatter_matrix(CTG_features[['LB','AC','FM','UC']])
    plt.show()
    ######################################
    orig_feat = CTG_features.columns.values
    X_train, X_test, y_train, y_test = train_test_split(CTG_features, np.ravel(fetal_state), test_size=0.2, random_state=0, stratify=np.ravel(fetal_state))
    logreg = LogisticRegression(solver='saga', multi_class='multinomial', penalty='none', max_iter=10000)
    y_pred, w = pred_log(logreg, X_train, y_train, X_test)
    print("Accuracy is: " + str("{0:.2f}".format(100 * metrics.accuracy_score(y_test, y_pred))) + "%")
    print("F1 score is: " + str("{0:.2f}".format(100 * metrics.f1_score(y_test, y_pred, average='macro'))) + "%")
    ######################################
    #mode = # choose a mode from the `nsd` 
    y_pred, w_norm_std = pred_log(logreg,) # complete this function using nsd function
    print("Accuracy is: " + str("{0:.2f}".format(100 * metrics.accuracy_score(y_test, y_pred))) + "%")
    print("F1 score is: " + str("{0:.2f}".format(100 * metrics.f1_score(y_test, y_pred, average='macro'))) + "%")
    #####################################
    #input_mat =  # Fill this argument
    #w_no_p_table(input_mat,orig_feat)
    ########################################
    cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
    ax = plt.subplot()
    sns.heatmap(cnf_matrix, annot=True, xticklabels=['Normal','Suspect','Pathology'], yticklabels=['Normal','Suspect','Pathology'])
    ax.set(ylabel='True labels', xlabel='Predicted labels')
    plt.show()
    #########################################
    #mode = # choose a mode from the `nsd`
    logreg_l2 = LogisticRegression(solver='saga', multi_class='multinomial', max_iter=10000) # complete the arguments for L2
    y_pred_2, w2 = pred_log(logreg_l2,) # complete this function using nsd function
    cnf_matrix = metrics.confusion_matrix(y_test, y_pred_2)
    ax1 = plt.subplot(211)
    sns.heatmap(cnf_matrix, annot=True, xticklabels=['Normal','Suspect','Pathology'], yticklabels=['Normal','Suspect','Pathology'])
    ax1.set(ylabel='True labels', xlabel='Predicted labels')
    plt.show()
    logreg_l1 = LogisticRegression(solver='saga', multi_class='multinomial', max_iter=10000) # complete the arguments for L1
    y_pred_1, w1 = pred_log(logreg_l1,)  # complete this function using nsd function
    cnf_matrix = metrics.confusion_matrix(y_test, y_pred_1)
    ax2 = plt.subplot(212)
    sns.heatmap(cnf_matrix, annot=True, xticklabels=['Normal','Suspect','Pathology'], yticklabels=['Normal','Suspect','Pathology'])
    ax2.set(ylabel='True labels', xlabel='Predicted labels')
    plt.show()
    ###############################################
    w_all_tbl(w2, w1, orig_feat)
    #############################################
    #C =  # make a list of up to 6 different values of regularization parameters and examine their effects
    #K =  # choose a number of folds
    #mode =  # mode of nsd function
    val_dict = cv_kfold(X_train, y_train, C=C, penalty=['l1', 'l2'], K=K,mode=mode)
    ################################################
    for d in val_dict:
        x = np.linspace(0, d['mu'] + 3 * d['sigma'], 1000)
        plt.plot(x,stats.norm.pdf(x, d['mu'], d['sigma']), label="p = " + d['penalty'] + ", C = " + str(d['C'])) 
        plt.title('Gaussian distribution of the loss')
        plt.xlabel('Average loss')
        plt.ylabel('Probabilty density')
    plt.legend()
    plt.show()
    ##################################################
    #C =  # complete this part according to your best result
    #penalty =  # complete this part according to your best result
    logreg = LogisticRegression(solver='saga', multi_class='multinomial', penalty=penalty, C=C, max_iter=10000)
    y_pred, w = pred_log(logreg,) # complete this function using nsd function
    cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
    ax1 = plt.subplot(211)
    sns.heatmap(cnf_matrix, annot=True, xticklabels=['Normal','Suspect','Pathology'], yticklabels=['Normal','Suspect','Pathology'])
    ax1.set(ylabel='True labels', xlabel='Predicted labels')
    plt.show()
    print("Accuracy is: " + str("{0:.2f}".format(100 * metrics.accuracy_score(y_test, y_pred))) + "%")
    print("F1 score is: " + str("{0:.2f}".format(100 * metrics.f1_score(y_test, y_pred, average='macro'))) + "%")
    ################################################
    #mode =  # choose your method
    clf = rfc(n_estimators=10)
    clf.fit(nsd(X_train, mode=mode), y_train)
    y_pred = clf.predict(nsd(X_test, mode=mode))
    cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
    ax = plt.subplot()
    sns.heatmap(cnf_matrix, annot=True, xticklabels=['Normal', 'Suspect', 'Pathology'],
                yticklabels=['Normal', 'Suspect', 'Pathology'])
    ax.set(ylabel='True labels', xlabel='Predicted labels')
    plt.show()
    print("Accuracy is: " + str("{0:.2f}".format(100 * metrics.accuracy_score(y_test, y_pred))) + "%")
    print("F1 score is: " + str("{0:.2f}".format(100 * metrics.f1_score(y_test, y_pred, average='macro'))) + "%")