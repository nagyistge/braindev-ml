import numpy as np
import nibabel as nb
import joblib as jl
import scipy.stats as ss
import time
import os
import sklearn.svm as sklsvm
import sklearn.preprocessing as sklpre
import sklearn.cross_validation as sklcv
import sklearn.feature_selection as sklfs
import sklearn.linear_model as skllm
import sklearn.metrics as sklm
import sklearn.cluster as sklcl
import sklearn.feature_extraction.image as sklim
from sklearn.grid_search import GridSearchCV
import sklearn.decomposition as skldec

import handy_functions as hf

#-------------Functions
def prepare_modality(filelist, mask):
    '''calculate feature matrix for files'''
    matrix = hf.format_grouped_images([filelist])
    mask_matrix = hf.format_grouped_images([[mask]])
    indata_matrix = matrix[0][mask_matrix[0][:,0]>0,:]
    indata_matrix = indata_matrix.T
    return indata_matrix

def wardCV(data, labels, cut_level, connect):
    '''calculate cross-validated amount of ward-clusters'''
    #loop for list
    accuracies = np.zeros(len(cut_level))
    for i in cut_level:
        #reduce to set amount of clusters
        agglo = sklcl.WardAgglomeration(connectivity=connect, n_clusters=i)
        cross = sklcv.KFold(n=len(labels), n_folds=10)
        pred_vec = np.zeros_like(labels)
        for train_i, test_i in cross:
            use_train = agglo.fit_transform(data[train_i])
            use_test = agglo.transform(data[test_i])
            model = sklsvm.NuSVR(kernel='linear',nu=1,C=1000)
            model.fit(use_train,labels[train_i])
            pr = model.predict(use_test)
            pred_vec[test_i] = pr
        #save accuracy
        accuracies[cut_level==i], _ = ss.pearsonr(pred_vec,labels)
    #based on loo-accuracy, select the optimal number of features
    #TODO -smooth this?
    best_model = cut_level[accuracies.argmax()]
    return best_model

def univCV(data,labels,cut_level, ward_level, connect):
    '''calculate cross-validated univariate cut'''
    #loop for the list
    correlations = np.zeros(len(cut_level))
    for inx, i in enumerate(cut_level):
        cross = sklcv.KFold(n=len(labels), n_folds=10)
        prediction = np.zeros_like(labels)
        for train, test in cross:
            univ = sklfs.SelectFpr(sklfs.f_regression,
                    alpha=i)
            univ_agglo = sklcl.WardAgglomeration(connectivity=connect, n_clusters=ward_level)
            prep_data = univ_agglo.fit_transform(data[train])
            prep_data = univ.fit_transform(prep_data, labels[train])
            mod = sklsvm.NuSVR(kernel='linear', nu=1 ,C=1000)#Change model
            mod.fit(prep_data, labels[train])
            prep_test = univ_agglo.transform(data[test])
            prep_test = univ.transform(prep_test)
            pred = mod.predict(prep_test)
            prediction[test] = pred
        #calculate prediction
        correlations[inx], _ = ss.pearsonr(prediction,labels)
    #TODO - smooth this?
    best_cut = cut_level[correlations.argmax()]
    return best_cut

def do_model(train_d, train_l, test_d, connect):

    #chose number of ward clusters
    no_feat = len(train_d[0,:])
    ward_sizes = np.array([int(no_feat),int(no_feat*0.8),int(no_feat*0.5),int(no_feat*0.1),int(no_feat*0.01)]) # set to about 100, 50 and 10% add 1/10000 for dbm
    #use_wardsize = 100
    use_wardsize=wardCV(train_d, train_l, ward_sizes, connect)
    agglo = sklcl.WardAgglomeration(connectivity=connect, n_clusters=use_wardsize)

    #chose univariate cutoff
    univ_levels = np.array([1,0.1,0.01])#set univariate cutoff here
    #use_cut = 0.1
    use_cut = univCV(train_d, train_l, univ_levels,use_wardsize,connect)
    univ_select = sklfs.SelectFpr(alpha=use_cut)

    #define model
    nus = np.array([1,0.5,0.1])#set nu threshold
    #use_nu = 0.5
    params = dict (nu=nus)
    #model = sklsvm.NuSVR(kernel='linear',C=1000, degree=1,nu=use_nu)
    model = GridSearchCV(estimator = sklsvm.NuSVR(kernel='linear',C=10, degree=1) #changed from 1000 to 10 for dbm
            , param_grid=params, cv=10, n_jobs= 1, loss_func=sklm.mean_squared_error)

    #train model
    train_d = agglo.fit_transform(train_d)
    train_d = univ_select.fit_transform(train_d, train_l)
    model.fit(train_d, train_l)

    #test model
    test_d = agglo.transform(test_d)
    test_d = univ_select.transform(test_d)
    pred = model.predict(test_d)

    use_nu = model.best_params_['nu']

    results = [pred, use_wardsize, use_cut, use_nu]



    return results


def run_pipe(input_files, input_labels, input_mask):
    '''run svr forkflow on data'''

    #TODO - track running time

    #--------------Organise inputs
    #calculate matrix
    feature_matrix = prepare_modality(input_files, input_mask)
    #--------------Execute analysis
    #prepare feature agglomeration
    mask_handle = nb.load(input_mask)
    connect = sklim.grid_to_graph(*mask_handle.shape, mask=mask_handle.get_data()>0)
    #cross validation
    loo = sklcv.KFold(len(input_labels), n_folds=len(input_labels))
    print('Starting svr')

    cv_pred = jl.Parallel(n_jobs=12, verbose = 1,pre_dispatch=20)(jl.delayed(do_model)(
        feature_matrix[train], input_labels[train], feature_matrix[test], connect)
        for train, test in loo)
    cv_pred = np.array(cv_pred)
    #np.save('svr_pipelog'+time.strftime('%H:%M:%S'),cv_pred)
    corr, p = ss.pearsonr(cv_pred[:,0], input_labels)
    #creating final model
    print('creating final model')
    final_agglo = sklcl.WardAgglomeration(connectivity=connect,
            n_clusters=int(np.median(cv_pred[:,1])))
    final_univ = sklfs.SelectFpr(alpha=np.median(cv_pred[:,2]))
    final_model = sklsvm.NuSVR(kernel='linear',C=1000, degree=1,
            nu=np.median(cv_pred[:,3]))

    feature_matrix = final_agglo.fit_transform(feature_matrix)
    feature_matrix = final_univ.fit_transform(feature_matrix, input_labels)
    final_model.fit(feature_matrix, input_labels)

    return cv_pred, corr, p, final_agglo, final_univ, final_model
