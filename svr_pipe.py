## braindev-ml
## Author: Henrik Ullman
## License: GPL Version 3
import numpy as np
import nibabel as nb
import handy_functions as hf
import sklearn.preprocessing as sklpre
import sklearn.svm as sklsvm
import joblib as jl
import sklearn.cross_validation as sklcv
import time
import sklearn.feature_selection as sklfs
import scipy.stats as ss
import sklearn.linear_model as skllm
import sklearn.metrics as sklm
import sklearn.cluster as sklcl
import sklearn.feature_extraction.image as sklim
from sklearn.grid_search import GridSearchCV
import sklearn.decomposition as skldec
import os


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

            scaler = sklpre.StandardScaler()
            use_train = scaler.fit_transform(use_train)
            use_test = scaler.transform(use_test)

            model = sklsvm.NuSVR(kernel='linear',nu=1,C=100)
            model.fit(use_train,labels[train_i])
            pr = model.predict(use_test)
            pred_vec[test_i] = pr
        #save accuracy
        accuracies[cut_level==i], _ = ss.pearsonr(pred_vec,labels)
    #based on loo-accuracy, select the optimal number of features
    #TODO -smooth this?
    best_model = cut_level[accuracies.argmax()]
    return best_model


def direction_cutoff(data):
    mean_vector = np.mean(data, axis=0)
    bool_pos = mean_vector>0
    bool_neg = mean_vector<0
    return bool_pos, bool_neg


#def univCV(data,labels,cut_level, ward_level, connect, use_modules):
def univCV(data,labels,cut_level):
    '''calculate cross-validated univariate cut'''
    #loop for the list
    correlations = np.zeros(len(cut_level))
    for inx, i in enumerate(cut_level):
        cross = sklcv.KFold(n=len(labels), n_folds=10)
        prediction = np.zeros_like(labels)
        for train, test in cross:
            univ = sklfs.SelectFpr(sklfs.f_regression,
                    alpha=i)
            prep_data = data[train]
            prep_test = data[test]

            #if use_modules.find('a') != -1:
            #    univ_agglo = sklcl.WardAgglomeration(connectivity=connect, n_clusters=ward_level)
            #    prep_data = univ_agglo.fit_transform(prep_data)
            #    prep_test = univ_agglo.transform(prep_test)

            #if use_modules.find('b') != -1:
            #    bool_pos, bool_neg = direction_cutoff(prep_data)
            #    prep_data = prep_data[:, bool_pos]
            #    prep_test = prep_test[:, bool_pos]

            #if use_modules.find('c') != -1:
            #    scaler = sklpre.StandardScaler()
            #    prep_data = scaler.fit_transform(prep_data)
            #    prep_test = scaler.transform(prep_test)

            prep_data = univ.fit_transform(prep_data, labels[train])
            mod = sklsvm.NuSVR(kernel='linear', nu=1 ,C=100)#Change model
            mod.fit(prep_data, labels[train])
            prep_test = univ.transform(prep_test)
            pred = mod.predict(prep_test)
            prediction[test] = pred
        #calculate prediction
        correlations[inx], _ = ss.pearsonr(prediction,labels)
    #TODO - smooth this?
    best_cut = cut_level[correlations.argmax()]

    return best_cut

def do_model(train_d, train_l, test_d, connect, use_modules):

    #ward clustering (a)
    if use_modules.find('a') != -1:
        no_feat = len(train_d[0,:])
        ward_sizes = np.array([int(no_feat),int(no_feat*0.8),int(no_feat*0.5),int(no_feat*0.1),int(no_feat*0.01)]) # set to about 100, 50 and 10% add 1/10000 for dbm
        use_wardsize=wardCV(train_d, train_l, ward_sizes, connect)
        agglo = sklcl.WardAgglomeration(connectivity=connect, n_clusters=use_wardsize)

        train_d = agglo.fit_transform(train_d)
        test_d = agglo.transform(test_d)
    else:
        use_wardsize = '0'

    #include positive values only(b)
    if use_modules.find('b') != -1:
        bool_pos, bool_neg = direction_cutoff(train_d)

        train_d = train_d[:, bool_pos]
        test_d = test_d[:, bool_pos]

    #scale features to z scores(c)
    if use_modules.find('c') != -1:
        scaler = sklpre.StandardScaler()

        train_d = scaler.fit_transform(train_d)
        test_d = scaler.transform(test_d)

    #univariate selection(d)
    if use_modules.find('d') != -1:
        univ_levels = np.array([1,0.5,0.1, 0.05,0.01, 0.005, 0.001, 0.0001])
        #use_cut = univCV(train_d, train_l, univ_levels,use_wardsize,connect,use_modules)
        use_cut = univCV(train_d, train_l, univ_levels)
        univ_select = sklfs.SelectFpr(alpha=use_cut)

        train_d = univ_select.fit_transform(train_d, train_l)
        test_d = univ_select.transform(test_d)
    else:
        use_cut = '0'


    #train model

    nus = np.array([1,0.8,0.5,0.1])#set nu threshold
    params = dict (nu=nus)
    model = GridSearchCV(estimator = sklsvm.NuSVR(kernel='linear',C=100, degree=1) #changed from 1000 to 10 for dbm
            , param_grid=params, cv=10, n_jobs= 1, scoring='r2')#TODO changed from mse

    model.fit(train_d, train_l)
    pred = model.predict(test_d)

    use_nu = model.best_params_['nu']
    results = [pred, use_wardsize, use_cut, use_nu]

    return results


def run_pipe(input_files, input_labels, use_modules, no_proc):
    '''run svr forkflow on data'''

    #--------------Organise inputs
    #calculate matrix
    #feature_matrix = prepare_modality(input_files, input_mask)
    #--------------Execute analysis
    #prepare feature agglomeration
    #mask_handle = nb.load(input_mask)
    connect = sklim.grid_to_graph(*input_files[0].shape, mask=np.invert(np.isnan(np.sum(input_files,0))))
    inshape = input_files.shape

    feature_matrix = input_files.reshape((inshape[0],-1))

    #remove nans
    sum_features = np.sum(feature_matrix,0)
    feature_matrix = feature_matrix[:,np.invert(np.isnan(sum_features))]


    #cross validation
    loo = sklcv.KFold(len(input_labels), n_folds=len(input_labels))
    print('Starting svr')

    cv_pred = jl.Parallel(n_jobs=no_proc, verbose = 1,pre_dispatch=no_proc*2)(jl.delayed(do_model)(
        feature_matrix[train], input_labels[train], feature_matrix[test], connect, use_modules)
        for train, test in loo)
    cv_pred = np.array(cv_pred)
    corr, p = ss.pearsonr(cv_pred[:,0], input_labels)

    #creating final model
    print('creating final model')
    if use_modules.find('a') != -1:
        final_agglo = sklcl.WardAgglomeration(connectivity=connect,
                n_clusters=int(np.median(cv_pred[:,1])))
        feature_matrix = final_agglo.fit_transform(feature_matrix)
    else:
        final_agglo = 0

    if use_modules.find('b') != -1:
        bool_pos, bool_neg = direction_cutoff(feature_matrix)
        feature_matrix = feature_matrix[:, bool_pos]

    if use_modules.find('c') != -1:
        final_scaler = sklpre.StandardScaler()
        feature_matrix = final_scaler.fit_transform(feature_matrix)


    if use_modules.find('d') != -1:
        final_univ = sklfs.SelectFpr(alpha=np.median(cv_pred[:,2]))
        feature_matrix = final_univ.fit_transform(feature_matrix, input_labels)
    else:
        final_univ = 0

    final_model = sklsvm.NuSVR(kernel='linear',C=100, degree=1,
            nu=np.median(cv_pred[:,3]))
    final_model.fit(feature_matrix, input_labels)

    return cv_pred, corr, p, final_agglo, final_univ, final_model
