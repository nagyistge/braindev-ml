## braindev-ml
## Author: Henrik Ullman
## License: GPL Version 3

def format_grouped_images(group_list):
    """Make a subj x voxel 2d array for each group"""
    import nibabel as nb

    def make_matrix(inlist):
        images_4d = nb.funcs.concat_images(inlist)
        sub_data = images_4d.get_data()
        data_matrix = sub_data.transpose(3, 0, 1, 2).reshape(len(inlist), -1)
        out_data_matrix = data_matrix.transpose(1, 0)
        return out_data_matrix

    combined_matrix = []
    for idx, g in enumerate(group_list):
        combined_matrix.append(make_matrix(g))

    return combined_matrix


def kl_distance(p, q):
    """calculates KL-divergence for discrete distributions"""
    import numpy as np
    p = np.asarray(p, dtype=np.float)
    q = np.asarray(q, dtype=np.float)
    if (p == 0).sum() + (q == 0).sum() > 0:
        raise Exception, 'Zeros found'
    kl_dist = np.sum(p * np.log(p / q))
    return kl_dist


def make_4d_matrix(files):
    """creates 4d matrix from a list of .nii files"""
    import nibabel as nb
    import numpy as np
    dim_file = nb.load(files[0])
    dims = list(dim_file.shape)
    dims.append(len(files))
    out_matrix = np.zeros(dims)
    for inx, i in enumerate(files):
        handle = nb.load(i)
        out_matrix[:, :, :, inx] = handle.get_data()

    return out_matrix

def filter_4d_matrix(matrix, maskfile):
    """filter out mask and nan from 4d matrix"""
    import nibabel as nb
    import numpy as np
    mask_handle = nb.load(maskfile)
    maskdata = mask_handle.get_data()
    matrix[maskdata==0,:] = np.nan
    #sum_matrix = np.sum(matrix,0)
    #bool_matrix = np.isnan(sum_matrix)
    #matrix[bool_matrix,:] = 0
    #matrix = matrix.T

    return matrix


def mask_image(nii, maskfile,outname):
    """mask bimage with binary file"""
    import nibabel as nb
    import numpy as np
    matrix = nb.load(nii).get_data()
    mask_handle = nb.load(maskfile)
    maskdata = mask_handle.get_data()

    matrix[maskdata==0] = np.nan
    out_img=nb.Nifti1Image(matrix, mask_handle.get_affine())
    nb.save(out_img,outname)
    return



def information_map(input_4d, method='mi'):
    """calculate similarity of signal for each voxel compared to surrounding voxels
    input: input_4d - four-dimentional array with group or timeseries
    method - method to calculate signal similarity (mi, pearson, spearman)
    output: three-dimentional array with metric for the chosen method """
    import numpy as np
    import sklearn.metrics as skl_met
    import scipy.stats as ss
    import scipy.ndimage as si

    #Applying filtering prior to calculation
    #filter_ind = 0
    #temp_4d = np.zeros_like(input_4d.T)
    #for i in input_4d.T:
        #temp_4d[filter_ind] = si.median_filter(i,size=4)
    #    temp_4d[filter_ind] = si.gaussian_filter(i,0.5)
    #    filter_ind += 1

    #input_4d = temp_4d.T

    #scale data first?

    metric_map = np.zeros_like(input_4d[:, :, :, 0])
    metric_map = metric_map.astype(float)
    for fir in range(len(input_4d)):
        for sec in range(len(input_4d[0])):
            for thr in range(len(input_4d[(0, 0)])):
                left = input_4d[fir, sec, max(0, thr - 1):thr]
                right = input_4d[fir, sec, thr + 1:thr + 2]
                up = input_4d[fir, max(0, sec - 1):sec, thr]
                down = input_4d[fir, sec + 1:sec + 2, thr]
                front = input_4d[max(0, fir - 1):fir, sec, thr]
                back = input_4d[fir + 1:fir + 2, sec, thr]
                neigh_dist = [left.ravel(),
                 right.ravel(),
                 up.ravel(),
                 down.ravel(),
                 front.ravel(),
                 back.ravel()]
                use_neigh = 0
                for i in neigh_dist:
                    if len(i) != 0:
                        use_neigh += 1


                pruned_neigh_dist = np.zeros((use_neigh, len(input_4d[(0, 0, 0)])))
                add_neigh = 0
                for i in neigh_dist:
                    if i != []:
                        pruned_neigh_dist[add_neigh] = i
                        add_neigh += 1

                mean_neigh_dist = np.mean(pruned_neigh_dist, 0)
                center_dist = input_4d[fir, sec, thr]
                if method == 'mi':
                    _, bins_center = np.histogram(center_dist, 10) #10 bins chosen since it seems to work
                    _, bins_surround = np.histogram(mean_neigh_dist, 10)
                    discrete_center = np.digitize(center_dist, bins_center)
                    discrete_surround = np.digitize(mean_neigh_dist, bins_surround)
                    dist_metric = skl_met.normalized_mutual_info_score(discrete_center, discrete_surround)
                elif method == 'pearson':
                    dist_metric, _ = ss.pearsonr(mean_neigh_dist, center_dist)

                elif method == 'spearman':
                    dist_metric, _ = ss.spearmanr(mean_neigh_dist, center_dist)
                metric_map[fir, sec, thr] = dist_metric

    return metric_map


def correlation_map(filelist, labels):
    """calculate a correlation map with the labels"""
    pass


def smooth_4d_matrix(matrix,kernel_width):
    """apply smoothing kernel on a 4d array of data
    for each subject separately.
    Subject dimension should be the last dimension"""

    import scipy.ndimage as spim
    import numpy as np
    matrix = matrix.T
    matrix[np.isnan(matrix)] = 0
    for ind, subj in enumerate(matrix):
        subj_sm = spim.gaussian_filter(subj,kernel_width)
        matrix[ind] = subj_sm

    matrix = matrix.T
    return matrix


def regression_map(filelist, vector):
    import statsmodels.api as sm
    import numpy as np
    #import images
    matrix = make_4d_matrix(filelist)

    #create output array
    output_array = np.zeros_like(matrix[:,:,:,0])

    #reshape
    matrix = matrix.reshape((-1,len(filelist)))
    output_stats = output_array.ravel()
    #loop over voxels
    for ind, voxel in enumerate(matrix):
        X=sm.add_constant(vector)
        voxel_ols=sm.OLS(voxel,X)
        voxel_results=voxel_ols.fit()
        out_param=voxel_results.tvalues[1]
        output_stats[ind]=out_param

    output_array=output_stats.reshape(output_array.shape)
    return output_array

def average_movement(affine_list):
    """calculate a movement index based on a series of affine transformation"""
    import nipy.algorithms.registration as nar
    import numpy as np
    translation = np.zeros(len(affine_list))
    rotation = np.zeros(len(affine_list))

    for inx, i in enumerate(affine_list):
        affine_matrix = np.genfromtxt(i)
        affine_object = nar.Affine(affine_matrix)
        translation_par = np.abs(affine_object.translation)
        rotation_par = np.abs(affine_object.rotation)
        translation_par = np.mean(translation_par)
        rotation_par = np.mean(rotation_par)
        translation[inx] = translation_par
        rotation[inx] = rotation_par
    translation_mean = np.mean(translation_par)
    rotation_mean = np.mean(rotation_par)
    return rotation_mean, translation_mean

def combine_movement_vectors(rot,trans):
    import numpy as np

    def standardize(vector):
        vector = vector-np.median(vector)
        quartile_range = np.percentile(vector,75)-np.percentile(vector,25)
        vector = vector/(quartile_range)
        return vector

    rot_stand = standardize(rot)
    trans_stand = standardize(trans)

    move_index = (rot_stand+trans_stand)/2

    return move_index


