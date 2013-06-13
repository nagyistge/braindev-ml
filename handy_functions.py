def format_grouped_images(group_list):
    '''Make a subj x voxel 2d array for each group'''
    import nibabel as nb

    def make_matrix(inlist):
        images_4d = nb.funcs.concat_images(inlist)
        sub_data = images_4d.get_data()
        data_matrix = sub_data.transpose(3,0,1,2).reshape(len(inlist),-1)
        out_data_matrix = data_matrix.transpose(1, 0)
        return out_data_matrix

    combined_matrix = []
    for idx, g in enumerate(group_list):
        combined_matrix.append(make_matrix(g))

    return combined_matrix

def kl_distance(p,q):
    '''calculates KL-divergence for discrete distributions'''
    import numpy as np

    p = np.asarray(p, dtype=np.float)
    q = np.asarray(q, dtype=np.float)

    if ((p==0).sum()+(q==0).sum()) > 0:
        raise Exception, "Zeros found"
    kl_dist = np.sum(p * np.log(p/q))
    return kl_dist


def information_map(filelist):
    '''calculate similarity of signal for each voxel compared to surrounding voxels'''
    pass

def correlation_map(filelist, labels):
    '''calculate a correlation map with the labels'''
    pass
