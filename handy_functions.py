def format_grouped_images(group_list):
    import nibabel as nb

    #Make a subj x voxel 2d array for each group
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
