## braindev-ml
## Author: Henrik Ullman
## License: GPL Version 3

#Batch script for formating simulated data and input into processing workflow
#the labels are done by creating a mean from 10 pixels and adding some noise



#loading data
import skimage.io as skio
import glob
import numpy as np
import random
from svr_pipe import *

folder = 'artificial_data_segmentation/generated_data/*.jpg'
concat_data = np.concatenate([skio.imread(p,as_grey=True)[np.newaxis,:] for p in glob.glob(folder)])
concat_data = concat_data.astype(float)

#generate pseudolabels
subset = np.concatenate([concat_data[:,p,p][np.newaxis,:] for p in range(10,255,25)])
subset = subset.T

labels = np.mean(subset,1)
labels = 0.5*labels+0.5*np.array(random.sample(labels,len(labels)))
#labels = labels

result, corr, p, agglo, uni, model = run_pipe(concat_data,labels,'bc',2)


