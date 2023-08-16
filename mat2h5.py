import h5py
import scipy
import scipy.io as io
import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter
import cv2

def gaussian_filter_density(gt):
    #print(gt.shape)
    density = np.zeros(gt.shape, dtype=np.float32)
    gt_count = np.count_nonzero(gt)
    if gt_count == 0:
        return density

    pts = np.array(list(zip(np.nonzero(gt)[1], np.nonzero(gt)[0])))
    leafsize = 2048
    # build kdtree
    tree = scipy.spatial.KDTree(pts.copy(), leafsize=leafsize)
    # query kdtree
    distances, locations = tree.query(pts, k=4)

    #print 'generate density...'
    for i, pt in enumerate(pts):
        pt2d = np.zeros(gt.shape, dtype=np.float32)
        pt2d[pt[1],pt[0]] = 1.
        if gt_count > 1:
            sigma = (distances[i][1]+distances[i][2]+distances[i][3])*0.1
        else:
            sigma = np.average(np.array(gt.shape))/2./2. #case: 1 point
        density += gaussian_filter(pt2d, sigma, mode='constant')
    #print 'done.'
    ###### ADDED BY MYSELF: SHRINK SIZE #########
    # there are 3 maxpool layers => divide size by 2^3 = 8
    # note: this was done when creating the dataset in the original code.
    shrink_shape = (int(density.shape[1]//8), int(density.shape[0]//8))
    density      = cv2.resize(density, shrink_shape, interpolation = cv2.INTER_CUBIC) * 64
    ########################################
    return density

def make_hdf5(img_path, mat_path, out_path):
    mat = io.loadmat(mat_path)
    img= plt.imread(img_path)
    k = np.zeros((img.shape[0],img.shape[1]))
    gt = mat["image_info"][0,0][0,0][0]
    for i in range(0,len(gt)):
        if int(gt[i][1])<img.shape[0] and int(gt[i][0])<img.shape[1]:
            k[int(gt[i][1]),int(gt[i][0])]=1
    k = gaussian_filter_density(k)
    with h5py.File(out_path, 'w') as hf:
            hf['density'] = k

if __name__ == "__main__":
    shangai  = "/Users/xgillard/Downloads/ShanghaiTech_Crowd_Counting_Dataset/part_A_final"
    img_path = f"{shangai}/train_data/images/IMG_1.jpg"
    mat_path = f"{shangai}/train_data/ground_truth/GT_IMG_1.mat"
    out_path = f"{shangai}/train_data/GT_IMG_1.h5"
    make_hdf5(img_path, mat_path, out_path)
