import h5py
import scipy
import numpy as np
from matplotlib    import pyplot as plt
from scipy.ndimage import gaussian_filter
from math          import floor
from statistics    import stdev
from PIL           import Image
from os            import listdir, makedirs
from shutil        import copyfile

# Note: 
# I have modified the original instructions to make the computations of the 
# ground truth both faster and more accurate (no need for cv2 resize with cubic
# interpolation -- which proves quite extensive)

def resize_dataset(source, dest):
    #
    for name in listdir(f"{source}/images"):
        if not name.endswith(".jpeg"):
            continue
        else:
            im = Image.open(f"{source}/images/{name}")
            w,h= im.size
            im = im.resize((w//4, h//4))
            im.save(f"{dest}/images/{name}")

    for gt in listdir(f"{source}/ground_truth"):
        if not gt.endswith(".npy"):
            continue
        else:
            copyfile(f"{source}/ground_truth/{gt}", f"{dest}/ground_truth/{gt}")

def gaussian_filter_density(gt):
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
        pt2d[pt[1],pt[0]] += 1.
        if gt_count > 1:
            sigma = stdev([distances[i][1], distances[i][2], distances[i][3]])
        else:
            sigma = np.average(np.array(gt.shape))/2./2. #case: 1 point
        density += gaussian_filter(pt2d, sigma, mode='constant') * gt[(pt[1], pt[0])]
    return density

def make_hdf5(img_path, mat_path, out_path):
    try:
        gt  = np.load(mat_path)
        img = plt.imread(img_path)
        w = img.shape[0]//8
        h = img.shape[1]//8
        k = np.zeros((w,h))
        for i in range(0,len(gt)):
            x = int(floor(gt[i][1] * w))
            y = int(floor(gt[i][0] * h))
            if x<w and y<h:
                k[x, y]+=1
        k = gaussian_filter_density(k)
        with h5py.File(out_path, 'w') as hf:
                hf['density'] = k
    except Exception as e:
        print(f"failed to process {img_path}")

def process_dataset(original, destination):
    imdir    = original + "/images/"
    matdir   = original + "/ground_truth/"
    dimdir   = destination + "/images/"
    dtruth   = destination + "/ground_truth/"
    #
    for im in listdir(imdir):
        if not im.endswith(".jpeg"):
            continue
        img_path = imdir  + im 
        mat_path = matdir + im.replace(".jpeg", ".npy")
        out_path = dtruth + "GT_" + im.replace(".jpeg", ".h5") 
        copyfile(img_path, dimdir + im)
        make_hdf5(img_path, mat_path, out_path)

if __name__ == "__main__":
    input  = "/Users/xgillard/Documents/REPO/csrnet-rs/resources/UCLouvain/a10-labeled"
    middle = "/Users/xgillard/Documents/REPO/csrnet-rs/resources/UCLouvain/a10-labeled-resized"
    output = "/Users/xgillard/Documents/REPO/csrnet-rs/resources/UCLouvain/a10-labeled-resized-h5"
    makedirs(f"{middle}/images",       exist_ok = True)
    makedirs(f"{middle}/ground_truth", exist_ok = True)
    makedirs(f"{output}/images",       exist_ok = True)
    makedirs(f"{output}/ground_truth", exist_ok = True)
    #
    resize_dataset(input, middle)
    process_dataset(middle, output)