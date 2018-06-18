import numpy as np
from xcdf import XCDFFile
from glob import glob
import cPickle as pickle
import os
import sys
import matplotlib.colors
import matplotlib.pyplot as plt
import scipy.interpolate
import scipy.ndimage
import plot

#generate dataset of basic features of events
def gen_gamma_params(path="./HAWC/"):
    path = os.path.join(path, 'gamma/*.xcd')
    print(path)
    files = glob(path)
    print(files)
    total_data = []
    for xcdf_file in files:
        print(xcdf_file)
        xf = XCDFFile(xcdf_file)
        data = []
        # Normal 1D (no conditional)
        # params = "rec.logNPE, rec.nHit, rec.nTankHit, rec.zenithAngle, rec.azimuthAngle, rec.coreX, rec.coreY, rec.CxPE40PMT"
        # For conditional
        params = "rec.logNPE, rec.nHit, rec.nTankHit, rec.zenithAngle, rec.azimuthAngle, rec.coreX, rec.coreY, rec.CxPE40PMT, \
        SimEvent.energyTrue, SimEvent.thetaTrue, SimEvent.phiTrue"
        for param in xf.fields(params):
            if abs(param[3] - np.pi) > .01:
                data.append(param)
        total_data.extend(data)
    total_data = np.array(total_data, dtype=np.float32)
    # Simple data augmentation
    total_data[:, 1] = np.log(total_data[:, 1]) # Take the log of rec.nHit
    total_data[:, 8] = np.log(total_data[:, 8]) # For conditional only (Very important to have)
    assert total_data.shape == (total_data.shape[0], 11)  # 8 expected for no condition, 11 expected for conditional
    print("shuffling")
    np.random.shuffle(total_data)
    print(total_data[:15, :])
    np.save("gamma_data", total_data)

#
# generate 40x40 images, each pixel is either 0 or mapped to a PMT
# using mapping of tanks to pixels
# Note that some pixels do not have a corresponding tank so they are always 0
# pixel_range corresponds normalizing the images from 0-255
def gen_images_mapping(path="./HAWC/", sub='gamma/', display=False, log=True, train_split=0.8, two_dims=False,
                       small=False, normalize=False, pixel_range=True):
    from squaremapping import sqmap
    files = glob(os.path.join(path, sub, '*.xcd'))
    total_data = []
    labels = []
    params = "event.hit.charge, event.hit.time, event.hit.gridId, " \
             "rec.zenithAngle, rec.azimuthAngle"
    # try just the first .xcd file
    if small: files = files[0:1]
    for xcdf_file in files:
        print(xcdf_file)
        xf = XCDFFile(xcdf_file)
        # We generate a 40x40 grid for each event, from a mapping of gid -> x, y positions on grid
        for charge, time, gridid, zen, azi in list(xf.fields(params)):
            # first dim = charge, second dim = time
            if two_dims:
                grid = np.zeros((40, 40, 2))
                grid[:,:,1] = np.full((40, 40), -500.)
            else:
                grid = np.zeros((40, 40, 1))
            for c, gid, t in zip(charge, gridid, time):
                if gid > 8:
                    coorsq = sqmap[int(gid)]
                    # smallest charge is 0.1, anything smaller is 0, so we can take a safe log
                    if log: c = max(np.log(c + 1e-8) + 2.302585, 0.)
                    grid[coorsq[0], coorsq[1], 0] = c
                    if two_dims: grid[coorsq[0], coorsq[1], 1] = t if not np.isclose(c, 0) else 500.
            # grid /= max(grid.flatten())
            total_data.append(grid)
            labels.append([zen, azi])
    total_data = np.array(total_data, dtype=np.float32)
    labels = np.array(labels)
    print('data shape', total_data.shape, 'labels shape', labels.shape) # (N, 40, 40, 2)
    # We can normalize values to [0, 255] to put data in same domain as images, or [-1, 1]
    if normalize:
        min_vals, max_vals = [], []
        # normalize each channel independently
        for i in range(total_data.shape[3]):
            min_val, max_val = np.amin(total_data[:,:,:,i]), np.amax(total_data[:,:,:,i])
            min_vals.append(min_val), max_vals.append(max_val)
            print('Dimension %i, min: %f, max: %f' %(i, min_val, max_val))
        # Math to normalize channels to between [-1, 1]
        # Forward: -1. + 2. * (x + 4.605170) / (9.210341 + 4.605170)
        # -1. + 2. * (x + 500.) / (1000. - 500.)
        # inverse: (y + 1) / 2 * (max - min) + min
        # (y + 1) / 2 * (9.210341 + 4.605170) - 4.605170
        # (y + 1) / 2 * (1000. + 500.) - 500.
        dims = []
        # dims.append(-1. + 2. * (total_data[:,:,:,0] - min_vals[0]) / (max_vals[0] - min_vals[0]))
        
        if pixel_range:
            dims.append(total_data[:,:,:,0] * 255. / max_vals[0]) # we normalize to [0, 255]
        if two_dims:
            # assert False, "not computed yet for [0, 255]"
            # dims.append(-1. + 2. * (total_data[:,:,:,1] - min_vals[1]) / (max_vals[1] - min_vals[1]))
            dims.append(255. * (total_data[:,:,:,1] - min_vals[1]) / (max_vals[1] - min_vals[1]))
        print(dims[0].shape)
        # sanity check for normalization
        for d in dims:
            assert np.amax(d) <= 255.01, np.amax(d)
            assert np.amin(d) >= -0.01, np.amin(d)
        total_data = np.stack(dims, axis=3)
    print("shuffling")
    np.random.seed(0)
    p = np.random.permutation(len(total_data))
    total_data, labels = total_data[p], labels[p]
    if display:
        plot.plot_hists(total_data)
        plot.plot_40x40(total_data[:16], 'ground truth - gamma - log - 40x40 grid')
        plot.plot_pmts(total_data[:16], 'ground truth - gamma - log - pmts', sparse=True)
        for i in range(5, 16):
            plot.plot_pmts(total_data[:16], 'ground truth - gamma - log - pmts - single', sparse=True, single=i)
    split = int(train_split * len(total_data))
    print("split size:", split)
    train_data, test_data, train_labels, test_labels = \
        total_data[:split], total_data[split:], labels[:split], labels[split:]
    path = os.path.join(path, 'data/')
    if not os.path.exists(path):
        os.makedirs(path)
    # create both training and testing data
    suffix = "_2" if two_dims else ""
    np.save(path+"gamma_image_mapping_data" + suffix, train_data)
    np.save(path+"gamma_labels" + suffix, train_labels)
    np.save(path+"gamma_test_image_mapping_data" + suffix, test_data)
    np.save(path+"gamma_test_labels" + suffix, test_labels)

#
# attempt to generate image using interpolation of PMT locations
# doesn't work too well...
#
def gen_images_interp(path="./HAWC/", sub='gamma/'):
    files = glob(os.path.join(path, sub, '*.xcd'))
    total_data = []
    labels = []
    params = "event.hit.effcharge, event.hit.time, event.hit.gridId, " \
             "rec.zenithAngle, rec.azimuthAngle"
    layout = np.load("./data/layout.npy")
    def layout_info():
        print(layout)
        minx, miny, minz, maxx, maxy, maxz = 1e6, 1e6, 1e6, -1, -1, -1
        for x, y, z in layout:
            if np.isclose(x, 0) and np.isclose(y, 0):
                continue
            if x < minx:
                minx = x
            if x > maxx:
                maxx = x
            if y < miny:
                miny = y
            if y > maxy:
                maxy = y
            if z < minz:
                minz = z
            if z > maxz:
                maxz = z
        print(minx, miny, minz, maxx, maxy, maxz)
        print(maxx-minx, maxy-miny, maxz-minz)
        return minx, miny, maxx, maxy
    minx, miny, maxx, maxy = layout_info()
    good_cords = layout[:, 0] != 0.0
    def plot_grid():
        clean = good_cords
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(clean(layout[:, 0]), clean(layout[:, 1]), clean(layout[:, 2]))
        plt.show()
    print(layout.shape)
    x_cord = layout[:, 0]
    y_cord = layout[:, 1]
    grid_x, grid_y = np.mgrid[minx:maxx:60j, miny:maxy:40j]
    for xcdf_file in files[:1]:
        print(xcdf_file)
        xf = XCDFFile(xcdf_file)
        for charge, time, gridId, zen, azi in list(xf.fields(params)):
            A = np.zeros((1200, 2))
            for c, t, g in zip(charge, time, gridId):
                g -= 1
                A[g][0] = c
                A[g][1] = t
            # grid = scipy.interpolate.Rbf(x_cord, y_cord, A[:, 0], function='gaussian', smooth=0.5, epsilon=1.1)
            grid = scipy.interpolate.griddata(layout[:, :2], A[:, 0], (grid_x, grid_y), method='linear')
            # grid = grid(grid_x, grid_y)
            print(grid)
            def plot_interp():
                # plt.imshow(A[:, 0].reshape((30, 40)))
                # plt.show()
                print(grid.shape)
                # https://github.com/ednaruiz/ConvoNN/blob/master/MakeHawcImages.py
                plt.subplot(121)
                plt.scatter(x_cord[good_cords], y_cord[good_cords], s=50, c=A[:, 0][good_cords]+0.1, marker='o', linewidths=0.01, alpha=0.7, norm=matplotlib.colors.LogNorm())
                plt.subplot(122)
                plt.imshow(scipy.ndimage.rotate(grid, 90))
                plt.show()
            plot_interp()
            total_data.append(grid)
            labels.append([zen, azi])
    total_data = np.array(total_data, dtype=np.float32)
    labels = np.array(labels)
    print('data shape', total_data.shape, 'labels shape', labels.shape)
    # print("shuffling")
    # np.random.shuffle(total_data)
    path = 'data/'
    if not os.path.exists(path):
        os.makedirs(path)
    np.save(path+"gamma_image_data", total_data)
    np.save(path+"gamma_labels", labels)


# Gets the x, y, z coordinates of every PMT
def get_layout(path="./HAWC/", sub='gamma/'):
    files = glob(os.path.join(path, sub, '*.xcd'))
    pmt_locs = np.zeros((1200, 3))
    zeros = np.count_nonzero(pmt_locs)
    params = "event.hit.xPMT, event.hit.yPMT, event.hit.zPMT, event.hit.gridId"
    for xcdf_file in files:
        xf = XCDFFile(xcdf_file)
        print(xcdf_file)
        for xs, ys, zs, ids in list(xf.fields(params)):
            for x, y, z, id in zip(xs, ys, zs, ids):
                id -= 1
                if not np.isclose(pmt_locs[id][0], 0):
                    assert pmt_locs[id][0] == x
                if not np.isclose(pmt_locs[id][1], 0):
                    assert pmt_locs[id][1] == y
                if not np.isclose(pmt_locs[id][2], 0):
                    assert pmt_locs[id][2] == z
                pmt_locs[id][0] = x
                pmt_locs[id][1] = y
                pmt_locs[id][2] = z
        zeros = np.count_nonzero(pmt_locs)
        print("{0} out of {1} pixels have data".format(zeros, 1200*3))
        if zeros == 3378:
            break
    path = 'data/'
    if not os.path.exists(path):
        os.makedirs(path)
    np.save(path+"layout", pmt_locs)


if __name__ == "__main__":
    # Generate data for images
    if len(sys.argv) > 1:
        path = sys.argv[1]
        gen_images_mapping(path)
    else:
        if not os.path.exists("./data/layout.npy"):
            print("generating mapping of grid IDs to x/y/z coordinates")
            get_layout()
        gen_images_mapping("../daqsim-nvidia/", display=False, two_dims=False, small=False, normalize=True, pixel_range=False) #path="/home/danny/HAWC/", display=False)
    # Generate data for 1D distributions
    gen_gamma_params("../daqsim-nvidia/")