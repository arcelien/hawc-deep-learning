import numpy as np
from xcdf import XCDFFile
from glob import glob
import os
import plot
import os.path as osp


def gen_gamma_params(path, save_path):
    """ Generate dataset of basic features (1D) of cosmic events"""
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
        # params = "rec.logNPE, rec.nHit, rec.nTankHit, rec.zenithAngle, rec.azimuthAngle, rec.coreX, rec.coreY, rec.CxPE40"
        # For conditional
        params = "rec.logNPE, rec.nHit, rec.nTankHit, rec.zenithAngle, rec.azimuthAngle, rec.coreX, rec.coreY, rec.CxPE40, \
        SimEvent.energyTrue, SimEvent.thetaTrue, SimEvent.phiTrue"
        for param in xf.fields(params):
            if abs(param[3] - np.pi) > .01:
                data.append(param)
        total_data.extend(data)
    total_data = np.array(total_data, dtype=np.float32)
    # Simple data augmentation
    total_data[:, 1] = np.log(total_data[:, 1])  # Take the log of rec.nHit
    total_data[:, 7] = np.log(total_data[:, 7] + .01)  # Take the log of rec.rec.CxPE40
    total_data[:, 8] = np.log(total_data[:, 8])  # For conditional only (Very important to have)
    assert total_data.shape == (total_data.shape[0], 11)  # 8 expected for no condition, 11 expected for conditional
    print("shuffling")
    np.random.shuffle(total_data)
    print(total_data[:15, :])
    np.save(osp.join(save_path, "gamma_data"), total_data)


def gen_images_mapping(path="./HAWC/", sub="gamma/", display=False, log=True, train_split=0.8, two_dims=False,
                       small=False, normalize=False, pixel_range=True, save_path="data/"):
    """
    generate 40x40 images, each pixel is either 0 or mapped to a PMT
    using mapping of tanks to pixels
    Note that some pixels do not have a corresponding tank so they are always 0
    pixel_range corresponds to normalizing the images from 0-255

    :param path: path to data directory
    :param sub: subdirectory / type of event (currently only tried gamma)
    :param display: whether to show matplotlib visualizations of parsed data
    :param log: take log of charge data
    :param train_split: ratio of train to test data
    :param two_dims: whether to include a second channel (time)
    :param small: only use one XCDF file to run sanity check
    :param normalize: normalize data between (-1, 1)
    :param pixel_range: normalize data between (0, 256), the valid range of pixel values
    """
    from squaremapping import sqmap  # sqmap is a mapping of {PMT Grid ID: index into 40x40 image}
    files = glob(os.path.join(path, sub, '*.xcd'))
    total_data = []
    labels = []
    params = "event.hit.charge, event.hit.time, event.hit.gridId, " \
             "rec.zenithAngle, rec.azimuthAngle"
    if small: files = files[0:1]
    for xcdf_file in files:
        print(xcdf_file)
        xf = XCDFFile(xcdf_file)
        for charge, time, gridid, zen, azi in list(xf.fields(params)):
            # first dim = charge, second dim = time
            # We generate a 40x40 grid for each event, from a mapping of gid -> x, y positions on grid
            if two_dims:
                grid = np.zeros((40, 40, 2))
                # fill the grid with the smallest possible value, -500
                grid[:, :, 1] = np.full((40, 40), -500.)
            else:
                grid = np.zeros((40, 40, 1))

            # load in data to the created array
            for c, gid, t in zip(charge, gridid, time):
                # sqmap is currently only defined for PMTs with ID > 8
                if gid > 8:
                    coorsq = sqmap[int(gid)]
                    # smallest charge is 0.1, so we set 0.1 and smaller to 0 to avoid log(0) issues
                    if log:
                        c = max(np.log(c + 1e-8) - np.log(0.1), 0.)
                    grid[coorsq[0], coorsq[1], 0] = c
                    if two_dims:
                        grid[coorsq[0], coorsq[1], 1] = t if not np.isclose(c, 0) else 500.

            total_data.append(grid)
            labels.append([zen, azi])
    total_data = np.array(total_data, dtype=np.float32)
    labels = np.array(labels)
    print('data shape', total_data.shape, 'labels shape', labels.shape)  # shape should be (N, 40, 40, 2)
    # We can normalize values to [0, 255] to put data in same domain as images, or [-1, 1]
    if normalize:
        min_vals, max_vals = [], []
        # normalize each channel independently
        for i in range(total_data.shape[3]):
            min_val, max_val = np.amin(total_data[:, :, :, i]), np.amax(total_data[:, :, :, i])
            min_vals.append(min_val), max_vals.append(max_val)
            print('Dimension %i, min: %f, max: %f' % (i, min_val, max_val))
        # Math to normalize channels to between [-1, 1]
        # Forward: -1. + 2. * (x - min) / (max - min)
        # inverse: (y + 1) / 2 * (max - min) + min
        dims = []
        if pixel_range:
            dims.append(255. * (total_data[:, :, :, 0] - min_vals[0]) / (max_vals[0] - min_vals[0]))
            if two_dims:
                dims.append(255. * (total_data[:, :, :, 1] - min_vals[1]) / (max_vals[1] - min_vals[1]))
        else:
            dims.append(-1. + 2. * (total_data[:, :, :, 0] - min_vals[0]) / (max_vals[0] - min_vals[0]))
            if two_dims:
                dims.append(-1. + 2. * (total_data[:, :, :, 1] - min_vals[1]) / (max_vals[1] - min_vals[1]))

        # sanity check for normalization
        for d in dims:
            assert np.amax(d) <= 255.01 if two_dims else 1.01, np.amax(d)
            assert np.amin(d) >= 0.0 if two_dims else -1.01, np.amin(d)
        total_data = np.stack(dims, axis=3)

    # shuffle data and fix seed
    np.random.seed(0)
    p = np.random.permutation(len(total_data))
    total_data, labels = total_data[p], labels[p]
    if display:
        plot.plot_hists(total_data)
        plot.plot_40x40(total_data[:16], 'ground truth - gamma - log - 40x40 grid')
        plot.plot_pmts(total_data[:16], 'ground truth - gamma - log - pmts', sparse=True, layout_path=save_path)
        for i in range(5, 8):
            plot.plot_pmts(total_data[:16], 'ground truth - gamma - log - pmts - single', sparse=True, single=i,
                           layout_path=save_path)

    # make the train / test split
    split = int(train_split * len(total_data))
    print("split size:", split)
    train_data, test_data, train_labels, test_labels = \
        total_data[:split], total_data[split:], labels[:split], labels[split:]

    # save data according to save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # create both training and testing data
    suffix = "_2" if two_dims else ""
    np.save(osp.join(save_path, "gamma_image_mapping_data" + suffix), train_data)
    np.save(osp.join(save_path, "gamma_labels" + suffix), train_labels)
    np.save(osp.join(save_path, "gamma_test_image_mapping_data" + suffix), test_data)
    np.save(osp.join(save_path, "gamma_test_labels" + suffix), test_labels)


# Gets the x, y, z coordinates of every PMT
def get_layout(path, save_path='data/', sub='gamma/'):
    """
    Get the x, y, and z coordinates of every PMT to visualize data later
    """
    files = glob(os.path.join(path, sub, '*.xcd'))
    pmt_locs = np.zeros((1200, 3))
    params = "event.hit.xPMT, event.hit.yPMT, event.hit.zPMT, event.hit.gridId"
    for xcdf_file in files:
        xf = XCDFFile(xcdf_file)
        print(xcdf_file)
        for xs, ys, zs, ids in list(xf.fields(params)):
            for x, y, z, id in zip(xs, ys, zs, ids):
                id -= 1  # change id to be 0 indexed
                # make sure all pmts with the same ID have the same coords
                if not np.isclose(pmt_locs[id][0], 0):
                    assert pmt_locs[id][0] == x
                if not np.isclose(pmt_locs[id][1], 0):
                    assert pmt_locs[id][1] == y
                if not np.isclose(pmt_locs[id][2], 0):
                    assert pmt_locs[id][2] == z
                # save coord locations
                pmt_locs[id][0] = x
                pmt_locs[id][1] = y
                pmt_locs[id][2] = z
        zeros = np.count_nonzero(pmt_locs)
        print("{0} out of {1} pixels have data".format(zeros, 1200 * 3))
        # assuming 3378 PMTs have value for early termination
        if zeros == 3378:
            break
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    np.save(osp.join(save_path, "layout"), pmt_locs)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--hawc-dir', help='location of HAWC dataset', default='./HAWC/')
    parser.add_argument('--save-dir', help='where to save processed data', default='./HAWC/data/')
    parser.add_argument('--gen', help='how to process the dataset', default='two-channel-map',
                        choices=['layout', 'one-channel-map', 'two-channel-map', 'one-dim'])
    parser.add_argument('--display', help='visualize processing of dataset', default=False, action='store_true')
    parser.add_argument('--small', help='use only one xcdf file instead of all', default=False, action='store_true')
    args = parser.parse_args()

    if not os.path.exists("./data/layout.npy"):
        print("generating mapping of grid IDs to x/y/z coordinates")
    if args.gen == 'layout':
        get_layout(path=args.hawc_dir, save_path=args.save_dir)
    elif args.gen == 'two-channel-map':
        gen_images_mapping(args.hawc_dir, display=args.display, two_dims=True, small=args.small, normalize=True,
                           pixel_range=True, save_path=args.save_dir)
    elif args.gen == 'one-channel-map':
        gen_images_mapping(args.hawc_dir, display=args.display, two_dims=False, small=args.small, normalize=True,
                           pixel_range=True, save_path=args.save_dir)
    elif args.gen == 'one-dim':
        gen_gamma_params(args.hawc_dir, save_path=args.save_dir)
    else:
        assert False
