import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors
import numpy as np
import os.path as osp

labels = {0: "theta",
          1: "azmiuth",
          2: "nPE",
          3: "nHits"}


def plot_ground_truth(gamma_path):
    """ plot the features extracted from gamma dataset """
    samples = np.load(gamma_path)
    # normalization values (mean and std)
    samples -= [4.32868898e-01, -1.03701055e-02, 2.64389896e+00, 1.58296906e+02]
    samples /= [0.294857, 1.7961102, 0.5629358, 178.98788]
    plot_4(samples)


def plot_hists(grid):
    """ plot distribution of pixel values in 40x40 mapped image """
    plt.figure(figsize=(16, 16))
    plt.title('distribution of values per dimension')
    for i in range(grid.shape[3]):
        plt.hist(grid[:, :, :, i].flatten(), bins=50, alpha=0.5, label='dim %i' % i)
    plt.legend()
    plt.show()


def plot_tanks_from_grid(save_path="HAWC/saves", num=2, dim=1, layout_path="HAWC/data"):
    """
    plot a variety of visualizations from pixelcnn 40x40 image output
    if 1 dim: raw 40x40 grid, grid of pmts, single pmts
    if 2 dim: 40x40 grid side by side, grid of pmts side by side
    """

    prefix = "2" if dim == 2 else ""
    grid = osp.join(save_path, "hawc" + prefix + "_sample" + str(num) + ".npz")
    grid = np.load(grid)['arr_0']
    plot_40x40((grid + 1) * 127.5, 'pixelcnn pmt hits - 40x40 grid, log(charge)')
    plot_pmts(grid, 'pixelcnn pmt hits - log(charge)', layout_path=layout_path)
    for i in range(5, 16):
        plot_pmts(grid, 'pixelcnn pmt hits - log(charge) - single', single=i, layout_path=layout_path)


def plot_40x40(grid, title, frame):
    """ plot a simple grid of 40x40 images """
    fig = plt.figure(figsize=(20, 20))
    for i in range(16):
        plt.subplot(4, 4, i + 1)
        if grid.shape[3] == 1:
            plt.imshow(grid[i][:, :, 0])
        elif grid.shape[3] == 2:
            plt.title('event %i, dim %i' % (i // 2, i % 2))
            if i % 2 == 0:
                grid[i // 2][:, :, i % 2] = np.clip(grid[i // 2][:, :, i % 2], 0, 4000)
            plt.imshow(grid[i // 2][:, :, i % 2])
        plt.colorbar()
    fig.suptitle(title)
    # plt.show()
    

def plot_pmts(grid, title, single=None, sparse=False, layout_path="data/"):
    """
    plot a scatterplot from 40x40 mapped images corresponding to real locations of pmts
    plot a 2 by 2 plot, or a single plot if single=True
    does some additional data cleaning if data is not sparse
    """
    layout = np.load(osp.join(layout_path, "layout.npy"))
    from squaremapping import sqmap
    inv_sqmap = dict((v, k) for (k, v) in sqmap.iteritems())
    good_cords = layout[:, 0] != 0.0
    x_cord = layout[:, 0]
    y_cord = layout[:, 1]
    if not single:
        fig = plt.figure(figsize=(16, 16))
        fig.suptitle(title)
        for i in range(4):
            A = np.zeros((1200, 1))
            for x, y in inv_sqmap.keys():
                pmt_num = inv_sqmap[(x, y)]
                if grid.shape[3] == 1:
                    A[pmt_num - 1] = grid[i][x, y, 0]
                elif grid.shape[3] == 2:
                    A[pmt_num - 1] = grid[i // 2][x, y, i % 2]
            if not sparse:
                # remove values of 0 for a cleaner plot
                good_cords = np.minimum(A[:, 0] != 0, good_cords)
            A += 1
            plt.subplot(2, 2, i + 1)
            plt.scatter(x_cord[good_cords], y_cord[good_cords], s=50, c=A[:, 0][good_cords] + 0.1, marker='o',
                        linewidths=0.01, alpha=0.7, norm=matplotlib.colors.LogNorm())
            plt.colorbar()
        plt.show()
    else:
        i = single
        fig = plt.figure(figsize=(16, 16))
        plt.title('pixelcnn pmt hits - log(charge) - single')
        A = np.zeros((1200, 1))
        for x, y in inv_sqmap.keys():
            pmt_num = inv_sqmap[(x, y)]
            A[pmt_num - 1] = grid[i][x, y, 0] + 1

        plt.scatter(x_cord[good_cords], y_cord[good_cords], s=50, c=A[:, 0][good_cords] + 0.1, marker='o',
                    linewidths=0.01, alpha=0.7, norm=matplotlib.colors.LogNorm())
        plt.colorbar()
        plt.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--num', help='which epoch of pixelcnn to display', type=int, default=0)
    parser.add_argument('--chs', help='number of channels of data', type=int, default=2, choices=[1, 2])
    parser.add_argument('--data-path', help='path to data (layout)', default='data/')
    parser.add_argument('--save-path', help='path to saves', default='HAWC/saves/')
    args = parser.parse_args()

    plot_tanks_from_grid(num=args.num, dim=args.chs, save_path=args.save_path, layout_path=args.data_path)
