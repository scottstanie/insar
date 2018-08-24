import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from insar import utils, sario, plotting
from insar.flood import stack

START_ROW = 18000
END_ROW = 18500
START_COL = 1000
END_COL = 1500

mlcpaths2 = sorted(glob.glob('/home/scott/uav-sar-data/trinity-mlc/*HHHH*.mlc'))
dates = ['2017/08/31', '2017/09/02', '2017/09/03']
print(mlcpaths2)
mlcpaths2 = mlcpaths2[0::2]
dates = dates[0::2]


def hide_axes(axis):
    axis.get_xaxis().set_visible(False)
    axis.get_yaxis().set_visible(False)


def date_diffs(dates):
    return ['%s to %s' % (dates[0], d) for d in dates[1:]]


def ratio_images(image_list, savename=None):
    fig, axes = plt.subplots(1, len(dates) - 1)
    axes = [axes] if len(dates) <= 2 else axes

    fig.tight_layout()
    fig.subplots_adjust(right=0.8)

    ratio_list = stack.make_uavsar_time_diffs(image_list)
    # Use same under image for all 3
    under = plotting.equalize_and_mask(
        sario.load(image_list[0])[START_ROW:END_ROW, START_COL:END_COL], fill_value=0.0)

    for idx in range(len(ratio_list)):
        ax = axes[idx]
        over = ratio_list[idx][START_ROW:END_ROW, START_COL:END_COL]

        # cmap = plotting.make_shifted_cmap(over, cmap_name='seismic')
        stack.overlay(under, over, ax=ax, show_colorbar=False)
        hide_axes(ax)
        ax.set_title(date_diffs(dates)[idx])

    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(axes[-1].get_images()[-1], cax=cbar_ax)

    if savename:
        plt.savefig(savename, bbox_inches='tight', dpi=300)
    else:
        plt.show(block=True)


if __name__ == '__main__':

    if len(sys.argv) < 2:
        print('Usage: %s (ratio | thresh) [savename]' % sys.argv[0])
        sys.exit(1)

    savename = None if len(sys.argv) < 3 else sys.argv[2]
    print('saving to %s' % savename if savename else 'Not saving')

    if sys.argv[1] == 'ratio':
        ratio_images(mlcpaths2, savename)
    elif sys.argv[1] == 'thresh':

        mlcs2 = [sario.load(p) for p in mlcpaths2]

        print([m.shape for m in mlcs2])
        blocks2 = np.stack((m[START_ROW:END_ROW, START_COL:END_COL] for m in mlcs2), axis=0)

        dates = dates[0:1] + dates[-2:]
        threshold = 0.02

        colors = [(1, 0, 0, c) for c in np.linspace(0, 1, 100)]
        cmapred = mcolors.LinearSegmentedColormap.from_list('mycmap', colors, N=5)

        fig, axes = plt.subplots(1, 3)
        fig.tight_layout()

        for ridx in range(len(dates)):
            # idx = 2*ridx
            block = blocks2[ridx, :500, :500]
            ax = axes.ravel()[ridx]
            im = ax.imshow(utils.db(block), cmap='gray')
            im = ax.imshow(block < threshold, cmap=cmapred, alpha=0.4)
            # ax.set_title('Water extent for ' + dates[ridx], fontsize=20)
            ax.set_title(dates[ridx], fontsize=20)
            hide_axes(ax)

        if savename:
            plt.savefig(
                savename,
                bbox_inches='tight',
                # transparent=True,
                # pad_inches=0,
                dpi=300,
            )
        else:
            plt.show(block=True)
