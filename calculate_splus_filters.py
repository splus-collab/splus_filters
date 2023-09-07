#!/bin/python3

import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import sys
import os
import argparse
import pandas as pd


def get_args():
    parser = argparse.ArgumentParser(description=" ".join([
        'Calculate the transmission curve or a given filtre.',
        'Estimate the central lambda from the FWHM of that filtre.']))
    # parser.add_argument('--filter_file', type=str,
    #                     help='File containing the lab measures for the filtre.',
    #                     required=True)
    parser.add_argument('--color', type=str, help='Matplotlib colour to plot the filtre.',
                        default='k')
    parser.add_argument('--name_of_filter', type=str, help='Name of the filtre.',
                        default='filtre1')
    parser.add_argument('--work_dir', type=str, help='Working directory.',
                        default=os.getcwd())
    parser.add_argument('--save_fig', action='store_true',
                        help='Save the plot of the filtre.')

    # if len(sys.argv) == 1:
    #     parser.print_help()
    #     sys.exit(1)

    args = parser.parse_args()
    return args


def calc_trasm_curve(args):
    work_dir = args.work_dir
    list_of_filter_files = os.listdir(os.path.join(work_dir, 'data-from-lab'))
    for filter_file in list_of_filter_files:
        print(filter_file)
        filtre_filename = os.path.join(work_dir, 'data-from-lab', filter_file)
        df = pd.read_csv(filtre_filename, delimiter='\t',
                         decimal=',', skiprows=42)
        print(df.head())


if __name__ == '__main__':
    args = get_args()
    calc_trasm_curve(args)


# f = ascii.read(Filter)
# # f = fits.open(Filter)[1].data
#
# wave = f['col0']*10.
# flux = f['col1']
# # wave = f.wavelength * 10.
# # flux = f.transmit
#
# halfheight = flux.max() / 2.
#
# interp = interp1d(wave, flux)
# synwave = np.linspace(wave.min(), wave.max(), 100000)
# syncol = interp(synwave)
#
# cont = True
# inivalue = 0.0001
# while cont:
#     mask = (syncol > 0.01) & (syncol > (halfheight - inivalue)
#                               ) & (syncol < (halfheight + inivalue))
#     if mask.sum() >= 2:
#         minlam = synwave[mask][0]
#         maxlam = synwave[mask][-1]
#         # if (maxlam - minlam) < 90.:
#         #    minlam = np.mean(synwave[mask])
#         #    maxlam = max(synwave)
#         cont = False
#     else:
#         cont = True
#         inivalue += inivalue
#
#
# diff = maxlam - minlam
# leff = (maxlam + minlam) / 2.
#
# plt.plot(wave, flux, c=color, label='obs')
# plt.plot([minlam, maxlam], [halfheight, halfheight], c='k',
#          label=r'$\Delta\lambda = %.0f\mathrm{\AA}$' % diff)
# plt.scatter(minlam, halfheight, color='k', marker='d',
#             label=r'$\lambda_0 = %.0f$' % minlam)
# plt.scatter(maxlam, halfheight, color='k', marker='d',
#             label=r'$\lambda_1 = %.0f$' % maxlam)
# plt.scatter(leff, halfheight, marker='o', color='k',
#             label=r'$\lambda_\mathrm{eff} = %.0f\mathrm{\AA}$' % leff)
# plt.legend(loc='upper right')
# plt.grid()
#
# if save_fig:
#     plt.title(sys.argv[3])
#     print('safing fig', sys.argv[1].split('.')[0] + '.png')
#     plt.savefig(sys.argv[1].split('.')[0] + '.png', format='png', dpi=100)
#
# plt.show()
