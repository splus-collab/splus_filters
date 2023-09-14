#!/bin/python3

import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import sys
import os
import argparse
import pandas as pd
from astropy.io import ascii, fits
import glob


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


def main(args):

    fnames2filters = {
        '20150918C080uJAVA02': {'fname': 'uJAVA', 'color': 'b'},
        '20150429C080F037802': {'fname': 'J0378', 'color': 'royalblue'},
        '20150922C080F039502': {'fname': 'J0395', 'color': 'skyblue'},
        '20150923C080F041002': {'fname': 'J0410', 'color': 'forestgreen'},
        '20150514C080F043002': {'fname': 'J0430', 'color': 'cyan'},
        '20150924C080gSDSS02': {'fname': 'gDSSS', 'color': 'g'},
        '20140606C080F051502': {'fname': 'J0515', 'color': 'limegreen'},
        '20140604C080F062502': {'fname': 'rSDSS', 'color': 'r'},
        '20140609C080F066002': {'fname': 'J0660', 'color': 'magenta'},
        '20150506C080iSDSS02': {'fname': 'iSDSS', 'color': 'violet'},
        '20150922C080F086102': {'fname': 'J0861', 'color': 'brown'},
        '20150504C080zSDSS02': {'fname': 'zSDSS', 'color': 'purple'}}

    lab_filters = get_lab_curves(args)
    plot_lab_curves(lab_filters, fnames2filters)

    calc_trasm_curve(args)


def get_lab_curves(args):
    work_dir = args.work_dir
    data_dir = 'data-from-lab'
    list_of_filter_files = glob.glob(os.path.join(work_dir, data_dir, '*.txt'))

    lab_filters = {}
    for filter_file in list_of_filter_files:
        # print(filter_file)
        n = 1
        with open(filter_file, 'r') as f:
            for line in f.readlines():
                if line.startswith('Wavelength nm'):
                    break
                if line.startswith('J-PLUS #07'):
                    n += 2
                    break
                n += 1
            f.close()
        # filter_filename = os.path.join(work_dir, data_dir, filter_file)
        print(n)
        df = pd.read_csv(filter_file, delimiter='\t',
                         decimal=',', skiprows=n, header=None)
        # print(df.head())
        mid_columns_average = df[df.columns[2:-1]].mean(axis=1)
        transmission_mean = mid_columns_average / df[[1, 102]].mean(axis=1)
        wave = df[0]
        # print(list(zip((wave, transmission_mean))))

        lab_filters[filter_file.split('/')[-1].split('.')[0]] = \
            {'wave': wave, 'transm': transmission_mean}

    return lab_filters


def plot_lab_curves(lab_filters, fnames2filters):
    fig = plt.figure(figsize=(10, 10))
    for i, filter_name in enumerate(fnames2filters.keys()):
        ax = fig.add_subplot(4, 3, i+1)
        w = lab_filters[filter_name]['wave']
        t = lab_filters[filter_name]['transm']
        ax.plot(w, t, lw=1.5, label=fnames2filters[filter_name]['fname'],
                color=fnames2filters[filter_name]['color'])
        plt.legend()

    plt.show()


def calc_trasm_curve(args):
    work_dir = args.work_dir
    data_dir = 'data-from-lab'
    list_of_filter_files = os.listdir(os.path.join(work_dir, data_dir))

    atm_transm_file = os.path.join(work_dir, data_dir, 'sky_trans.ascii')
    atmosph_transmitance = pd.read_csv(atm_transm_file, delimiter=' ')
    atm_wave = atmosph_transmitance['wave']
    atm_transm = atmosph_transmitance['transm']
    atm_ius = interp1d(atm_wave, atm_transm)

    mirror_reflectance_file = os.path.join(
        work_dir, data_dir, 'mirror_reflectance.fits')
    mirror_reflect = fits.open(mirror_reflectance_file)[1].data
    mirror_wave = np.array([float(a) for a in mirror_reflect.col1])
    mirror_reflect = np.array([float(a) for a in mirror_reflect.col2])
    # the reflectance bellow was obtained from:
    # https://laserbeamproducts.wordpress.com/2014/06/19/reflectivity-of-aluminium-uv-visible-and-infrared/
    mr_ius = interp1d(mirror_wave, mirror_reflect)
    # measured
    mirror_measured_wave = np.array([300., 350., 420., 470., 530., 650., 880.,
                                     950., 1000., 1100])
    mirror_measured_flux = np.array([.9126, .9126, .9126, .9126,
                                     .911, .8725, .7971, .82, .84, .85])
    mr_meas = interp1d(mirror_measured_wave, mirror_measured_flux)
    mask = (mirror_wave > min(mirror_measured_wave)) & (
        mirror_wave < max(mirror_measured_wave))
    measur_interp = mr_meas(np.array(mirror_wave)[mask])

    ccd_efficiency_file = os.path.join(work_dir, data_dir, 'ccd_curve.fits')
    ccd_curve = fits.open(ccd_efficiency_file)[1].data
    ccd_wave = np.array([float(a) for a in ccd_curve.col1])
    ccd_eff = np.array([float(a) for a in ccd_curve.col2])
    ccd_ius = interp1d(np.float_(ccd_wave), np.float_(ccd_eff))
    # measured
    ccd_measured_wave = np.array([300., 350., 400., 450., 500., 550., 600.,
                                  650., 725., 800., 850., 900, 970.])
    ccd_measured_flux = np.array([.2, .45, .90, .93, .88, .88, .91, .92, .95,
                                  .88, .8, .6, .3])


if __name__ == '__main__':
    args = get_args()
    main(args)


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
