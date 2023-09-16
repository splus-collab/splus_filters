#!/bin/python3

import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import os
import argparse
import pandas as pd
from astropy.io import ascii, fits
import glob


def get_args():
    parser = argparse.ArgumentParser(description=" ".join([
        'Calculate the transmission curve or a given filtre.',
        'Estimate the central lambda from the FWHM of that filter.']))
    # parser.add_argument('--filter_file', type=str,
    #                     help='File containing the lab measures for the filtre.',
    #                     required=True)
    parser.add_argument('--color', type=str, help='Matplotlib colour to plot the filter.',
                        default='k')
    parser.add_argument('--name_of_filter', type=str, help='Name of the filter.',
                        default='filter1')
    parser.add_argument('--work_dir', type=str, help='Working directory.',
                        default=os.getcwd())
    parser.add_argument('--save_fig', action='store_true',
                        help='Save the plot of the filter.')
    parser.add_argument('--save_csv_filters', action='store_true',
                        help='Save the transmission curve of the filter.')
    parser.add_argument('--show_individual_filters', action='store_true',
                        help='Show the individual filters. Only activate when --save_csv_filters is used.')
    parser.add_argument('--save_lab_fig', action='store_true',
                        help='Save the plot of the lab filters.')

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
    plot_lab_curves(lab_filters, fnames2filters, 'lab_curves.png', args)

    allcurves = calc_trasm_curve(lab_filters, fnames2filters, args)
    plot_lab_curves(allcurves, fnames2filters, 'convoluted_curves.png', args)
    plot_all_curves(allcurves, args)
    # next
    # make_final_plot(allcurves, args)
    # calculate_central_lambda(allcurves, args)
    return allcurves


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


def plot_lab_curves(lab_filters, fnames2filters, outname, args):
    fig = plt.figure(figsize=(10, 10))
    for i, filter_name in enumerate(fnames2filters.keys()):
        ax = fig.add_subplot(4, 3, i+1)
        w = lab_filters[filter_name]['wave']
        t = lab_filters[filter_name]['transm']
        ax.plot(w, t, lw=1.5, label=fnames2filters[filter_name]['fname'],
                color=fnames2filters[filter_name]['color'])
        plt.legend()

    if args.save_lab_fig:
        plt.savefig(os.path.join(args.work_dir, outname), dpi=300)
    else:
        plt.show()


def calc_trasm_curve(lab_filters, fnames2filters, args):
    work_dir = args.work_dir
    data_dir = 'data-from-lab'
    allcurves = {}

    atm_transm_file = os.path.join(work_dir, data_dir, 'sky_trans.ascii')
    atmosph_transmitance = pd.read_csv(atm_transm_file, delimiter=' ')
    atm_wave = atmosph_transmitance['wave']
    atm_transm = atmosph_transmitance['transm']
    atm_ius = interp1d(atm_wave, atm_transm)
    allcurves['atm'] = {'wave': atm_wave, 'transm': atm_transm,
                        'fname': 'atm', 'color': 'k'}

    mirror_reflectance_file = os.path.join(
        work_dir, data_dir, 'mirror_reflectance.fits')
    mirror_reflect = fits.open(mirror_reflectance_file)[1].data
    mirror_wave = np.array([float(a) for a in mirror_reflect.col1])
    mirror_reflect = np.array([float(a) for a in mirror_reflect.col2]) / 100.
    # the reflectance bellow was obtained from:
    # https://laserbeamproducts.wordpress.com/2014/06/19/reflectivity-of-aluminium-uv-visible-and-infrared/
    mr_ius = interp1d(mirror_wave, mirror_reflect)
    allcurves['mirror'] = {'wave': mirror_wave, 'transm': mirror_reflect,
                           'fname': 'mirror', 'color': 'grey'}
    # measured
    mirror_measured_wave = np.array([300., 350., 420., 470., 530., 650., 880.,
                                     950., 1000., 1100])
    mirror_measured_flux = np.array([.9126, .9126, .9126, .9126,
                                     .911, .8725, .7971, .82, .84, .85])
    mr_meas = interp1d(mirror_measured_wave, mirror_measured_flux)
    allcurves['mirror_measured'] = {'wave': mirror_measured_wave,
                                    'transm': mirror_measured_flux,
                                    'fname': 'mirror_measured',
                                    'color': 'g'}
    mask = (mirror_wave > min(mirror_measured_wave)) & (
        mirror_wave < max(mirror_measured_wave))
    measur_interp = mr_meas(np.array(mirror_wave)[mask])

    ccd_efficiency_file = os.path.join(work_dir, data_dir, 'ccd_curve.fits')
    ccd_curve = fits.open(ccd_efficiency_file)[1].data
    ccd_wave = np.array([float(a) for a in ccd_curve.col1])
    ccd_eff = np.array([float(a) for a in ccd_curve.col2]) / 100.
    ccd_ius = interp1d(np.float_(ccd_wave), np.float_(ccd_eff))
    allcurves['ccd'] = {'wave': ccd_wave, 'transm': ccd_eff,
                        'fname': 'ccd', 'color': 'b'}
    # measured
    ccd_measured_wave = np.array([300., 350., 400., 450., 500., 550., 600.,
                                  650., 725., 800., 850., 900, 970.])
    ccd_measured_flux = np.array([.2, .45, .90, .93, .88, .88, .91, .92, .95,
                                  .88, .8, .6, .3])
    allcurves['ccd_measured'] = {'wave': ccd_measured_wave,
                                 'transm': ccd_measured_flux,
                                 'fname': 'ccd_measured', 'color': None}

    for lab_curve in lab_filters:
        lab_wave = lab_filters[lab_curve]['wave']
        lab_transm = lab_filters[lab_curve]['transm']
        lab_ius = interp1d(lab_wave, lab_transm)
        xmin = np.array([min(atm_wave / 10.), min(mirror_wave),
                         min(ccd_wave), min(lab_wave)])
        xmax = np.array([max(atm_wave / 10.), max(mirror_wave),
                         max(ccd_wave), max(lab_wave),
                         max(lab_filters['20150504C080zSDSS02']['wave'])])
        wave_range = np.arange(max(xmin), max(lab_wave), 1.)
        new_transm = lab_ius(wave_range)
        new_atm_transm = atm_ius(wave_range)
        new_mirror_reflect = mr_ius(wave_range)
        mirror_measured_wave = np.array(
            [300., 350., 420., 470., 530., 650., 88.])
        mirror_measured_flux = np.array(
            [.9126, .9126, .9126, .9126, .911, .8725, .7971])
        new_ccd_eff = ccd_ius(wave_range)
        new_filter_trans = (new_transm * new_atm_transm * new_mirror_reflect *
                            new_ccd_eff)

        if args.save_csv_filters:
            dat = wave_range, new_filter_trans
            outputname = "".join([fnames2filters[lab_curve]['fname'], '.csv'])
            print('Wrinting file: ', outputname)
            np.savetxt(outputname, np.transpose(dat), delimiter=',',
                       header='wavelength,transmittance')
            if args.show_individual_filters:
                plt.plot(wave_range, new_filter_trans, fnames2filters[lab_curve]['color'],
                         label=fnames2filters[lab_curve]['fname'])
                plt.legend()
                plt.show()
        allcurves[lab_curve] = {'wave': wave_range,
                                'transm': new_filter_trans,
                                'fname': fnames2filters[lab_curve]['fname'],
                                'color': fnames2filters[lab_curve]['color']}

    return allcurves


def plot_all_curves(allcurves, args):
    work_dir = args.work_dir
    # import pdb
    # pdb.set_trace()
    plt.figure(figsize=(10, 6))
    for curve in allcurves.keys():
        plt.plot(allcurves[curve]['wave'],
                 allcurves[curve]['transm'],
                 color=allcurves[curve]['color'],
                 label=allcurves[curve]['fname'])
    plt.xlim(300, 1100)
    plt.ylim(0, 1)
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Transmittance')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.tight_layout()
    plt.savefig(os.path.join(work_dir, 'allcurves.png'), dpi=300)
    plt.show()


def make_final_plot(allcurves, args):
    print('Making final plot')
    work_dir = args.work_dir
    # TODO: make pretty plot for publication


def calculate_central_lamda(allcurves, args):
    print('Calculating central wavelength')
    # TODO: calculate central wavelength via trapezoidal rule
    # TODO: calculate central wavelength via fwhm
    # TODO: calculate central wavelength via gaussian fit


if __name__ == '__main__':
    args = get_args()
    allcurves = main(args)


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
