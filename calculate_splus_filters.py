#!/bin/python3

import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import os
import argparse
import pandas as pd
from astropy.io import fits
import glob
import colorlog
import logging
from scipy.optimize import curve_fit


# define logger with different colours for different levels
def get_logger(name, loglevel='INFO'):
    logger = logging.getLogger(name)
    logger.setLevel(loglevel)
    handler = logging.StreamHandler()
    handler.setFormatter(colorlog.ColoredFormatter(
        '%(log_color)s %(levelname)s %(reset)s %(asctime)s - %(message)s',
        log_colors={
            'DEBUG': 'cyan',
            'INFO': 'green',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'white,bg_red',
        },
    ))
    logger.addHandler(handler)
    return logger


def get_args():
    parser = argparse.ArgumentParser(description=" ".join([
        'Calculate the transmission curve or a given filtre.',
        'Estimate the central lambda from the FWHM of that filter.']))
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
    parser.add_argument('--loglevel', type=str, help='Log level.',
                        default='INFO')

    # if len(sys.argv) == 1:
    #     parser.print_help()
    #     sys.exit(1)

    args = parser.parse_args()
    return args


def main(args):

    fnames2filters = {
        '20150918C080uJAVA02': {'fname': 'uJAVA', 'color': 'navy', 'pos': (3563, -500, -4)},
        '20150429C080F037802': {'fname': 'J0378', 'color': 'darkviolet', 'pos': (3770, -350, 1)},
        '20150922C080F039502': {'fname': 'J0395', 'color': 'b', 'pos': (3940, -400, 1)},
        '20150923C080F041002': {'fname': 'J0410', 'color': 'dodgerblue', 'pos': (4094, -350, -0.1)},
        '20150514C080F043002': {'fname': 'J0430', 'color': 'c', 'pos': (4292, -400, 1.3)},
        '20150924C080gSDSS02': {'fname': 'gDSSS', 'color': 'turquoise', 'pos': (4751, -300, 0.)},
        '20140606C080F051502': {'fname': 'J0515', 'color': 'lime', 'pos': (5133, -100, 1.5)},
        '20140604C080F062502': {'fname': 'rSDSS', 'color': 'greenyellow', 'pos': (6258, -500, -2)},
        '20140609C080F066002': {'fname': 'J0660', 'color': 'orange', 'pos': (6614, -300, 1)},
        '20150506C080iSDSS02': {'fname': 'iSDSS', 'color': 'darkorange', 'pos': (7690, -300, 1)},
        '20150922C080F086102': {'fname': 'J0861', 'color': 'orangered', 'pos': (8611, -250, 1)},
        '20150504C080zSDSS02': {'fname': 'zSDSS', 'color': 'r', 'pos': (8831, 300, -12)}}

    logger = get_logger('calculate_splus_filters', loglevel=args.loglevel)
    logger.info('Calculating the lab transmission curves of the filters.')
    lab_filters = get_lab_curves(args)
    plot_lab_curves(lab_filters, fnames2filters, 'lab_curves.png', args)

    allcurves = calc_trasm_curve(lab_filters, fnames2filters, args)
    plot_lab_curves(allcurves, fnames2filters, 'convoluted_curves.png', args)
    plot_all_curves(allcurves, args)
    make_final_plot(allcurves, fnames2filters, args)
    calculate_central_lambda(allcurves, fnames2filters, args)
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
        print(n)
        df = pd.read_csv(filter_file, delimiter='\t',
                         decimal=',', skiprows=n, header=None)
        mid_columns_average = df[df.columns[2:-1]].mean(axis=1)
        transmission_mean = mid_columns_average / df[[1, 102]].mean(axis=1)
        wave = df[0]

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
    # mirror_measured_wave = np.array([300., 350., 420., 470., 530., 650., 880.,
    #                                  950., 1000., 1100])
    # mirror_measured_flux = np.array([.9126, .9126, .9126, .9126,
    #                                  .911, .8725, .7971, .82, .84, .85])
    # mr_meas = interp1d(mirror_measured_wave, mirror_measured_flux)
    # allcurves['mirror_measured'] = {'wave': mirror_measured_wave,
    #                                 'transm': mirror_measured_flux,
    #                                 'fname': 'mirror_measured',
    #                                 'color': 'g'}
    # mask = (mirror_wave > min(mirror_measured_wave)) & (
    #     mirror_wave < max(mirror_measured_wave))
    # measur_interp = mr_meas(np.array(mirror_wave)[mask])

    ccd_efficiency_file = os.path.join(work_dir, data_dir, 'ccd_curve.fits')
    ccd_curve = fits.open(ccd_efficiency_file)[1].data
    ccd_wave = np.array([float(a) for a in ccd_curve.col1])
    ccd_eff = np.array([float(a) for a in ccd_curve.col2]) / 100.
    ccd_ius = interp1d(np.float_(ccd_wave), np.float_(ccd_eff))
    allcurves['ccd'] = {'wave': ccd_wave, 'transm': ccd_eff,
                        'fname': 'ccd', 'color': 'b'}
    # measured CCD efficiency on Tololo
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
        # xmax = np.array([max(atm_wave / 10.), max(mirror_wave),
        #                  max(ccd_wave), max(lab_wave),
        #                  max(lab_filters['20150504C080zSDSS02']['wave'])])
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


def make_final_plot(allcurves, fnames2filters, args):
    print('Making final plot')
    work_dir = args.work_dir
    plt.figure(figsize=(8, 4))
    for key in fnames2filters.keys():
        plt.plot(allcurves[key]['wave'] * 10.,
                 allcurves[key]['transm'] * 100.,
                 color=fnames2filters[key]['color'],
                 label=fnames2filters[key]['fname'], lw=2)
        plt.text(fnames2filters[key]['pos'][0] + fnames2filters[key]['pos'][1],
                 max(allcurves[key]['transm'] * 100.) +
                 fnames2filters[key]['pos'][2],
                 fnames2filters[key]['fname'], fontsize=12, color=fnames2filters[key]['color'])

    plt.xlabel(r'$\lambda\ \mathrm{[\AA]}$', fontsize=16)
    plt.ylabel(r'$R_\lambda\ \mathrm{[\%]}$', fontsize=16)
    plt.xlim(3000, 10000)
    plt.ylim(0.2, 83)
    plt.tight_layout()

    plt.savefig(os.path.join(work_dir, 'splus_filters.png'),
                format='png', dpi=300)
    plt.show(block=False)


def calculate_central_lambda(allcurves, fnames2filters, args):
    logger = get_logger(__name__, loglevel=args.loglevel)
    logger.info('Calculating central wavelength')
    logger.info('Claculating curves via trapezoidal rule approach')
    print('Filter, central wavelength, FWHM')
    for curve in fnames2filters.keys():
        wave = allcurves[curve]['wave'] * 10.
        transm = allcurves[curve]['transm']
        interp = interp1d(wave, transm)
        synt_wave = np.linspace(min(wave), max(wave), 1000000)
        synt_transm = interp(synt_wave)
        half_height = max(synt_transm) / 2.
        left_index = np.where(synt_transm > half_height)[0][0]
        right_index = np.where(synt_transm > half_height)[0][-1]
        min_wave = synt_wave[left_index]
        max_wave = synt_wave[right_index]
        central_wave = (max_wave + min_wave) / 2.
        print('Trapz: %s %.0f %.0f' % (fnames2filters[curve]['fname'],
              central_wave, max_wave - min_wave))

        norm_transm = synt_transm / max(synt_transm)
        left_idx = np.where(norm_transm > 0.5)[0][0]
        right_idx = np.where(norm_transm > 0.5)[0][-1]
        mid_wave = (synt_wave[right_idx] + synt_wave[left_idx]) / 2.
        delta_wave = synt_wave[right_idx] - synt_wave[left_idx]
        print('Norm: %s %.0f %.0f' % (fnames2filters[curve]['fname'],
              mid_wave, delta_wave))


if __name__ == '__main__':
    args = get_args()
    main(args)
