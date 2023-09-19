#!/bin/python3

import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import os
import sys
import argparse
import pandas as pd
from astropy.io import fits
import glob
import colorlog
import logging


def get_logger(name, loglevel='INFO'):
    logger = logging.getLogger(name)
    logger.setLevel(loglevel)
    handler = logging.StreamHandler()
    handler.setFormatter(colorlog.ColoredFormatter(
        "%(asctime)s [%(log_color)s%(levelname)s%(reset)s] @%(module)s.%(funcName)s() %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        log_colors={
            'DEBUG': 'cyan',
            'INFO': 'green',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'bold_yellow,bg_red',
        },
    ))
    logger.addHandler(handler)
    return logger


def get_args():
    parser = argparse.ArgumentParser(description=" ".join([
        'Calculate the transmission curve or a given filtre.',
        'Estimate the central lambda from the FWHM of that filter.']))
    parser.add_argument('--work_dir', type=str, help='Working directory.',
                        default=os.getcwd())
    parser.add_argument('--save_fig', action='store_true',
                        help='Save the plot of the filter.')
    parser.add_argument('--save_csv_filters', action='store_true',
                        help='Save the transmission curve of the filter.')
    parser.add_argument('--show_individual_filters', action='store_true',
                        help='Show the individual filters. Only activate when --save_csv_filters is used.')
    parser.add_argument('--show_plots', action='store_true',
                        help='Show the main plots.')
    parser.add_argument('--loglevel', type=str, help='Log level.',
                        default='INFO')
    parser.add_argument('--debig', action='store_true',
                        help='Activate debug mode.')

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

    logger = get_logger(__name__, loglevel=args.loglevel)
    logger.info('Calculating the lab transmission curves of the filters.')
    lab_filters = get_lab_curves(args)
    plot_lab_curves(lab_filters, fnames2filters, args,
                    output='lab_curves.png', figlevel='lab')

    allcurves = calc_trasm_curve(lab_filters, fnames2filters, args)
    plot_lab_curves(allcurves, fnames2filters, 'convoluted_curves.png', args)
    plot_all_curves(allcurves, args)
    make_final_plot(allcurves, fnames2filters, args)
    allcurves = calculate_central_lambda(allcurves, fnames2filters, args)
    plot_lab_curves(allcurves, fnames2filters,
                    'convoluted_curves_centralambda.png', args)
    make_html(allcurves, fnames2filters, args)
    return allcurves


def get_lab_curves(args):
    logger = get_logger(__name__, loglevel=args.loglevel)
    work_dir = args.work_dir
    data_dir = 'data-from-lab'
    list_of_filter_files = glob.glob(os.path.join(work_dir, data_dir, '*.txt'))

    lab_filters = {}
    for filter_file in list_of_filter_files:
        logger.debug('Reading {}'.format(filter_file))
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
        logger.debug('Skipping {} lines'.format(n))
        try:
            df = pd.read_csv(filter_file, delimiter='\t',
                             decimal=',', skiprows=n, header=None)
        except Exception as e:
            logger.error(" ".join(['Reading the filter file failed.',
                         'Error: {}'.format(e)]))
            sys.exit(1)
        logger.info(" ".join(['The transmission level is calculated from the',
                    'average of the measures in the lab file.']))
        mid_columns_average = df[df.columns[2:-1]].mean(axis=1)
        transmission_mean = mid_columns_average / df[[1, 102]].mean(axis=1)
        wave = df[0]

        logger.debug('Test if filters were read correctly.')
        if wave.size != transmission_mean.size:
            logger.error(" ".join(['Reading the filter file failed.',
                         'Size of wave and transmission differ.']))
            sys.exit(1)
        if wave.size < 100:
            logger.error(" ".join(['Reading the filter file failed.',
                         'Size of wave is too small.']))
            sys.exit(1)
        if (wave is None) or (transmission_mean is None):
            logger.error(" ".join(['Reading the filter file failed.',
                         'Wave or transmission is None.']))
            sys.exit(1)
        logger.debug('Loding filters to memory.')
        lab_filters[filter_file.split('/')[-1].split('.')[0]] = \
            {'wave': wave, 'transm': transmission_mean}

    return lab_filters


def plot_lab_curves(lab_filters, fnames2filters, args, outname='fig.png', figlevel='lab'):
    logger = get_logger(__name__, loglevel=args.loglevel)
    fig = plt.figure(figsize=(10, 10))
    for i, filter_name in enumerate(fnames2filters.keys()):
        ax = fig.add_subplot(4, 3, i+1)
        w = lab_filters[filter_name]['wave'] * 10.
        t = lab_filters[filter_name]['transm']
        logger.debug('Plotting filter: %s' % filter_name)
        ax.plot(w, t, lw=1.5, label=fnames2filters[filter_name]['fname'],
                color=fnames2filters[filter_name]['color'])
        if figlevel == 'lab':
            logger.info('Plotting lab curve %s' % filter_name)
        elif figlevel == 'trapz':
            central_wave = lab_filters[filter_name]['trapz']['central_wave']
            min_wave = lab_filters[filter_name]['trapz']['central_wave'] - \
                lab_filters[filter_name]['trapz']['delta_wave'] / 2.
            max_wave = lab_filters[filter_name]['trapz']['central_wave'] + \
                lab_filters[filter_name]['trapz']['delta_wave'] / 2.
            half_height = lab_filters[filter_name]['transm'].max() / 2.
            logger.debug('Plotting min and max wavelengths: %f, %f' %
                         (min_wave, max_wave))
            ax.plot([min_wave, max_wave], [half_height, half_height],
                    'k-', marker='p', lw=1.5)
            logger.debug('Plotting central wavelength: %f' % central_wave)
            ax.plot([central_wave], [half_height],
                    marker='o', color='c', lw=1.5)
            logger.debug('Plotting vertical lines.')
            ax.plot([min_wave, min_wave], [0, half_height], 'k--', lw=1.)
            ax.plot([max_wave, max_wave], [0, half_height], 'k--', lw=1.)
            ax.plot([central_wave, central_wave],
                    [0, lab_filters[filter_name]['transm'].max()],
                    'k--', lw=1.)
        elif figlevel == 'eq_width':
            logger.error('Not implemented yet.')
        plt.legend()

    if args.save_lab_fig:
        logger.info('Saving fig to %s' % os.path.join(args.work_dir, outname))
        plt.savefig(os.path.join(args.work_dir, outname), dpi=300)
    if args.show_plots:
        logger.debug('Showing plot.')
        plt.show()
    else:
        logger.debug('Closing plot.')
        plt.close()


def calc_trasm_curve(lab_filters, fnames2filters, args):
    logger = get_logger(__name__, loglevel=args.loglevel)
    logger.info('Calculating transmission curves.')
    work_dir = args.work_dir
    data_dir = 'data-from-lab'
    allcurves = {}

    logger.debug('Calculating atmospheric transmission.')
    atm_transm_file = os.path.join(work_dir, data_dir, 'sky_trans.ascii')
    atmosph_transmitance = pd.read_csv(atm_transm_file, delimiter=' ')
    atm_wave = atmosph_transmitance['wave']
    atm_transm = atmosph_transmitance['transm']
    atm_ius = interp1d(atm_wave, atm_transm)
    allcurves['atm'] = {'wave': atm_wave, 'transm': atm_transm,
                        'fname': 'atm', 'color': 'k'}

    logger.debug('Calculating mirror reflectance.')
    mirror_reflectance_file = os.path.join(
        work_dir, data_dir, 'mirror_reflectance.fits')
    mirror_reflect = fits.open(mirror_reflectance_file)[1].data
    mirror_wave = np.array([float(a) for a in mirror_reflect.col1])
    mirror_reflect = np.array([float(a) for a in mirror_reflect.col2]) / 100.
    mr_ius = interp1d(mirror_wave, mirror_reflect)
    allcurves['mirror'] = {'wave': mirror_wave, 'transm': mirror_reflect,
                           'fname': 'mirror', 'color': 'grey'}
    if args.debug:
        # the reflectance bellow was obtained from:
        # https://laserbeamproducts.wordpress.com/2014/06/19/reflectivity-of-aluminium-uv-visible-and-infrared/
        logger.debug(" ".join(['The following values are not measured.',
                               'They were taken as a reference to check',
                               'the extrapolation of the measured values.',
                               'This was necessary because the measured',
                               'values do not cover the whole wavelength',
                               'range to the red. After debating on using',
                               'a linear extrapolation or the values bellow,',
                               'we reached that the conclusion that any of',
                               'the methods would involve fabricating data.',
                               'We decided to use the extrapolation because',
                               'it gives as good as invented values as any',
                               'other similar method. The values bellow',
                               'are kept here are reference only.']))

        mirror_measured_wave = np.array([300., 350., 420., 470., 530., 650., 880.,
                                         950., 1000., 1100])
        mirror_measured_flux = np.array([.9126, .9126, .9126, .9126,
                                         .911, .8725, .7971, .82, .84, .85])
        allcurves['aluminum_reflect'] = {'wave': mirror_measured_wave,
                                         'transm': mirror_measured_flux,
                                         'fname': 'mirror_measured',
                                         'color': 'g'}

    logger.debug('Calculating CCD efficiency from lab.')
    ccd_efficiency_file = os.path.join(work_dir, data_dir, 'ccd_curve.fits')
    ccd_curve = fits.open(ccd_efficiency_file)[1].data
    ccd_wave = np.array([float(a) for a in ccd_curve.col1])
    ccd_eff = np.array([float(a) for a in ccd_curve.col2]) / 100.
    ccd_ius = interp1d(np.float_(ccd_wave), np.float_(ccd_eff))
    allcurves['ccd'] = {'wave': ccd_wave, 'transm': ccd_eff,
                        'fname': 'ccd', 'color': 'b'}
    logger.debug('Calculating CCD efficiency from Tololo.')
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
            logger.info('Wrinting file: %s.', outputname)
            np.savetxt(outputname, np.transpose(dat), delimiter=',',
                       header='wavelength,transmittance')
            if args.show_individual_filters:
                plt.plot(wave_range, new_filter_trans, fnames2filters[lab_curve]['color'],
                         label=fnames2filters[lab_curve]['fname'])
                plt.legend()
                plt.show()
            else:
                plt.close()
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
    if args.show_plots:
        plt.show()
    else:
        plt.close()


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
    if args.show_plots:
        plt.show(block=False)
    else:
        plt.close()


def calculate_central_lambda(allcurves, fnames2filters, args):
    logger = get_logger(__name__, loglevel=args.loglevel)
    logger.info('Calculating central wavelength')
    logger.info('Claculating curves via trapezoidal rule approach')
    print('Filter, central wavelength, FWHM')

    for curve in fnames2filters.keys():
        wave = allcurves[curve]['wave'] * 10.
        transm = allcurves[curve]['transm']
        interp = interp1d(wave, transm)
        synt_wave = np.linspace(min(wave), max(wave), 100000)
        synt_transm = interp(synt_wave)
        half_height = max(synt_transm) / 2.
        left_index = np.where(synt_transm > half_height)[0][0]
        right_index = np.where(synt_transm > half_height)[0][-1]
        min_wave = synt_wave[left_index]
        max_wave = synt_wave[right_index]
        central_wave = (max_wave + min_wave) / 2.
        allcurves[curve]['trapz'] = {'central_wave': central_wave,
                                     'delta_wave': max_wave - min_wave}
        print('Trapz: %s %.0f %.0f' % (fnames2filters[curve]['fname'],
              central_wave, max_wave - min_wave))

        norm_transm = synt_transm / max(synt_transm)
        left_idx = np.where(norm_transm > 0.5)[0][0]
        right_idx = np.where(norm_transm > 0.5)[0][-1]
        mid_wave = (synt_wave[right_idx] + synt_wave[left_idx]) / 2.
        delta_wave = synt_wave[right_idx] - synt_wave[left_idx]
        equi_width = np.trapz(synt_transm, synt_wave) / max(synt_transm)
        allcurves[curve]['norm'] = {'central_wave': mid_wave,
                                    'delta_wave': equi_width}
        print('Norm: %s %.0f %.0f' % (fnames2filters[curve]['fname'],
              mid_wave, delta_wave))
        lambda_mean = np.sum(wave * transm) / np.sum(transm)
        std_mean = np.sqrt(np.sum((wave - lambda_mean)**2) / wave.size)
        allcurves[curve]['mean'] = {'central_wave': lambda_mean,
                                    'delta_wave': std_mean}
        print('Mean: %s %.0f Wmean: %.0f' % (fnames2filters[curve]['fname'],
                                             lambda_mean, std_mean))

    return allcurves


def make_html(allcurves, fnames2filters, args):
    logger = get_logger(__name__, loglevel=args.loglevel)
    htmlf = open(os.path.join(args.work_dir, 'central_wavelengths.html'), 'w')
    htmlf.write('<div class="dlpage">\n')
    htmlf.write('<table class="docutils" style="width:100%" border=1>\n')
    htmlf.write('<colgroup>\n')
    htmlf.write('<tr>')
    htmlf.write('<th colspan="4"><b>S-PLUS filters summary</b></th>\n')
    htmlf.write('</tr>\n')
    htmlf.write('<tr>')
    htmlf.write('<td>Filter</td>\n')
    htmlf.write('<td><sub>λ</sub><sub>trapz</sub></td>\n')
    htmlf.write('<td><sub>Δλ</sub><sub>trapz</sub></td>\n')
    htmlf.write('<td><sub>λ</sub><sub>norm</sub></td>\n')
    htmlf.write('<td><sub>W</sub><sub>eq</sub></td>\n')
    htmlf.write('<td><sub>λ</sub><sub>mean</sub></td>\n')
    htmlf.write('<td><sub>Δλ</sub><sub>mean</sub></td>\n')
    htmlf.write('<td><sub>λ</sub><sub>eff</sub></td>\n')
    htmlf.write('<td><sub>Δλ</sub><sub>eff</sub></td>\n')
    htmlf.write('</tr>\n')
    htmlf.write('</colgroup>\n')
    for curve in fnames2filters.keys():
        logger.info('Writing central wavelengths to html file')
        htmlf.write('<tr>\n')
        htmlf.write('<td>%s</td>\n' % fnames2filters[curve]['fname'])
        htmlf.write('<td>%.0f</td>\n' %
                    allcurves[curve]['trapz']['central_wave'])
        htmlf.write('<td>%.0f</td>\n' %
                    allcurves[curve]['trapz']['delta_wave'])
        htmlf.write('<td>%.0f</td>\n' %
                    allcurves[curve]['norm']['central_wave'])
        htmlf.write('<td>%.0f</td>\n' % allcurves[curve]['norm']['delta_wave'])
        htmlf.write('<td>%.0f</td>\n' %
                    allcurves[curve]['mean']['central_wave'])
        htmlf.write('<td>%.0f</td>\n' % allcurves[curve]['mean']['delta_wave'])
        # htmlf.write('<td>%.0f</td>\n' % lambda_eff)
        # htmlf.write('<td>%.0f</td>\n' % effective_width)
        htmlf.write('</tr>\n')
    htmlf.write('</table>\n')
    htmlf.write('</div>\n')
    htmlf.close()


if __name__ == '__main__':
    args = get_args()
    main(args)
