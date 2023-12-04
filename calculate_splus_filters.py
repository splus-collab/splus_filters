#!/usr/bin/env python3

import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import os
import sys
import argparse
import pandas as pd
from astropy.io import fits, ascii
import glob
import colorlog
import logging


def get_logger(name, loglevel='INFO'):
    """Return a logger with a default ColoredFormatter."""
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
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description=" ".join([
        'Calculate the transmission curve or a given filtre.',
        'Estimate the central lambda from the FWHM of that filter.']))
    parser.add_argument('--work_dir', type=str, help='Working directory. Default: current directory.',
                        default=os.getcwd())
    parser.add_argument('--save_plots', action='store_true',
                        help='Save the plot of the filter.')
    parser.add_argument('--save_csv_filters', action='store_true',
                        help='Save the transmission curve of the filter.')
    parser.add_argument('--save_central_wavelentghs', action='store_true',
                        help='Save the central wavelengths of the filters in a csv file.')
    parser.add_argument('--show_individual_filters', action='store_true',
                        help='Show the individual filters. Only activate when --save_csv_filters is used.')
    parser.add_argument('--show_plots', action='store_true',
                        help='Show the main plots.')
    parser.add_argument('--prepare_latex', action='store_true',
                        help='Prepare the latex table with the central wavelengths.')
    parser.add_argument('--loglevel', type=str, help='Log level.',
                        default='INFO')
    parser.add_argument('--debug', action='store_true',
                        help='Activate debug mode.')

    args = parser.parse_args()
    return args


def main(args):
    """Main function. Run all steps of the code in the sequence required."""

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
                    outname='lab_curves.png', figlevel='lab')

    allcurves = calc_trasm_curve(lab_filters, fnames2filters, args)
    plot_lab_curves(allcurves, fnames2filters, args,
                    outname='convoluted_curves.png', figlevel='convoluted')
    plot_all_curves(allcurves, args)
    make_final_plot(allcurves, fnames2filters, args)
    allcurves = calculate_central_lambda(allcurves, fnames2filters, args)
    plot_lab_curves(allcurves, fnames2filters, args,
                    outname='convoluted_curves_central.png', figlevel='central')
    plot_lab_curves(allcurves, fnames2filters, args,
                    outname='convoluted_curves_trapz.png', figlevel='trapz')
    plot_lab_curves(allcurves, fnames2filters, args,
                    outname='convoluted_curves_mean.png', figlevel='mean')
    plot_lab_curves(allcurves, fnames2filters, args,
                    outname='convoluted_curves_mean_1.png', figlevel='mean_1')
    plot_lab_curves(allcurves, fnames2filters, args,
                    outname='convoluted_curves_pivot.png', figlevel='pivot')
    calculate_alambda(allcurves, fnames2filters, args)
    make_html(allcurves, fnames2filters, args)
    if args.save_central_wavelentghs:
        make_csv_of_central_lambdas(allcurves, fnames2filters, args)
    if args.prepare_latex:
        prepare_latex_table(allcurves, fnames2filters, args)


def get_lab_curves(args):
    """
    Read the lab files and return a dictionary with the transmission curves.
    The files containing the lab measures were sent to the S-PLUS team by
    the J-PLUS team. The files were read and the transmission curves were
    calculated as the average of the measures. The files for each filter
    contain the wavelength, the lab transmission (before convolution with
    atmosphere and instrument), and the standard deviation of the measures.

    Parameters
    ----------
    args : argparse.Namespace

    Returns
    -------
    lab_filters : dict
    """
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
        mid_columns_std = df[df.columns[2:-1]].std(axis=1)
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
            {'wave': wave, 'transm': transmission_mean, 'std': mid_columns_std}

    del logger
    return lab_filters


def plot_lab_curves(lab_filters, fnames2filters, args, outname='fig.png', figlevel='lab'):
    """
    Plot the lab transmission curves for each filter and configuration.

    Parameters
    ----------
    lab_filters : dict
    fnames2filters : dict
    args : argparse.Namespace
    outname : str
    figlevel : str
    """
    logger = get_logger(__name__, loglevel=args.loglevel)
    fig = plt.figure(figsize=(10, 10))
    for i, filter_name in enumerate(fnames2filters.keys()):
        ax = fig.add_subplot(4, 3, i+1)
        w = lab_filters[filter_name]['wave'] * 10.
        t = lab_filters[filter_name]['transm']
        if t.max() > 1.:
            t = t / 100.
        logger.debug('Plotting filter: %s' % filter_name)
        ax.plot(w, t, lw=1.5, label=fnames2filters[filter_name]['fname'],
                color=fnames2filters[filter_name]['color'])
        if figlevel == 'lab':
            title = 'Lab transmission curves'
            logger.info('Plotting lab curve %s' % filter_name)
        elif figlevel == 'convoluted':
            title = 'Convoluted transmission curves'
            logger.info('Plotting convoluted curve %s' % filter_name)
        elif figlevel == 'central':
            title = 'Central wavelength'
            logger.debug('Plotting central wavelength %s' % filter_name)
            central_wave = lab_filters[filter_name]['central']['central_wave']
            min_wave = lab_filters[filter_name]['central']['central_wave'] - \
                lab_filters[filter_name]['central']['delta_wave'] / 2.
            max_wave = lab_filters[filter_name]['central']['central_wave'] + \
                lab_filters[filter_name]['central']['delta_wave'] / 2.
            half_height = lab_filters[filter_name]['transm'].max() / 2.
            ax.plot([min_wave, max_wave], [half_height, half_height],
                    'k-', marker='p', lw=1.5)
            ax.plot([central_wave], [half_height], 'kx', ms=10)
            ax.plot([min_wave, min_wave], [0., half_height], 'k-', lw=1.5)
            ax.plot([max_wave, max_wave], [0., half_height], 'k-', lw=1.5)
            ax.plot([central_wave, central_wave], [
                    0., half_height * 2.], 'k-', lw=1.5)
        elif figlevel == 'trapz':
            title = 'Method: Trapezoidal rule'
            logger.debug('Plotting trapezoidal rule for %s' % filter_name)
            central_wave = lab_filters[filter_name]['trapz']['central_wave']
            min_wave = lab_filters[filter_name]['trapz']['central_wave'] - \
                lab_filters[filter_name]['trapz']['delta_wave'] / 2.
            max_wave = lab_filters[filter_name]['trapz']['central_wave'] + \
                lab_filters[filter_name]['trapz']['delta_wave'] / 2.
            height = lab_filters[filter_name]['transm'].max()
            ax.fill_between([min_wave, max_wave], [height, height], color='brown',
                            alpha=0.7)
            ax.plot([central_wave, central_wave],
                    [0, height], 'k--', lw=1.5)
        elif figlevel == 'mean':
            title = 'Method: Mean'
            centr_wave = lab_filters[filter_name]['mean']['central_wave']
            min_wave = lab_filters[filter_name]['mean']['central_wave'] - \
                lab_filters[filter_name]['mean']['delta_wave'] / 2.
            max_wave = lab_filters[filter_name]['mean']['central_wave'] + \
                lab_filters[filter_name]['mean']['delta_wave'] / 2.
            print(min_wave, max_wave,
                  lab_filters[filter_name]['mean']['delta_wave'])
            ax.fill_between([min_wave, max_wave], 0, t.max(), color='brown',
                            alpha=0.7)
            ax.plot([centr_wave, centr_wave], [
                    0, t.max()], '--', color='k', lw=1.5)
        elif figlevel == 'mean_1':
            title = 'Method: Mean with 1\% threshold'
            centr_wave = lab_filters[filter_name]['mean_1']['central_wave']
            min_wave = lab_filters[filter_name]['mean_1']['central_wave'] - \
                lab_filters[filter_name]['mean_1']['delta_wave'] / 2.
            max_wave = lab_filters[filter_name]['mean_1']['central_wave'] + \
                lab_filters[filter_name]['mean_1']['delta_wave'] / 2.
            ax.fill_between([min_wave, max_wave], 0, t.max(), color='brown',
                            alpha=0.7)
            ax.plot([centr_wave, centr_wave], [
                    0, t.max()], '--', color='k', lw=1.5)
        elif figlevel == 'pivot':
            title = 'Method: Pivot lambda'
            centr_wave = lab_filters[filter_name]['pivot']['central_wave']
            min_wave = lab_filters[filter_name]['pivot']['central_wave'] - \
                lab_filters[filter_name]['pivot']['delta_wave'] / 2.
            max_wave = lab_filters[filter_name]['pivot']['central_wave'] + \
                lab_filters[filter_name]['pivot']['delta_wave'] / 2.
            ax.plot([centr_wave, centr_wave], [
                    0, t.max()], '--', color='k', lw=1.5)
        else:
            logger.critical('Unknown figlevel: %s' % figlevel)
            raise ValueError('Unknown figlevel: %s' % figlevel)
        plt.legend()
        ax.set_ylim(0, 1.)
        if i == 1:
            ax.set_title(title)
        if i == 10:
            ax.set_xlabel('Wavelength [A]')
        if i == 3:
            ax.set_ylabel('Transmission')
        plt.grid()
    plt.tight_layout()

    if args.save_plots:
        logger.info('Saving fig to %s' % os.path.join(args.work_dir, outname))
        plt.savefig(os.path.join(args.work_dir, outname), dpi=300)
    if args.show_plots:
        logger.debug('Showing plot.')
        plt.show()
    else:
        logger.debug('Closing plot.')
        plt.close()
    del logger


def calc_trasm_curve(lab_filters, fnames2filters, args):
    """
    Calculate the transmission curve of the filters.

    Parameters
    ----------
    lab_filters : dict
    fnames2filters : dict
    args : argparse.Namespace

    Returns
    -------
    allcurves : dict
    """
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
        lab_std = lab_filters[lab_curve]['std']
        lab_ius = interp1d(lab_wave, lab_transm)
        std_ius = interp1d(lab_wave, lab_std)
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
            new_std = std_ius(wave_range)
            dat = wave_range, new_filter_trans, new_std
            outputname = "".join([fnames2filters[lab_curve]['fname'], '.csv'])
            logger.info('Wrinting file: %s.', outputname)
            np.savetxt(outputname, np.transpose(dat), delimiter=',',
                       header='wavelength,transmittance,std')
            if args.show_individual_filters:
                logger.info('Plotting filter: %s.', lab_curve)
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

    del logger
    return allcurves


def plot_all_curves(allcurves, args):
    """
    Plot all the transmission curves in the same plot.

    Parameters
    ----------
    allcurves : dict
    args : argparse.Namespace
    """
    logger = get_logger(__name__, loglevel=args.loglevel)
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
    plt.grid()
    plt.tight_layout()
    if args.save_plots:
        logger.debug('Saving plot to file.')
        plt.savefig(os.path.join(work_dir, 'allcurves.png'), dpi=300)
    if args.show_plots:
        plt.show()
        plt.close()
    else:
        plt.close()


def make_final_plot(allcurves, fnames2filters, args):
    """
    Make the final plot of the transmission curves.

    Parameters
    ----------
    allcurves : dict
    fnames2filters : dict
    args : argparse.Namespace
    """

    logger = get_logger(__name__, loglevel=args.loglevel)
    logger.info('Making presentation plot.')
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

    if args.save_plots:
        logger.info('Saving plot to %s.', os.path.join(
            work_dir, 'splus_filters.png'))
        plt.savefig(os.path.join(work_dir, 'splus_filters.png'),
                    format='png', dpi=300)
    if args.show_plots:
        logger.debug('Showing plot.')
        plt.show()
        plt.close()
    else:
        plt.close()
    del logger


def calculate_central_lambda(allcurves, fnames2filters, args):
    """
    Calculate the central wavelength of the filters.

    Parameters
    ----------
    allcurves : dict
    fnames2filters : dict
    args : argparse.Namespace

    Returns
    -------
    allcurves : dict
    """

    logger = get_logger(__name__, loglevel=args.loglevel)
    logger.info('Calculating central wavelength')

    for curve in fnames2filters.keys():
        wave = allcurves[curve]['wave'] * 10.
        transm = allcurves[curve]['transm']
        interp = interp1d(wave, transm)
        synt_wave = np.linspace(min(wave), max(wave), 100000)
        synt_transm = interp(synt_wave)
        logger.debug('Claculating curves via trapezoidal rule approach')
        half_height = max(synt_transm) / 2.
        left_index = np.where(synt_transm > half_height)[0][0]
        right_index = np.where(synt_transm > half_height)[0][-1]
        min_wave = synt_wave[left_index]
        max_wave = synt_wave[right_index]
        central_wave = (max_wave + min_wave) / 2.
        allcurves[curve]['central'] = {'central_wave': central_wave,
                                       'delta_wave': max_wave - min_wave}

        logger.debug('Calculating trapezoidal rule')
        mid_wave = np.trapz(synt_wave * synt_transm, synt_wave) / np.trapz(
            synt_transm, synt_wave)
        trapz_width = np.trapz(synt_transm, synt_wave) / max(synt_transm)
        allcurves[curve]['trapz'] = {'central_wave': mid_wave,
                                     'delta_wave': trapz_width}

        logger.debug('Calculating mean with 0% threshold')
        mask = synt_transm > synt_transm.max() * 0.0
        lambda_mean = np.sum(
            synt_wave[mask] * synt_transm[mask]) / np.sum(synt_transm[mask])
        mean_width = np.trapz(synt_transm[mask], synt_wave[mask]) / max(
            synt_transm[mask])
        allcurves[curve]['mean'] = {'central_wave': lambda_mean,
                                    'delta_wave': mean_width}

        logger.debug('Calculating mean with 1% threshold')
        mask = synt_transm > synt_transm.max() * 0.01
        lambda_mean = np.sum(synt_wave[mask] * synt_transm[mask]) / np.sum(
            synt_transm[mask])
        mean_width = np.trapz(synt_transm[mask], synt_wave[mask]) / max(
            synt_transm[mask])
        allcurves[curve]['mean_1'] = {'central_wave': lambda_mean,
                                      'delta_wave': mean_width}

        logger.debug('Calculating pivot wavelength')
        lambda_pivot = np.sqrt(np.sum(synt_transm) /
                               np.sum(synt_transm / synt_wave**2))
        mean_pivot = max_wave - min_wave
        allcurves[curve]['pivot'] = {'central_wave': lambda_pivot,
                                     'delta_wave': mean_pivot}

    del logger
    return allcurves


def make_html(allcurves, fnames2filters, args):
    """
    Make a html file with the central wavelengths of the filters.

    Parameters
    ----------
    allcurves : dict
    fnames2filters : dict
    args : argparse.Namespace
    """
    logger = get_logger(__name__, loglevel=args.loglevel)
    htmlf = open(os.path.join(args.work_dir, 'central_wavelengths.html'), 'w')
    htmlf.write('<div class="dlpage">\n')
    htmlf.write('<table class="docutils" style="width:100%" border=1>\n')
    htmlf.write('<colgroup>\n')
    htmlf.write('<tr>')
    htmlf.write('<th colspan="11"><b>S-PLUS filters summary</b></th>\n')
    htmlf.write('</tr>\n')
    htmlf.write('<tr>')
    htmlf.write('<td>Filter</td>\n')
    htmlf.write('<td>λ<sub>central</sub></td>\n')
    htmlf.write('<td>FWHM</td>\n')
    htmlf.write('<td>λ<sub>trapz</sub></td>\n')
    htmlf.write('<td>W<sub>trapz</sub></td>\n')
    htmlf.write('<td>λ<sub>mean</sub></td>\n')
    htmlf.write('<td>Δλ<sub>mean</sub></td>\n')
    htmlf.write('<td>λ<sub>mean</sub> (>1%)</td>\n')
    htmlf.write('<td>W<sub>mean</sub> (>1%)</td>\n')
    htmlf.write('<td>λ<sub>pivot</sub></td>\n')
    htmlf.write('<td>A<sub>λ</sub>/A<sub>V</sub></td>\n')
    htmlf.write('</tr>\n')
    htmlf.write('</colgroup>\n')
    logger.info('Writing central wavelengths to html file')
    for curve in fnames2filters.keys():
        htmlf.write('<tr>\n')
        htmlf.write('<td>%s</td>\n' % fnames2filters[curve]['fname'])
        htmlf.write('<td>%.0f</td>\n' %
                    allcurves[curve]['central']['central_wave'])
        htmlf.write('<td>%.0f</td>\n' %
                    allcurves[curve]['central']['delta_wave'])
        htmlf.write('<td>%.0f</td>\n' %
                    allcurves[curve]['trapz']['central_wave'])
        htmlf.write('<td>%.0f</td>\n' %
                    allcurves[curve]['trapz']['delta_wave'])
        htmlf.write('<td>%.0f</td>\n' %
                    allcurves[curve]['mean']['central_wave'])
        htmlf.write('<td>%.0f</td>\n' % allcurves[curve]['mean']['delta_wave'])
        htmlf.write('<td>%.0f</td>\n' %
                    allcurves[curve]['mean_1']['central_wave'])
        htmlf.write('<td>%.0f</td>\n' %
                    allcurves[curve]['mean_1']['delta_wave'])
        htmlf.write('<td>%.0f</td>\n' %
                    allcurves[curve]['pivot']['central_wave'])
        htmlf.write('<td>%.3f</td>\n' % allcurves[curve]['a_lambda_a_v'])
        htmlf.write('</tr>\n')
    htmlf.write('</table>\n')
    htmlf.write('</div>\n')
    htmlf.close()
    del logger


def make_csv_of_central_lambdas(allcurves, fnames2filters, args):
    """
    Make a csv file with the central wavelengths of the filters.

    Parameters
    ----------
    allcurves : dict
    fnames2filters : dict
    args : argparse.Namespace

    Returns
    -------
    allcurves : dict
    """
    logger = get_logger(__name__, loglevel=args.loglevel)
    workdir = args.work_dir
    logger.info('Writing central wavelengths to csv file')
    filters = []
    central_wave = []
    delta_wave = []
    trapz_wave = []
    trapz_width = []
    mean_wave = []
    mean_width = []
    mean_1_wave = []
    mean_1_width = []
    pivot_wave = []
    alambda_av = []
    for curve in fnames2filters.keys():
        logger.debug('Getting params for %s' % fnames2filters[curve]['fname'])
        filters.append(fnames2filters[curve]['fname'])
        central_wave.append(allcurves[curve]['central']['central_wave'])
        delta_wave.append(allcurves[curve]['central']['delta_wave'])
        trapz_wave.append(allcurves[curve]['trapz']['central_wave'])
        trapz_width.append(allcurves[curve]['trapz']['delta_wave'])
        mean_wave.append(allcurves[curve]['mean']['central_wave'])
        mean_width.append(allcurves[curve]['mean']['delta_wave'])
        mean_1_wave.append(allcurves[curve]['mean_1']['central_wave'])
        mean_1_width.append(allcurves[curve]['mean_1']['delta_wave'])
        pivot_wave.append(allcurves[curve]['pivot']['central_wave'])
        alambda_av.append(allcurves[curve]['a_lambda_a_v'])
    data = {'filter': filters,
            'central_wave': central_wave,
            'delta_wave': delta_wave,
            'trapz_wave': trapz_wave,
            'trapz_width': trapz_width,
            'mean_wave': mean_wave,
            'mean_width': mean_width,
            'mean_1_wave': mean_1_wave,
            'mean_1_width': mean_1_width,
            'pivot_wave': pivot_wave,
            'alambda_av': alambda_av}
    df = pd.DataFrame(data)
    logger.info('Writing central wavelengths to csv file')
    df.to_csv(os.path.join(workdir, 'central_wavelengths.csv'), index=False)
    del logger


def calculate_alambda(allcurves, fnames2filters, args):
    """
    Calculate the A_lambda/A_V for each filter.
    The opacity file was obtained from:
    http://svo2.cab.inta-csic.es/theory/fps/getextlaw.php
    The value for kv was obtained from:
    http://svo2.cab.inta-csic.es/theory/fps/index.php?id=CTIO/S-PLUS.z&&mode=browse&gname=CTIO&gname2=S-PLUS#filter

    Parameters
    ----------
    allcurves : dict
    fnames2filters : dict
    args : argparse.Namespace

    Returns
    -------
    allcurves : dict
    """

    logger = get_logger(__name__, loglevel=args.loglevel)
    logger.info('Calculating A_lambda')
    workdir = args.work_dir
    data_extra_dir = os.path.join(workdir, 'data-extra')
    if not os.path.exists(data_extra_dir):
        logger.critical(
            'Directory %s does not exist. Please make sure work_dir points to the right place.' % data_extra_dir)
        raise ValueError(
            'Directory %s does not exist. Please make sure work_dir points to the right place.' % data_extra_dir)
    data_extinction_file = os.path.join(
        data_extra_dir, 'ExtLaw_FitzIndeb_3.1.dat')
    opacity_tab = ascii.read(data_extinction_file)
    opacity_wave = opacity_tab['wave(A)']
    opacity = opacity_tab['opacity(cm2/g)']
    kv = 211.4
    interp_opacity = interp1d(opacity_wave, opacity)
    for curve in fnames2filters.keys():
        lambda_pivot = allcurves[curve]['pivot']['central_wave']
        a_lambda_a_v = interp_opacity(lambda_pivot) / kv
        allcurves[curve]['a_lambda_a_v'] = a_lambda_a_v

    del logger
    return allcurves


def prepare_latex_table(allcurves, fnames2filters, args):
    """
    Prepare a latex table with the central wavelengths of the filters.
    """

    logger = get_logger(__name__, loglevel=args.loglevel)
    logger.info('Preparing latex table')
    latex_filename = os.path.join(args.work_dir, 'central_wavelengths.tex')
    with open(latex_filename, 'w') as f:
        f.write('\\begin{table*}\n')
        f.write('\\centering\n')
        f.write('\\caption{Central wavelengths of the S-PLUS filters.}\n')
        f.write('\\label{tab:central_wavelengths}\n')
        f.write('\\begin{tabular}{ccccccc}\n')
        f.write('\\hline\n')
        f.write('\\hline\n')
        f.write(
            'Filter & $\\lambda_{\\mathrm{central}}$ & FWHM & $\\lambda_{\\mathrm{mean}}$ & $\\Delta\\lambda_{\\mathrm{mean}}$ & $\\lambda_{\\mathrm{pivot}}$ & $A_{\\lambda}/A_{V}$ \\\\\n')
        f.write(" & [\\AA] & [\\AA] & [\\AA] & [\\AA] & [\\AA] & \\\\\n")
        f.write('\\hline\n')
        for curve in fnames2filters.keys():
            f.write('%s & %.0f & %.0f & %.0f & %.0f & %.0f & %.3f\\\\\n' %
                    (fnames2filters[curve]['fname'],
                     allcurves[curve]['central']['central_wave'],
                     allcurves[curve]['central']['delta_wave'],
                     allcurves[curve]['mean']['central_wave'],
                     allcurves[curve]['mean']['delta_wave'],
                     allcurves[curve]['pivot']['central_wave'],
                     allcurves[curve]['a_lambda_a_v']))
        f.write('\\hline\n')
        f.write('\\end{tabular}\n')
        f.write('\\end{table*}\n')
    del logger
    return


if __name__ == '__main__':
    args = get_args()
    main(args)
