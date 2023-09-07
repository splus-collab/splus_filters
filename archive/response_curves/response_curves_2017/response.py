# Herpich F.R. 12/08/16

from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline as IUS
from scipy.interpolate import interp1d
import os

filters = ['20140604C080F062502.fits', '20140606C080F051502.fits',
           '20140609C080F066002.fits', '20150429C080F037802.fits',
           '20150504C080zSDSS02.fits', '20150506C080iSDSS02.fits',
           '20150514C080F043002.fits', '20150918C080uJAVA02.fits',
           '20150922C080F039502.fits', '20150922C080F086102.fits',
           '20150923C080F041002.fits', '20150924C080gSDSS02.fits']

def str2float(fits_file):
    mirror_col1 = []
    mirror_col2 = []
    mirror_reflec = fits.open(fits_file)[1].data
    for i in range(len(mirror_reflec.col1)):
        mirror_col1.append(np.float(mirror_reflec.col1[i]))
        if np.float(mirror_reflec.col2[i]) < 0.:
            mirror_col2.append(0.)
        else:
            mirror_col2.append(np.float(mirror_reflec.col2[i]) / 100.)
    return mirror_col1, mirror_col2

def calc_filters(filters):
    #wave_range = np.arange(300., 1201., 1)
    #order = 1

    ##################### for sky ########################
    atmosph_transm = fits.open('sky_trans.fits')[1].data
    #at_ius = IUS(atmosph_transm.col1, atmosph_transm.col2, k = order, ext=0)
    at_ius = interp1d(atmosph_transm.col1, atmosph_transm.col2)
    #new_atm_trans = at_ius(wave_range)

    ##################### for mirror ########################
    mirror_col1, mirror_col2 = str2float('mirror_reflectance.fits')
    #mr_ius = IUS(mirror_col1, mirror_col2, k = order)
    # the reflectance bellow was obtained from:
    # https://laserbeamproducts.wordpress.com/2014/06/19/reflectivity-of-aluminium-uv-visible-and-infrared/
    #mirror_col1 = np.array([248, 400, 532, 633, 800, 900, 1000, 3000])
    #mirror_col2 = np.array([.926, .92, .916, .907, .868, .89, .94, .98])
    mr_ius = interp1d(mirror_col1, mirror_col2)
    #new_mirror_reflec = mr_ius(wave_range)
    # measured
    mirror_measured_wave = np.array([300., 350., 420., 470., 530., 650., 880.,
                                     950., 1000., 1100])
    mirror_measured_flux = np.array([.9126, .9126, .9126, .9126,
                                     .911, .8725, .7971, .82, .84, .85])
    mr_meas = interp1d(mirror_measured_wave, mirror_measured_flux)
    mask = (mirror_col1 > min(mirror_measured_wave)) & (mirror_col1 < max(mirror_measured_wave))
    measur_interp = mr_meas(np.array(mirror_col1)[mask])

    ##################### for ccd ########################
    ccd_col1, ccd_col2 = str2float('ccd_curve.fits')
    #ccd_ius = IUS(ccd_wave, ccd_flux, k = order, ext=3)
    ccd_ius = interp1d(ccd_col1, ccd_col2)
    #new_ccd_eff = ccd_ius(wave_range)
    # measured (????)
    ccd_measured_wave = np.array([300., 350., 400., 450., 500., 550., 600.,
                                  650., 725., 800., 850., 900, 970.])
    ccd_measured_flux = np.array([.2, .45, .90, .93, .88, .88, .91, .92, .95,
                                     .88, .8, .6, .3])
    ccd_measured_wave2 = np.array([350., 400., 500., 650., 900])
    ccd_measured_flux2 = np.array([.39, .84, .92, .924, .61])


    fig = plt.figure()
    ax = fig.add_subplot(111)
    for nof in filters:
        t = fits.open(nof)[1].data

        #mask = (wave_range >= t.col1.min()) & (wave_range <= t.col1.max())
        xmin = np.array([min(atmosph_transm.col1), min(mirror_col1),
                         min(ccd_col1), min(t.col1)])
        xmax = np.array([max(atmosph_transm.col1), max(mirror_col1),
                         max(ccd_col1), max(t.col1)])
        wave_range = np.arange(max(xmin), min(xmax), 1.)

        col = t.col3 / t.col2

        k = 4
        while k < 103:
            ncol = 'col' + str(k)
            col += t[ncol] / t.col2
            k += 1

        medium_col = col / 100.
        ##################### for filter ########################
        #tex = IUS(t.col1, medium_col, k = order)
        tex = interp1d(t.col1, medium_col)
        transm = tex(wave_range)
        ##################### for sky ########################
        # typical
        new_atm_trans = at_ius(wave_range)
        ##################### for mirror ########################
        new_mirror_reflec = mr_ius(wave_range)
        mirror_measured_wave = np.array([300., 350., 420., 470., 530., 650., 880.])
        mirror_measured_flux = np.array([.9126, .9126, .9126, .9126,
                                         .911, .8725, .7971])
        ##################### for ccd ########################
        new_ccd_eff = ccd_ius(wave_range)

        ################ calc new transmittance for each filter ################
        new_filter_trans = transm * new_atm_trans * new_mirror_reflec * new_ccd_eff

        # writing fits files
        filter_name = nof[12:-7] + '.fits'
        column1 = fits.Column(name='wavelength', format='E', array = wave_range)
        column2 = fits.Column(name='transmit', format='E', array = new_filter_trans)
        cols = fits.ColDefs([column1, column2])
        tbhdu = fits.BinTableHDU.from_columns(cols)
        if os.path.isfile(filter_name):
            os.remove(filter_name)
        tbhdu.writeto(filter_name)
        print '---'
        print 'saving filter transmitance', filter_name, '\n'

        # writing ascii files
        nascii_file = nof[:-5] + '.ascii'
        f = open(nascii_file, 'w')
        f.write('wavelength transmittance\n')
        for i in range(len(wave_range)):
            linha = '%s %s\n' % (wave_range[i], new_filter_trans[i])
            f.write(linha)
        print 'saving filter transmittance', nascii_file, '\n'
        f.close()

        ax.plot(wave_range * 10., new_filter_trans, label = nof[12:-7])



    ax.plot(atmosph_transm.col1 * 10., atmosph_transm.col2, c = 'c',
            label = 'atmosphere')

    ax.plot(np.array(mirror_col1) * 10., mirror_col2, c = 'k', label = 'mirror')
    ax.plot(mirror_measured_wave * 10., mirror_measured_flux, 'o', c = 'k',
            label = 'meas mirror')
    ax.plot(np.array(mirror_col1)[mask] * 10., measur_interp, '--', c = 'k',
            label = 'meas mirror')

    ax.plot(np.array(ccd_col1) * 10., ccd_col2, c = 'r', label = 'ccd eff')
    ax.plot(ccd_measured_wave * 10., ccd_measured_flux, 's',
        c = 'r', label = 'meas ccd')
    ax.plot(ccd_measured_wave2 * 10., ccd_measured_flux2, 'd',
        c='y', label = 'meas ccd2')
    ax.set_xlim(2900, 12000)
    ax.set_ylim(-.01, 1.)
    ax.set_xlabel(r'$\mathrm{\lambda\ [\AA]}$')
    plt.legend(loc = 'upper right', fontsize = 10)

    plt.show()
    plt.close()
    return

calc_filters(filters)

def get_desv(filters):
    n = 1
    fig = plt.figure(figsize = (8, 7))
    for nof in filters:
        t = fits.open(nof)[1].data

        ax = fig.add_subplot(6, 2, n)
        for i in range(len(t.col3)):
            if t.col3[i] == max(t.col3):
                ax.plot(t.T[i][2:-1], '-', label = nof[12:-7])
        if n in [11, 12]:
            ax.set_xlabel(r'$\mathrm{pixel}$')
        else:
            plt.setp(ax.get_xticklabels(), visible = False)
        n += 1
        plt.legend(loc = 'upper right', fontsize = 10)
        #labels = [item.get_text() for item in ax.get_yticklabels()]
        #print labels
        #ax.set_yticklabels(labels[1:-1])

    plt.tight_layout()
    plt.subplots_adjust(hspace = 0.1)

    plt.show()
    plt.close()
    return

#get_desv(filters)
