# Herpich F.R. 12/08/16

from astropy.io import fits,ascii
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
import sys

if len(sys.argv) == 1:
    save_ascii_filters = False
else:
    save_ascii_filters = True

filters = ['20140604C080F062502.fits', '20140606C080F051502.fits',
           '20140609C080F066002.fits', '20150429C080F037802.fits',
           '20150924C080gSDSS02.fits', '20150506C080iSDSS02.fits',
           '20150514C080F043002.fits', '20150918C080uJAVA02.fits',
           '20150922C080F039502.fits', '20150922C080F086102.fits',
           '20150923C080F041002.fits', '20150504C080zSDSS02.fits']

z = fits.open('20150504C080zSDSS02.fits')[1].data
maxfilter = z.col1.max()
##################### for sky ########################
atmosph_transm = ascii.read('atmospheric_transmission2.csv')
at_wave = atmosph_transm['wave']/10.
at_transm = atmosph_transm['transm']
at_ius = interp1d(atmosph_transm['wave']/10., atmosph_transm['transm'])

##################### for mirror ########################
mirror_reflect = fits.open('mirror_reflectance.fits')[1].data
mirror_col1 = np.float_(mirror_reflect.col1)
mirror_col2 = np.float_(mirror_reflect.col2)
# the reflectance bellow was obtained from:
# https://laserbeamproducts.wordpress.com/2014/06/19/reflectivity-of-aluminium-uv-visible-and-infrared/
mr_ius = interp1d(mirror_col1, mirror_col2)
# measured
mirror_measured_wave = np.array([300., 350., 420., 470., 530., 650., 880.,
                                 950., 1000., 1100])
mirror_measured_flux = np.array([.9126, .9126, .9126, .9126,
                                 .911, .8725, .7971, .82, .84, .85])
mr_meas = interp1d(mirror_measured_wave, mirror_measured_flux)
mask = (mirror_col1 > min(mirror_measured_wave)) & (mirror_col1 < max(mirror_measured_wave))
measur_interp = mr_meas(np.array(mirror_col1)[mask])

##################### for ccd ########################
ccd_curve = fits.open('ccd_curve.fits')[1].data
ccd_col1 = np.float_(ccd_curve.col1)
ccd_col2 = np.float_(ccd_curve.col2)
ccd_ius = interp1d(np.float_(ccd_curve.col1), np.float_(ccd_curve.col2))
# measured
ccd_measured_wave = np.array([300., 350., 400., 450., 500., 550., 600.,
                              650., 725., 800., 850., 900, 970.])
ccd_measured_flux = np.array([.2, .45, .90, .93, .88, .88, .91, .92, .95,
                                 .88, .8, .6, .3])

fig = plt.figure()
ax = fig.add_subplot(111)

for nof in filters:
    t = fits.open(nof)[1].data
    fwave = t.col1
    fnorm = (t.col2 + t.col103) / 2.
    ttransm = t.col3
    for col in t.columns.names[3:-1]:
        ttransm += t[col]

    ttransm = ttransm / len(t.columns.names[2:-1])
    ttransm = ttransm / fnorm

    xmin = np.array([min(atmosph_transm['wave']/10.), min(mirror_col1),
                     min(ccd_col1), min(t.col1)])
    xmax = np.array([max(atmosph_transm['wave']/10.), max(mirror_col1),
                     max(ccd_col1), max(t.col1)], maxfilter)
    #wave_range = np.arange(max(xmin), min(xmax), 1.)
    wave_range = np.arange(max(xmin), fwave.max(), 1.)

    ##################### for filter ########################
    tex = interp1d(t.col1, ttransm)
    transm = tex(wave_range)
    ##################### for sky ########################
    # typical
    new_atm_trans = at_ius(wave_range)
    ##################### for mirror ########################
    new_mirror_reflec = mr_ius(wave_range) / 100.
    mirror_measured_wave = np.array([300., 350., 420., 470., 530., 650., 880.])
    mirror_measured_flux = np.array([.9126, .9126, .9126, .9126,
                                     .911, .8725, .7971])
    ##################### for ccd ########################
    new_ccd_eff = ccd_ius(wave_range) / 100.

    ################ calc new transmittance for each filter ################
    new_filter_trans = transm * new_atm_trans * new_mirror_reflec * new_ccd_eff

    if save_ascii_filters:
        # writing ascii files
        dat = wave_range, new_filter_trans
        nascii_file = nof[:-5] + '.ascii'
        print('writing file', nascii_file)
        ascii.write(dat, nascii_file, overwrite=True)

    ax.plot(wave_range * 10., new_filter_trans, label=nof[12:-7])

if not save_ascii_filters:
    ax.plot(atmosph_transm['wave'], atmosph_transm['transm'], c='c',
            label='atmosphere')

    ax.plot(np.array(mirror_col1) * 10., mirror_col2, c='k', label='mirror')
    ax.plot(mirror_measured_wave * 10., mirror_measured_flux, 'o', c='k',
            label='meas mirror')
    ax.plot(np.array(mirror_col1)[mask] * 10., measur_interp, '--', c='k',
            label='meas mirror')

    ax.plot(np.array(ccd_col1) * 10., ccd_col2/100., c='r', label='ccd eff')
    ax.plot(ccd_measured_wave * 10., ccd_measured_flux, 's', c='r', label='meas ccd')
ax.set_xlim(2900, 12000)
ax.set_ylim(-.01, 1.)
ax.set_xlabel(r'$\mathrm{\lambda\ [\AA]}$')
plt.legend(loc='upper right', fontsize=10)

plt.show()
