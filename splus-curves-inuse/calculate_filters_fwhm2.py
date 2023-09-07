#!/bin/python

from astropy.io import ascii
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import sys

if len(sys.argv) == 2:
    Filter = sys.argv[1]
    color = None
    save_fig = False
elif len(sys.argv) == 3:
    Filter = sys.argv[1]
    color = sys.argv[2]
    save_fig = False
elif len(sys.argv) == 4:
    Filter = sys.argv[1]
    color = sys.argv[2]
    save_fig = True
else:
    raise IOError('the format for input is: filter_file, color and name_of_filter')

f = ascii.read(Filter)
#f = fits.open(Filter)[1].data

wave = f['col0']*10.
flux = f['col1']
#wave = f.wavelength * 10.
#flux = f.transmit

halfheight = flux.max() / 2.

interp = interp1d(wave, flux)
synwave = np.linspace(wave.min(), wave.max(), 100000)
syncol = interp(synwave)

cont = True
inivalue = 0.0001
while cont:
    mask = (syncol > 0.01) & (syncol > (halfheight - inivalue)) & (syncol < (halfheight + inivalue))
    if mask.sum() >= 2:
        minlam = synwave[mask][0]
        maxlam = synwave[mask][-1]
        #if (maxlam - minlam) < 90.:
        #    minlam = np.mean(synwave[mask])
        #    maxlam = max(synwave)
        cont = False
    else:
        cont = True
        inivalue += inivalue


diff = maxlam - minlam
leff = (maxlam + minlam) / 2.

plt.plot(wave, flux, c=color, label='obs')
plt.plot([minlam, maxlam], [halfheight, halfheight], c='k',
         label=r'$\Delta\lambda = %.0f\mathrm{\AA}$' % diff)
plt.scatter(minlam, halfheight, color='k', marker='d',
         label=r'$\lambda_0 = %.0f$' % minlam)
plt.scatter(maxlam, halfheight, color='k', marker='d',
         label=r'$\lambda_1 = %.0f$' % maxlam)
plt.scatter(leff, halfheight, marker='o', color='k',
            label=r'$\lambda_\mathrm{eff} = %.0f\mathrm{\AA}$' % leff)
plt.legend(loc='upper right')
plt.grid()

if save_fig:
    plt.title(sys.argv[3])
    print('safing fig', sys.argv[1].split('.')[0] + '.png')
    plt.savefig(sys.argv[1].split('.')[0] + '.png', format='png', dpi=100)

plt.show()
