#!/bin/python

# make the filters transmitance for S-PLUS fig
# v01 from 2019-08-15 by F. Herpich
# contact: fabiorafaelh@gmail.com

from astropy.io import ascii
import matplotlib.pyplot as plt

filters = {'U': ('u', 'navy'), 'F378': ('J0378', 'darkviolet'),
           'F395': ('J0395', 'b'), 'F410': ('J0410', 'dodgerblue'),
           'F430': ('J0430', 'c'), 'G': ('g', 'turquoise'),
           'F515': ('J0515', 'lime'), 'R': ('r', 'greenyellow'),
           'F660': ('J0660', 'orange'), 'I': ('i', 'darkorange'),
           'Z': ('z', 'r'), 'F861': ('J0861', 'orangered')}

fig = plt.figure(figsize=(8, 4))
for key in filters.keys():
    f = ascii.read('%s.dat' % key)
    plt.plot(f['col0']*10., f['col1']*100., color=filters[key][1], lw=1.5)

plt.xlabel(r'$\lambda\ \mathrm{[\AA]}$', fontsize=16)
plt.ylabel(r'$R_\lambda\ \mathrm{[\%]}$', fontsize=16)
plt.xlim(3000, 10000)
plt.ylim(0.2, 80)
plt.tight_layout()

plt.savefig('splus_filters.png', format='png')
plt.show()
