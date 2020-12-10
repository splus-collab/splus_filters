# make the filters transmitance for S-PLUS fig
# v02 from 2020-09-25 by F. R. Herpich
# contact: herpich@usp.br

from astropy.io import ascii
import matplotlib.pyplot as plt

filters = {'uJAVA': ('uJAVA', 'navy', 3563, -500, -4),
           'F0378': ('J0378', 'darkviolet', 3770, -350, 1),
           'F0395': ('J0395', 'b', 3940, -400, 1),
           'F0410': ('J0410', 'dodgerblue', 4094, -350, -0.1),
           'F0430': ('J0430', 'c', 4292, -400, 1.3),
           'gSDSS': ('gSDSS', 'turquoise', 4751, -300, 0.),
           'F0515': ('J0515', 'lime', 5133, -100, 1.5),
           'rSDSS': ('rSDSS', 'greenyellow', 6258, -500, -2),
           'F0660': ('J0660', 'orange', 6614, -300, 1),
           'iSDSS': ('iSDSS', 'darkorange', 7690, -300, 1),
           'zSDSS': ('zSDSS', 'r', 8831, 300, -12),
           'F0861': ('J0861', 'orangered', 8611, -250, 1)
           }

fig = plt.figure(figsize=(8, 4))
for key in filters.keys():
    f = ascii.read('%s.dat' % key)
    plt.plot(f['col0']*10., f['col1']*100., color=filters[key][1], lw=1.5)
    plt.text(filters[key][2] + filters[key][3],
             max(f['col1']*100.) + filters[key][4],
             filters[key][0], fontsize=12, color=filters[key][1])

plt.xlabel(r'$\lambda\ \mathrm{[\AA]}$', fontsize=16)
plt.ylabel(r'$R_\lambda\ \mathrm{[\%]}$', fontsize=16)
plt.xlim(3000, 10000)
plt.ylim(0.2, 83)
plt.tight_layout()

plt.savefig('splus_filters.png', format='png')
plt.show(block=False)
