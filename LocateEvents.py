#!/usr/bin/python3

import os
import re
import sys
import cv2
import tkinter
from tkinter.filedialog import askopenfilename
from tkinter.messagebox import askyesno, showerror
from tkinter.simpledialog import askinteger
from struct import *
from numpy import *
from matplotlib.pyplot import *
from time import *
from scipy.interpolate import *
from sklearn.decomposition import PCA

# GLOBALS!!!
def set_GLOBALS():
    global indE, SKIP
    SKIP    =    1     # how many frames to skip while downloading dat-files


# Processing of the mouse double click to select the centers of the pulses
def BorderLocation(event):
    global indE
    if event.dblclick:
        sec = getattr(event, 'xdata')
        num = int(round(sec * EFREQ))
        indE.append(num)
        plot([sec,sec], [yl[0],yl[1]], '-.r', lw=1.0)
        fig.show()

# The Principal Components Analysis
def doTheJob():
    # select the continuous regions
    if len(indE) > 1:
        indE.sort()
    # the EventsList file
    fnum = 0
    while os.path.isfile(path + name + '-EventsList-%02d.log' % fnum):
        fnum += 1
    W = open(path + name + '-EventsList-%02d.log' % fnum, 'w')
    W.write('# %d Hz\n' % EFREQ)
    for p in range(len(indE)):
        W.write('%10.3f\t%9d\n' % (indE[p] / EFREQ, indE[p]))
        W.flush()
    W.close()

#
# Read the L-CARD data file 
# numc   - number of the channels => data columns in the file
# itype  - type of the data in the dat-file
# otype  - type of the data in the returned array
# lim    - limits to read from the file, frame numbers
# skip   - how namy wrames should be skipped
# 
def readLCARD(fname, numc = 32, itype = False, otype = float32, skip = 10):
    global D

    print(' - reading ' + fname + ' L-CARD data file')

    t = time()
    sz = os.path.getsize(fname) / 1024 ** 3

    # itype definition
    if not itype:
        # 16 kByte test sample to unpack and try to unpack
        num = 16 * 1024
        # types dictionary
        dct = {0 : float16, 1 : float32, 2 : float64} 
        F = open(fname, 'rb')
        tmp = F.read(num)
        F.close()
        var = []
        var.append(abs(array(unpack('%de' % (num / 2), tmp))).max())
        var.append(abs(array(unpack('%df' % (num / 4), tmp))).max())
        var.append(abs(array(unpack('%dd' % (num / 8), tmp))).max())
        var = (array(var) < 15).nonzero()[0][0]
        itype = dct[var] 
        print(' - itype was automatically assigned to %s' % itype)

    if itype == float16:
        line = '%de' % numc
        numb  =  2
    elif itype == float32:
        line = '%df' % numc
        numb  =  4
    elif itype == float64:
        line = '%dd' % numc
        numb  =  8
    else:
        print(' ERROR!!!')
        print(' intype must be one of the list: float16, float32, or float 64\n')
        print(' however it was: ' + itype)
        exit()
    
    F = open(fname, 'rb')
    D = []
    tmp = bytearray(F.read(numc * numb))
    # read the data
    while len(tmp) == numc * numb:
        tmp = array(unpack(line, tmp), dtype = itype)
        D.append(array(tmp))
        F.seek(numc * numb * (skip - 1), 1)
        tmp = bytearray(F.read(numc * numb))
        if len(D) % 100000 == 0:
            dn = len(D) * len(D[-1]) * numb * skip / 1024 ** 3
            tmp1 = time()
            print('\r   %.2f of %.2f GB read (%.1f MB/sec), remaining runtime %d sec       ' % (dn, sz, 1024 * dn / (tmp1 - t), (sz / dn - 1) * (tmp1 - t)), end = ' ')
        '''
        # part of the code usefull for continuous reading!!!
        tmp = array(unpack(line, tmp), dtype = itype)
        tmp = tmp.reshape(mult, numc)[::skip,:]
        tmp = tmp.reshape(int(mult * numc / skip))
        D.append(array(tmp))
        tmp = bytearray(F.read(mult * numc * numb))
        if len(D) % 5000 == 0:
            dn = len(D) * len(D[-1]) * numb * skip / 1024 ** 3
            print('\r   %.2f of %.2f GB read, remaining runtime %d sec       ' % (dn, sz, (sz / dn - 1) * (time() - t)), end = ' ')
        '''
    print(' ')
    D = array(D)
    D = D.reshape(int(shape(D)[0]*shape(D)[1] / numc), numc)
    F.close()

    print('   done in %.1f sec' % (time() - t))
    return D

# write the L-CARD like binary file
def writeLCARD(D, name):                                                                                
    print(' - writing file ' + name) 
    W = open(name, 'wb')
    for p in range(shape(D)[0]):
        W.write(pack('%sd' % shape(D)[1], *D[p,:]))
    W.close()


# when user pushes the cross on the electrical plot window :)
def onClose(event):
    fig.canvas.mpl_disconnect(BPE)
    fig.canvas.mpl_disconnect( CE)
    close(fig)
    doTheJob()

#+++++++++++++++++++++++++#
#                         #
#       MAIN BODY         #
#                         #
#+++++++++++++++++++++++++#

# global name for input/output files
# 902 - first device, 881 - second device
try:
    pname = sys.argv[1]
    mode  = 0
except:
    root = tkinter.Tk()
    root.withdraw()
    pname = askopenfilename(filetypes = [('L-CARD dat files','*.dat'), ('All files','*.*')])
    root.destroy()

if '-902.dat' in pname or '-881.dat' in pname:
    pname = re.sub('-[0-9]{3}.dat', '', pname)
    mode = 0
else:
    root = tkinter.Tk()
    root.withdraw()
    showerror('ERROR','I need the name of the L-CARD *.dat file!!!\n\nAborted :(')
    root.destroy()
    print('I need the name of the L-CARD *.dat file')
    exit()

# file name and path
name = pname.split('/')[-1]
path = pname.replace(name, '')

# L-CARD frequency definitions
flist = os.popen('ls ' + path + name + '-EventsList-*.log 2>1 1>\dev\\null').read().split()
try:
    if flist != []:
        F = open(flist[0], 'r')
        line = F.readline().split()
        F.close()
        if line[0] == '#' and len(line) == 3 and line[2] == 'Hz':
            EFREQ = int(line[1])
    if 'EFREQ' not in locals():
        root = tkinter.Tk()
        root.withdraw()
        EFREQ = 1000 * askinteger('L-CARD Frequency', 'Please, provide the frequency in kHz', initialvalue = 20)
        root.destroy()
except:
    root = tkinter.Tk()
    root.withdraw()
    showerror('ERROR','L-CARD frequency was not accepted!\n\nAborted :(')
    root.destroy()
    exit()

# set global settings
set_GLOBALS()
indE = []

# L-CARD data files
D1 = readLCARD(path + name + '-902.dat', skip = SKIP)
D2 = readLCARD(path + name + '-881.dat', skip = SKIP)

# electro range selection!!!
cm = get_cmap('rainbow')
fig = figure(101)

plot(linspace( 0, len(D1) / EFREQ * SKIP, len(D1)), D1[:, 0], color = cm(0.000), lw = 0.7)
plot(linspace( 0, len(D1) / EFREQ * SKIP, len(D1)), D1[:, 7], color = cm(0.143), lw = 0.7)
plot(linspace( 0, len(D1) / EFREQ * SKIP, len(D1)), D1[:,18], color = cm(0.286), lw = 0.7)
plot(linspace( 0, len(D1) / EFREQ * SKIP, len(D1)), D1[:,21], color = cm(0.429), lw = 0.7)

plot(linspace( 0, len(D2) / EFREQ * SKIP, len(D2)), D2[:,10], color = cm(0.571), lw = 0.7)
plot(linspace( 0, len(D2) / EFREQ * SKIP, len(D2)), D2[:,13], color = cm(0.714), lw = 0.7)
plot(linspace( 0, len(D2) / EFREQ * SKIP, len(D2)), D2[:,24], color = cm(0.857), lw = 0.7)
plot(linspace( 0, len(D2) / EFREQ * SKIP, len(D2)), D2[:,31], color = cm(1.000), lw = 0.7)

xlabel('time, sec')
ylabel('voltage, V')
ax = gca()
yl = ax.get_ylim()
BPE = fig.canvas.mpl_connect('button_press_event', BorderLocation)
CE  = fig.canvas.mpl_connect('close_event', onClose)
show()

print(' - buy buy baby - baby is a good buy!!!')


