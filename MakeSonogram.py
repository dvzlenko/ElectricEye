#!/usr/bin/python3

import os
import re
import sys
import time
import tkinter
from tkinter.filedialog import askopenfilename 
from tkinter.messagebox import askyesno, showerror
from tkinter.simpledialog import askinteger
from numpy import *
from matplotlib.pyplot import *
from scipy.io import wavfile

def set_GLOBALS():
    # GLOBALS
    global FRAME, DIV
    # Global Settings
    FRAME  =        10      # frame for sonorgam, Hz
    

# movavg function instead of the deprecated one
# num - frame length
# dim - dimension to average along for 2D arrays
def movavg(X, num, dim = 0):
    D = []
    if len(shape(X)) == 1:
        for p in range(len(X) - num + 1):
            D.append(mean(X[p:p+num]))
        return array(D)

    elif len(shape(X)) == 2:
        if dim == 0:
            for p in range(shape(X)[0] - num + 1):
                D.append(mean(X[p:p+num,:], 0))
            return array(D)

        elif dim == 1:
            for p in range(shape(X)[1] - num + 1):
                D.append(mean(X[:,p:p+num], 1))
            return array(D).T

        else:
            print(' !!! ERROR !!!\n    Wrong dimension for 2D array: %d' % dim)
            exit()
    else:
        print(' !!! ERROR !!!\n     movavg cannot digest tha array of the dimensionality greater than 2!!!')
        exit()


#+++++++++++++++++++++++++#
#                         #
#       MAIN BODY         #
#                         #
#+++++++++++++++++++++++++#

# global name for input/output files
try:
    fname = sys.argv[1]
    mode  = 0
except:
    root = tkinter.Tk()
    root.withdraw() 
    fname = askopenfilename(filetypes = [('WAV files', ['*wav','*.WAV']), ('All files','*.*')])
    root.destroy()

if 'wav' not in fname and 'WAV' not in fname:
    print('I need the name of the WAV file!!!')
    root = tkinter.Tk()
    root.withdraw() 
    showerror(title = 'ERG Error Message', message = 'The WAV file was not provided!!!\n\nExiting!!!')
    root.destroy()
    exit()

name  = re.sub('.wav', '', fname)
name  = re.sub('.WAV', '', fname)

# file name and path
name   = name.split('/')[-1]
path   = name.replace(name, '')

# globals
set_GLOBALS()

# read the sound
AFREQ, sound = wavfile.read(fname)

#print(' - filtering')
#tmp = movavg(sound, 1000)
#print(' - fouriering')
#SPG = specgram(sound, Fs = AFREQ, NFFT = 256, cmap = 'inferno')
#SPG = specgram(sound, Fs = AFREQ, NFFT = 256, cmap = 'binary', mode = 'magnitude')
SPG = specgram(sound, Fs = AFREQ, NFFT = int(AFREQ / FRAME), cmap = 'Greys', mode = 'magnitude')
ax = gca();
ax.cla()
pcolormesh(SPG[2], SPG[1], SPG[0]**2, cmap = 'Greys')
xlim(20, 160)
ylim(10, 5000)
xlabel('time, sec')
ylabel('frequency, Hz')
savefig('test_01.png')
show()

print(' - buy buy baby - baby is a good buy!!!')

