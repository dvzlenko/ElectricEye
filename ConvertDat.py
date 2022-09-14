#!/usr/bin/python3

import os, sys
import time as tm
import tkinter
from numpy import *
from struct import *
from tkinter.filedialog import askdirectory
from tkinter.messagebox import showerror

# directory to process
try:
    dirname = sys.argv[1]
    mode  = 0
except:
    root = tkinter.Tk()
    root.withdraw()
    dirname = askdirectory()
    root.destroy()

if not os.path.isdir(dirname):
    root = tkinter.Tk()
    root.withdraw()
    showerror('ERROR','Directory ' + dirname + ' does not exist!!!\n\nAborted :(')
    root.destroy()
    print('   ERROR\n   Directory ' + dirname + ' does not exist!!!\n\nProcess borted :(')

L1 = os.popen('ls ' + dirname + '/*902.dat 2>/dev/null').read().split()
L2 = os.popen('ls ' + dirname + '/*881.dat 2>/dev/null').read().split()
L3 = os.popen('ls ' + dirname + '/*440.dat 2>/dev/null').read().split()
flist = L1 + L2 + L3

print(' - processing directory ' + dirname)
if flist != []:
    for name in flist:    
        print(' - converting L-Card ' + name + ' file...' )
        t = tm.time()

        # itype definition
        # 16 kByte test sample to unpack and try to unpack
        num = 16 * 1024
        # types dictionary
        dct = {0 : float16, 1 : float32, 2 : float64}
        F = open(name, 'rb')
        tmp = F.read(num)
        F.close()
        var = []
        var.append(abs(array(unpack('%de' % (num / 2), tmp))).max())
        var.append(abs(array(unpack('%df' % (num / 4), tmp))).max())
        var.append(abs(array(unpack('%dd' % (num / 8), tmp))).max())
        #print(var)
        var = ((array(var) < 15) * (array(var) > 0.01)).nonzero()[0][0]
        itype = dct[var]
        print('   itype was automatically assigned to %s' % itype)


        if itype == float64:
            numc   =  32
            inumb  =  8
            onumb  =  4
            icline = '%sd' % numc
            ocline = '%sf' % numc
        
            # read and write
            os.system('mv ' + name + ' ' + name.replace('.dat','-float64.dat'))
            F = open(name.replace('.dat','-float64.dat'), 'rb')
            W = open(name.replace('.dat','-float32.dat'), 'wb')
        
            num = 0
            tmp1 = bytearray(F.read(numc * inumb))
            sz   = os.path.getsize(name.replace('.dat','-float64.dat')) / 1024**3
            while len(tmp1) == numc * inumb:
                tmp2 = unpack(icline, tmp1)
                W.write(pack(ocline, *tmp2))
                tmp1 = bytearray(F.read(numc * inumb))
                num += 1
                if num % 100000 == 0:
                    tmp3 = tm.time()
                    tmp4 = numc * inumb * num / 1024**3
                    print('\r%.1f GB of %.1f GB completed (%.1f MB/sec), %d sec remaining      ' % (tmp4,  sz, 1024 * tmp4 / (tmp3 - t), ((sz / tmp4) - 1) * (tmp3 - t)), end = '')
                    W.flush()
        
            F.close()
            W.close()
        
            tmp = tm.time()
            print('\n   done in %.1f sec (%.1f MB/sec)                                ' % (tmp - t, 1024 * sz / (tmp - t)))
        else:
            print('   file does not require conversion!!!')
else:
    print(' - directory ' + dirname + 'does not contain neigher *-902.dat, nor *-881.dat files')        

