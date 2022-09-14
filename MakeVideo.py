#!/usr/bin/python3

import os
import re
import sys
import cv2
import time
import tkinter
from tkinter.filedialog import askopenfilename 
from tkinter.messagebox import askyesno, showerror
from struct import *
from numpy import *
from matplotlib.pyplot import *
from scipy.interpolate import *
from scipy.io import wavfile
from moviepy.editor import VideoFileClip
from sklearn.decomposition import PCA
#from lib_ELPH import *

def set_GLOBALS():
    global limP, limG, AV, SLOW, WINDOW, EFREQ, MODE, NCOMP, NCONT, LW, FL, DST, GAIN, ASPECT, THEME, TSIZE, ASIZE, LSIZE, SET_ZERO, BARR_AMP
    #
    # GLOBAL SETINGS
    #
    limG        =       0.5    # Time period flanking the continuous periods (sec)
    AV          =         1    # Number of frames for movavg
    SLOW        =      1000    # Factor to slow down the video during the pulse. It MUST be a multiple of EFREQ / VFREQ
    BARR_AMP    =   10 ** 4    # Full scale amplitude in MICROvolts!!! Default is 10**4 = 10 mV
    GAIN        =      1100    # Amplifiers' gain

    NCONT       =        30    # Number of Contours
    LW          =       5.0    # Line Width
    FL          =         8    # Focal Length of the lense used
    DST         =  -5.0e-06    # Distorsion correction constant

    ASPECT      =       1.8    # video frame aspect ratio       
    THEME       =     'DARK'   # can be DARK or LIGHT, defining the overall appearance of the pictures
    SET_ZERO    =      True    # True - for those, who requires extraction of the beginning of the oscillogramm! False is absolutely necessary for wave gymnotides

    TSIZE       =         32#22#32   # title text size
    ASIZE       =         24#18#24   # axis label text size
    LSIZE       =         20#16#20   # axis values text size

    WINDOW      =    int(10 * EFREQ / VFREQ / SLOW)     # Width of the window to mean the field in the moving-field video
                                                        # became automatically defined!!!!!



def cropBorders(img, crop = True):
    if crop:
        xlim(G[4,0], G[4,1] + int((G[4,1] - G[4,0]) / 2))
        ylim(G[4,3], G[4,2])
        fig.set_size_inches((G[4,1] - G[4,0]) * (3 / 2) / my_dpi, (G[4,3] - G[4,2]) / my_dpi)
    else:
         fig.set_size_inches(PW / my_dpi, PH / my_dpi)
    #fig.set_size_inches(img.get_size()[1] / my_dpi, img.get_size()[0] / my_dpi)
    img.axes.get_xaxis().set_visible(False)
    img.axes.get_yaxis().set_visible(False)
    #ax.set_axis_off()
    tight_layout(pad = 0)


# movmed function instead of the deprecated one
# num - frame length
# dim - dimension to average along for 2D arrays
def movmed(X, num, dim = 0):
    D = []
    if len(shape(X)) == 1:
        for p in range(len(X) - num + 1):
            D.append(median(X[p:p+num]))
        return array(D)

    elif len(shape(X)) == 2:
        if dim == 0:
            for p in range(shape(X)[0] - num + 1):
                D.append(median(X[p:p+num,:], 0))
            return array(D)

        elif dim == 1:
            for p in range(shape(X)[1] - num + 1):
                D.append(median(X[:,p:p+num], 1))
            return array(D).T

        else:
            print(' !!! ERROR !!!\n    Wrong dimension for 2D array: %d' % dim)
            exit()
    else:
        print(' !!! ERROR !!!\n     movavg cannot digest tha array of the dimensionality greater than 2!!!')
        exit()

#
# Read the L-CARD data file 
# num    - number of the channels => data columns in the file
# numb   - number of bytes per a value
# type   - type of the data in the returned array
# full   - if true, it will read the full data, otherwise - only every 10-th frame
# lim    - limits to read from the file, frame numbers
# reading of the full file could be done if SKIP global variable equals to unity
# 

def readLCARD(fname, numc = 32, itype = False, otype = float16, lim = []):
    print(' - reading ' + fname + ' L-CARD data file')

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

    # line of decimals to unpack
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
        print(' itype must be one of the list: float16, float32, or float 64')
        exit()
    
    F = open(fname, 'rb')
    t = time.time()
    D = []
    # skippind frames until the start point reached
    #print('number of bytes to seek: %d' % int(lim[0] * numb * numc))
    F.seek(int(lim[0] * numb * numc), 0)
    # read the data
    for p in range(int((lim[1] - lim[0]))):
        tmp = bytearray(F.read(numc * numb))
        tmp = unpack(line, tmp)
        D.append(array(tmp, dtype = otype))
    D = array(D)
    F.close()

    print('   done in %.1f msec' % ((time.time() - t) * 1000))
    return D


# write the L-CARD like binary file
def writeLCARD(D, name):                                                                                
    print(' - writing file ' + name) 
    W = open(name, 'wb')
    for p in range(shape(D)[0]):
        W.write(pack('%sd' % shape(D)[1], *D[p,:]))
    W.close()

#+++++++++++++++++++++++++#
#                         #
#       MAIN BODY         #
#                         #
#+++++++++++++++++++++++++#

# global name files
# 902 - first device, 881 - second device
try:
    pname = sys.argv[1]
    mode  = 0
except:
    root = tkinter.Tk()
    root.withdraw()
    pname = askopenfilename(filetypes = [('Events List files', '*EventsList*log'), ('All files','*.*')])
    root.destroy()

if 'EventsList' in pname:
    elogname = pname
    dname  = re.sub('-EventsList', '', pname)
    dname  = re.sub('.log', '', dname)
    pname  = re.sub('-EventsList-.+', '', pname)
else:
    print('I need the name of the ERG EventsList log-file!!!')
    root = tkinter.Tk()
    root.withdraw()
    showerror(title = 'ERG Error Message', message = 'The EventsList file was not provided!!! Exiting!!!')
    root.destroy()
    exit()

# load the EventsList file and check for the EFREQ in it
try:
    F = open(elogname, 'r')
    lines = F.readlines()
    line  = lines[0].split()
    F.close()
    if len(line) == 3:
        if line[0] == '#' and line[2] == 'Hz':
            EFREQ = int(line[1])
    else:
        root = tkinter.Tk()
        root.withdraw()
        EFREQ = 1000 * askinteger('L-CARD Frequency', 'Please, provide the frequency in kHz', initialvalue = 20)
        root.destroy()
        # update the EventsList file
        os.system('cp ' + elogname + ' ' + elogname + '.bkp')
        W = open(elogname, 'w')
        W.write('# %d Hz\n' % EFREQ)
        W.writelines(lines)
        W.close()
except:
    root = tkinter.Tk()
    root.withdraw()
    showerror('ERROR','L-CARD frequency was not accepted!\n\nExecution Aborted :(')
    root.destroy()
    exit()

tmp = loadtxt(elogname)
if len(shape(tmp)) == 1:
    indE = array([tmp[1]])
    dura = array([tmp[3]])
else:
    indE = tmp[:,1]
    dura = tmp[:,3] 

# file name and path
name  = pname.split('/')[-1]
path  = pname.replace(name, '')
dname = dname.split('/')[-1]
dname  = re.sub(name + '-', '', dname)


#print(name, path, dname)

# video file handler
cap         = cv2.VideoCapture(path + name + '.avi')
VFREQ       = cap.get(cv2.CAP_PROP_FPS)                  # Frequency for the video file (Hz)
PW          = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))     # Real frame width (px) 
PH          = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))    # Frame height (px)
G           = loadtxt(path + name + '.rect')

# GLOBALS!!!
set_GLOBALS()

# some cheks
if (EFREQ / VFREQ) % SLOW:
    root = tkinter.Tk()
    root.withdraw()
    showerror('ERROR','(EFREQ / VFREQ) % SLOW != 0')
    root.destroy()
    exit()

# trying either read the alignment from right audio channel or *.align file
if not os.path.isfile(path + name + '.wav'):
    videoclip = VideoFileClip(path + name + '.avi')
    audioclip = videoclip.audio
    if audioclip:
        audioclip.write_audiofile(path + name + '.wav')
        AFREQ, sound = wavfile.read(path + name + '.wav')
else:
    AFREQ, sound = wavfile.read(path + name + '.wav')

# if sound was found in the video file
if 'sound' in locals():
    # counter signal MUST be in the right channel!!!!
    soundamp = 10 * sound[::100,1].std() 
    tmp     = (sound[:,0] > 100)
    apeaks  = (tmp[1:] < tmp[:-1]).nonzero()[0] # each at the end of a milisecond, i.e. after EFREQ / 1000 LCARD data samples collected
    aperiod = (apeaks[1:] - apeaks[:-1]).mean()
    if ((aperiod / AFREQ) - 0.001) > 0.0000001:
        print('The period of the meandr in the right audio channel is %.6f msec, which is abnormaly differs from 1 msec!\n   trying to read the *.align file')
        try:
            ALIGN = loadtxt(path + name + '.align')   # Linear regression coefficients for L-CARD to L-CARD and to video frame alignment and conversion
        except:
            print('I did not found either an appropriate pulse-series in the audio-track of the video file or the corresponding *.align file!!!!\n   exiting... :(')
            exit()
    else:
        enum    = len(apeaks)  # Number of the full 1 kHz pulses registered. 
                               # Each equals 2000 2MHz pulses and corresponds to EFREQ / 1000 LCARD samples. 
                               # Real sample number is always smaller!!!!
        start   = VFREQ * (apeaks[0] - aperiod) / AFREQ  # exact beginning of the LCARD series in the video frames scale
        finish  = VFREQ * apeaks[-1] / AFREQ             #       exact end of the LCARD series in the video frames scale
        cf      = polyfit([0, EFREQ * enum / 1000], [start, finish], 1)
        ALIGN   = array([[1.0, 0.0],[cf[0], cf[1]]])
        # create the ALIGN file. Just for any case!!!
        if not os.path.isfile(path + name + '.align'):
            W = open(path + name + '.align', 'w')
            W.write('# This file was created based of the 1 kHz meandr from the audio channel of the movie\n')
            W.write('# The parameters were:\n')
            W.write('# AFREQ   = %10d\n# VFREQ   = %10d\n# EFREQ   = %10d\n' % (AFREQ, VFREQ, EFREQ))
            W.write('# enum    = %10d\n' % enum)
            W.write('# aperiod = %10.6f\n' % (1000 * aperiod / AFREQ))
            W.write('# start   = %10.3f\n' % start)
            W.write('# finish  = %10.3f\n' % finish)
            W.write('%30.24f %30.24f\n' % (1, 0))
            W.write('%30.24f %30.24f\n' % (cf[0], cf[1]))
            W.close()
else:
    try:
        ALIGN = loadtxt(path + name + '.align')   # Linear regression coefficients for L-CARD to L-CARD and to video frame alignment and conversion
    except:
        print('I did not found either any audio-track in the video file or the corresponding *.align file!!!!\n   exiting... :(')
        exit()

# fill the indV variable with precise video coordinates!!!
indT = indE / EFREQ
indV = ALIGN[1,0] * indE + ALIGN[1,1]

# real video frame rate, suposing the LCARD as ideal
# I need it for sound production
realVFREQ = EFREQ * ALIGN[1,0]
print(' - real video frame rate is %.5f frames per second' % (realVFREQ))
if (realVFREQ / 50 < 0.9) and (realVFREQ / 25 < 0.9):
    root = tkinter.Tk()
    root.withdraw()
    showerror('ERROR','Real video frame rate is %.4f fps, which implies a mistake in L-CARD frequency preset\n\nExecution aborted :(' % realVFREQ)
    root.destroy()
    print('  !!!ERROR!!!!\n\n   Real video frame rate is %.4f fps, which implies a mistake in L-CARD frequency preset\n\nExecution aborted :(' % realVFREQ)
    exit()

# sceen dpi definition
root = tkinter.Tk()
root.withdraw()
my_dpi = root.winfo_fpixels('1i')
#print('my_dpi = %d' % my_dpi)
root.update()
root.destroy()

# Distortion correction stuff
cam = eye(3)
cam[0,2]        = PW / 2.0      # define center x
cam[1,2]        = PH / 2.0      # define center y
cam[0,0]        = FL            # define focal length x
cam[1,1]        = FL            # define focal length y
distCoeff       = zeros((4,1))
distCoeff[0]    = DST           # distorsion coefficient  

# Perspective correction matrix
M = cv2.getPerspectiveTransform(float32(G[:4,:2]), float32(G[:4,2:]))

# linspaces for contour and imshow
X  = linspace(G[0,2], G[1,2], 8)
Y  = linspace(G[0,3], G[3,3], 8)
XX = linspace(G[0,2], G[1,2], int(G[1,2] - G[0,2] + 1))
YY = linspace(G[0,3], G[3,3], int(G[3,3] - G[0,3] + 1))

#+++++++++++++++++++++++++
#
# do the job stuff!!!
#
#+++++++++++++++++++++++++
  
indES = []  # pulses splitted among the regions in electro space
indVS = []  # pulses splitted among the regions in video space
indAS = []  # pulses splitted among the regions in audio space
limV  = []  # borders of the video in video frames space
limE  = []  # borders of the event itself in electro frames space
limP  = []  # duration of the pulse for PCA and for plotting

contCM = get_cmap('seismic')
barrCM = get_cmap('YlGn')

# select the continuous regions for the continuous video files with multiple events
#print(dura)
ind = (indT[1:] - indT[:-1] > limG).nonzero()[0]
if len(ind):
    limV.append([indV[0] - limG * VFREQ, indV[ind[0]] + limG * VFREQ])
    limE.append([indE[0] - limG * EFREQ, indE[ind[0]] + limG * EFREQ])
    indES.append(indE[ : ind[0] + 1])
    indVS.append(indV[ : ind[0] + 1])
    limP.append( dura[ : ind[0] + 1])
    for p in range(len(ind)-1):
        limV.append([indV[ind[p] + 1] - limG * VFREQ, indV[ind[p+1]] + limG * VFREQ])
        limE.append([indE[ind[p] + 1] - limG * EFREQ, indE[ind[p+1]] + limG * EFREQ])
        indES.append(indE[ind[p] + 1 : ind[p+1] + 1])
        indVS.append(indV[ind[p] + 1 : ind[p+1] + 1])
        limP.append( dura[ind[p] + 1 : ind[p+1] + 1])
    limV.append([indV[ind[-1] + 1] - limG * VFREQ, indV[-1] + limG * VFREQ])
    limE.append([indE[ind[-1] + 1] - limG * EFREQ, indE[-1] + limG * EFREQ])
    indES.append(indE[ind[-1] + 1 : ])
    indVS.append(indV[ind[-1] + 1 : ])
    limP.append(dura[ind[-1] + 1 : ])
else:
    limP.append(dura)
    limV.append([indV[0] - limG * VFREQ, indV[-1] + limG * VFREQ])
    limE.append([indE[0] - limG * EFREQ, indE[-1] + limG * EFREQ])
    indES.append(indE)
    indVS.append(indV)
'''
print(' indT:')
print(indT)
print(' indE')
print(indE)
print(' indV')
print(indV)
print(' indES')
print(indES)
print(' indVS')
print(indVS)
print(' limP')
print(limP)
print(' limV')
print(limV)
print(' limE')
print(limE)
#exit()
'''

# iterator of the selected limG sec regions, expanded if necessary
for r in range(len(limV)):
    # directories and log file for the memory!!!
    if not os.path.isdir(path + name):
        os.mkdir(path + name)
    if not os.path.isdir(path + name + '/' + dname + '-AVI'):
        os.mkdir(path + name + '/' + dname + '-AVI')
    #W = open(path + name + '/' + dname + '-AVI/' + name + '.%04dsec.log', 'w')

    mlim   = []   # limits for the field plotting
    mfld   = []   # the fields themselves
    mint   = []   # limits for the vertical lines plotting
    cnr    = []   # corner where to draw the field
    pca    = []
    comp   = []
    coef   = []
    #W.write(' - processing interval between %.3f and %.3f sec\n' % (limV[r][0] / VFREQ, limV[r][1] / VFREQ))
    print(' - processing interval between %.3f and %.3f sec' % (limV[r][0] / VFREQ, limV[r][1] / VFREQ))
    # global data
    readlim = [indES[r][0] - limG * EFREQ, indES[r][-1] + limG * EFREQ]
    DD1 = readLCARD(path + name + '-902.dat', lim = readlim)
    DD2 = readLCARD(path + name + '-881.dat', lim = readlim)
    if len(DD1) > len(DD2):
        print(' - concatenation:')
        print('   shape DD1 = (%d, %d); shape DD2 = (%d, %d);' % (shape(DD1)[0], shape(DD1)[1], shape(DD2)[0], shape(DD2)[1]))
        DD1 = DD1[:len(DD2),:]
        print('   shape DD1 = (%d, %d); shape DD2 = (%d, %d);' % (shape(DD1)[0], shape(DD1)[1], shape(DD2)[0], shape(DD2)[1]))
    if len(DD1) < len(DD2):
        print(' - concatenation:')
        print('   shape DD1 = (%d, %d); shape DD2 = (%d, %d);' % (shape(DD1)[0], shape(DD1)[1], shape(DD2)[0], shape(DD2)[1]))
        DD2 = DD2[:len(DD1),:]       
        print('   shape DD1 = (%d, %d); shape DD2 = (%d, %d);' % (shape(DD1)[0], shape(DD1)[1], shape(DD2)[0], shape(DD2)[1]))
    DG  = -concatenate((DD1, DD2), 1)

    mDD = []
    # iterate per the pulses inside the region r
    for p in range(len(indES[r])):
        mfld.append([])
        mint.append([])
        # extract the full L-CARD data for the selected region
        readlim = [int(indES[r][p] - limP[r][p] * EFREQ / 2000 - (AV + WINDOW - 1) / 2), 
                   int(indES[r][p] + limP[r][p] * EFREQ / 2000 + (AV + WINDOW - 1) / 2)]
        DD1 = readLCARD(path + name + '-902.dat', lim = readlim)
        DD2 = readLCARD(path + name + '-881.dat', lim = readlim)
        if len(DD1) > len(DD2):
            print(' - concatenation:')
            print('   shape DD1 = (%d, %d); shape DD2 = (%d, %d);' % (shape(DD1)[0], shape(DD1)[1], shape(DD2)[0], shape(DD2)[1]))
            DD1 = DD1[:len(DD2),:]
            print('   shape DD1 = (%d, %d); shape DD2 = (%d, %d);' % (shape(DD1)[0], shape(DD1)[1], shape(DD2)[0], shape(DD2)[1]))
        if len(DD1) < len(DD2):
            print(' - concatenation:')
            print('   shape DD1 = (%d, %d); shape DD2 = (%d, %d);' % (shape(DD1)[0], shape(DD1)[1], shape(DD2)[0], shape(DD2)[1]))
            DD2 = DD2[:len(DD1),:]       
            print('   shape DD1 = (%d, %d); shape DD2 = (%d, %d);' % (shape(DD1)[0], shape(DD1)[1], shape(DD2)[0], shape(DD2)[1]))
        
        # negative sign is necessary, as the amplifier inverts the signal!!!
        DD  = -concatenate((DD1, DD2), 1)
        # SMOOOOOOOOOOOOOOOOOOOOTHING!!!!! 
        DD  = movmed(DD, AV)
        # OFFSET REMOVING!!!!! For everybody, including wave fish!!!
        DD = DD - DD[:10,:].mean(0)
        #
        mDD.append(DD)
        # PCA WHOLE
        pca.append(PCA(n_components = 7))
        comp.append(pca[-1].fit_transform(DD))
        coef.append(pca[-1].components_)
        # TODO: REMEMBER ME!!!!
        if SET_ZERO:
            comp[-1] = comp[-1] - comp[-1][:10,:].mean(0)
            #comp[-1][:,0] = comp[-1][:,0] - comp[-1][:10,0].mean()
        # field itself
        lims = []
        #print(len(DD))
        # iterate per fragmants inside the pulse for the field movie
        for q in range(0, len(DD) - WINDOW, int(EFREQ / VFREQ / SLOW)):
            # FIELD DISTRIBUTION: 1   9  17  25    33  41  49  57   
            #                     2  10  18  26    34  42  50  58  
            #                     3  11  19  27    35  43  51  59  
            #                     4  12  20  28    36  44  52  60  
            #                     5  13  21  29    37  45  53  61  
            #                     6  14  22  30    38  46  54  62  
            #                     7  15  23  31    39  47  55  63  
            #                     8  16  24  32    40  48  56  64  
            #print('    q : q + WINDOW = %d : %d' % (q, q + WINDOW))
            # matrix must be filled in the reverse order, as in the picture the origin is in the upper-left corner, while in the plot it in the lower-left one!
            F = zeros([8,8])
            for m in range(8):
                for n in range(8):
                    F[n,m] = DD[q : q + WINDOW, 8 * m + n].mean()
            lims.append(F.max() - F.min())
            mfld[-1].append(F)
            mint[-1].append([1000 * q / EFREQ , 1000 * (q + WINDOW) / EFREQ])
        mlim.append(array(lims).max())
        #print('IndES: r = %d, p = %d, len(mfld[-1]) = %d, q = %d' % (r, p, len(mfld[-1]), q))
    #exit()
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    #
    # the video file: limG prior the events sequence and limG after it !!!
    #
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    print(' - video crop started')
    #wt = int((3 / 2) * (G[4,1] - G[4,0]))
    #ht = int(G[4,3] - G[4,2])
    #print(wt, ht)
    outF = cv2.VideoWriter(path + name + '/' + dname + '-AVI/' + name + '.%04dsec.VidF.avi' % (limV[r][0] / VFREQ), cv2.VideoWriter_fourcc(*'X264'), VFREQ, (int(ASPECT * PH), PH))
    outB = cv2.VideoWriter(path + name + '/' + dname + '-AVI/' + name + '.%04dsec.VidB.avi' % (limV[r][0] / VFREQ), cv2.VideoWriter_fourcc(*'X264'), VFREQ, (PW, PH))

    # constructing the parent figure that it will modify!!!
    # settings
    #TSIZE = 22  # title text size
    #ASIZE = 14  # axis label text size
    #LSIZE = 10  # axis values text size

    # main picture creation stuff
    # theme, dakr or light!!!
    if THEME == 'DARK':
        bcolor = 'black'
        lcolor = 'white'
    elif THEME == 'LIGHT':
        bcolor = 'white'
        lcolor = 'black'
    else:
        print('Unknown theme: ' + theme + '\n   exiting :(')
        exit()

    #fig = figure(666, constrained_layout=True, dpi = my_dpi)
    fig = figure(666, dpi = my_dpi)
    fig.set_facecolor(bcolor)
    gs  = fig.add_gridspec(4, 73)
    # intensity barr
    ax0 = fig.add_subplot(gs[:,2])
    ax0.set_facecolor(bcolor)
    ax0.tick_params(axis='both', labelsize = ASIZE, labelcolor = lcolor)
    ax0.spines['top'].set_color(bcolor)
    ax0.spines['right'].set_color(bcolor)
    ax0.spines['bottom'].set_color(bcolor)
    ax0.spines['left'].set_color(bcolor)
    ax0.set_axis_off()
    # video frame + field distribution
    ax1 = fig.add_subplot(gs[:,2:49])
    ax1.set_axis_off()
    ax1.set_xlim(G[4,0], G[4,1])
    ax1.set_ylim(G[4,3], G[4,2])
    # full oscillogramm plot
    ax2 = fig.add_subplot(gs[0, 49:])
    ax2.set_facecolor(bcolor)
    ax2.tick_params(axis='both', labelsize = LSIZE, labelcolor = lcolor)
    ax2.spines['top'].set_color(bcolor)
    ax2.spines['right'].set_color(bcolor)
    ax2.spines['bottom'].set_color(lcolor)
    ax2.spines['bottom'].set_linewidth(2)
    ax2.spines['left'].set_color(lcolor)
    ax2.spines['left'].set_linewidth(2)
    # signal oscillogramm plot
    ax3 = fig.add_subplot(gs[1, 49:])
    ax3.set_facecolor(bcolor)
    ax3.tick_params(axis='both', labelsize = LSIZE, labelcolor = lcolor)
    ax3.spines['top'].set_color(bcolor)
    ax3.spines['right'].set_color(bcolor)
    ax3.spines['bottom'].set_color(lcolor)
    ax3.spines['bottom'].set_linewidth(2)
    ax3.spines['left'].set_color(lcolor)
    ax3.spines['left'].set_linewidth(2)
    ax3.set_axis_off()  
    # first PCA component plot
    ax4 = fig.add_subplot(gs[2, 49:])
    ax4.set_facecolor(bcolor)
    ax4.tick_params(axis='both', labelsize = LSIZE, labelcolor = lcolor)
    ax4.spines['top'].set_color(bcolor)
    ax4.spines['right'].set_color(bcolor)
    ax4.spines['bottom'].set_color(lcolor)
    ax4.spines['bottom'].set_linewidth(2)
    ax4.spines['left'].set_color(lcolor)
    ax4.spines['left'].set_linewidth(2)
    ax4.set_axis_off()  
    # second PCA component or sound pressure plot
    ax5 = fig.add_subplot(gs[3, 49:])
    ax5.set_facecolor(bcolor)
    ax5.tick_params(axis='both', labelsize = LSIZE, labelcolor = lcolor)
    ax5.spines['top'].set_color(bcolor)
    ax5.spines['right'].set_color(bcolor)
    ax5.spines['bottom'].set_color(lcolor)
    ax5.spines['bottom'].set_linewidth(2)
    ax5.spines['left'].set_color(lcolor)
    ax5.spines['left'].set_linewidth(2)
    ax5.set_axis_off()  

    # sort the curves from min to max for proper coloring!!!!    
    #tmp = abs(DG).max(0) * ((-1) * (DG.max(0) > -DG.min(0)) + (+1) * (DG.max(0) < -DG.min(0)))
    #tmp = tmp.argsort()
    #DG  = DG.take(tmp, 1)

    # plot full oscillogramm
    # scaling
    if abs(DG).max() > 0.1:
        #foL = ax2.plot(tm, dd, '-k', lw = 0.2)
        DG = (10**3) * DG / GAIN
        foY = ax2.set_ylabel('voltage, mV', size = ASIZE, color = lcolor)
    else:
        DG = (10**6) * DG / GAIN
        #foL =ax2.plot(tm, dd, '-k', lw = 0.2) 
        foY = ax2.set_ylabel('voltage, $\mu$V', size = ASIZE, color = lcolor)
    # plotting
    foL = []
    tm  = linspace(0, len(DG) / EFREQ, int(len(DG[::50,:])))
    #cm  = get_cmap('bwr')
    # uncomment here to switch on the oloring!!!
    #for p in range(shape(DG)[1]):
    #    foL.append(ax2.plot(tm, DG[::50, p], '-', lw = 0.5, color = (array(cm(1 - p / (shape(DG)[1] - 1))) / 2 + 0.5))[0])
    foL = ax2.plot(tm, DG[::50, ::4], '-', lw = 0.5, color = lcolor)
    # the rest of        
    foX = ax2.set_xlabel('time, sec', size = ASIZE, color = lcolor)
    spd = ax2.set_title('\n Speed: x1', size = TSIZE, color = lcolor)
    ddmin = DG[::50, ::4].min()
    ddmax = DG[::50, ::4].max()


    #foL = []
    #dd  = []
    #tm  = linspace(0, len(DG) / EFREQ, int(len(DG[::50,:])))
    # here I'm filtering the curves that fluctuate and makes the ocsillogramm awful
    #cm = get_cmap('rainbow')
    #for p in range(shape(DG)[1]):
    #    fltr = []
    #    fltr.append( DG[:int((limG - 0.1) * EFREQ) ,p].max())
    #    fltr.append(-DG[:int((limG - 0.1) * EFREQ), p].min())
    #    fltr.append( DG[-int((limG - 0.1) * EFREQ):,p].max())
    #    fltr.append(-DG[-int((limG - 0.1) * EFREQ):,p].min())
    #    if max(fltr) < 0.2:
    #        dd.append(1000 * DG[::50, p] / GAIN)
    #dd = array(dd).T
    #foL.append(ax2.plot(tm, dd, '-k', lw = 0.2)[0])

    # target frames!!!
    vfr  = 0
    efr  = 0
    nump = 0
    yes = []
    tt = time.time()
    #
    # MEGA COMMENT!!!!
    # indV - absolute video frame number
    # vfr  - video frame number inside the short video under creation: limV[0] + vfr - whould be the real frame number in the whole video
    # efr  - electro frame inside the pulse
    # nume - number of electro frames per current video frame
    # nump - number of pulse inside limES[r]
    #
    cap.set(cv2.CAP_PROP_POS_FRAMES, limV[r][0] + vfr)
    while vfr < int(limV[r][1] - limV[r][0]):
    #for p in range(5):
        # pulse and vframe boundaries
        lpb = indES[r][nump] - EFREQ * limP[r][nump] / 2000    #  left pulse boundary
        rpb = indES[r][nump] + EFREQ * limP[r][nump] / 2000    # right pulse boundary
        lfb =  vfr      * EFREQ / VFREQ + limE[r][0]   #  left frame boundary
        rfb = (vfr +1 ) * EFREQ / VFREQ + limE[r][0]   #  left frame boundary

        # if the begining of a new pulse is in the previous vframe!!!
        if (lpb > lfb) or nume or (nump + 1 == len(indES[r])) or stupidflag:
            # get and process a curent video frame
            OK, frame = cap.read()
            outB.write(frame)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            vfr -= 1
            lfb =  vfr      * EFREQ / VFREQ + limE[r][0]   #  left frame boundary
            rfb = (vfr +1 ) * EFREQ / VFREQ + limE[r][0]   #  left frame boundary

        #print(' vfr %d' % vfr)
        print(' vfr %d, lfb = %d, rfb = %d, lpb = %d, rpb = %d, nump = %d, efr = %d' % (vfr, lfb, rfb, lpb, rpb, nump, efr))
        #W.write(' vfr %d, lfb = %d, rfb = %d, lpb = %d, rpb = %d, nump = %d\n' % (vfr, lfb, rfb, lpb, rpb, nump))

        # To decide, if the field should be plotted
        stupidflag = False
        if lfb >= lpb and rfb <= rpb:
            nume = nume + SLOW
            print(' - frame inside the pulse, num = %d' % (nume - efr))
            #W.write(' - frame inside the pulse, num = %d\n' % (nume - efr))
        elif lfb <= lpb and rfb >= rpb:
            nume = nume + len(mfld[nump])
            print(' - pulse inside the frame, num = %d' % (nume - efr))
            #W.write(' - pulse inside the frame, num = %d\n' % (nume - efr))
        elif lfb <= lpb and rfb >= lpb:
            nume = int(SLOW * (rfb - lpb) / (rfb - lfb))
            # TODO: make it a bit more clear!!!
            if nume == 0:
                stupidflag = True
            print(' - start of a new pulse, num = %d' % (nume - efr))
            #W.write(' - start of a new pulse, num = %d\n' % (nume - efr))
        elif lfb < rpb and rfb >= rpb:
            nume = nume + int(SLOW * (rpb - lfb) / (rfb - lfb))
            print(' - end of the pulse, num = %d' % (nume - efr))
            #W.write(' - end of the pulse, num = %d\n' % (nume - efr))
        else:
            nume = 0
            stupidflag = False
        #print('nume = %d, len(fld) = %d' % (nume, len(mfld[nump])))


        # distortion and perspective
        dst  = cv2.undistort(frame, cam, distCoeff)
        warp = cv2.warpPerspective(dst, M, (PW, PH))

        # Draw the frame. It MUST be done after all of the drawings. Otherwise, the crop will not succide!!!
        img = ax1.imshow(warp, cmap = 'gray')
        #print('\r   %.2f of %.2f seconds done, remaining runtime %d sec       ' % (vfr / VFREQ, (limV[r][1] - limV[r][0]) / VFREQ, ((limV[r][1] - limV[r][0]) / vfr - 1) * (time.time() - tt)), end = ' ')

        # plot the corresponding vertical margins in long oscillogram
        rln11, = ax2.plot([ vfr      / VFREQ,  vfr      / VFREQ], [ddmin, ddmax], '--r', lw = 2.0)
        rln12, = ax2.plot([(vfr + 1) / VFREQ, (vfr + 1) / VFREQ], [ddmin, ddmax], '--r', lw = 2.0)

            
        if nume:
            #print('nume = %d, len(fld) = %d' % (nume, len(mfld[nump])))

            # side barr with the intensity
            ax0.set_axis_on()
            ax0.set_xlim(0,1)
            ax0.set_xticks([])
            ax0.set_ylim(1, BARR_AMP)
            #ax0.set_yticks([0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000])
            #ax0.set_yticklabels(['10$^4$','10$^3$','10$^2$','10','0', '-10', '-10$^2$', '-10$^3$', '-10$^4$'])
            ax0.set_ylabel('signal amplitude, $\mu$V', size = ASIZE, color = lcolor)
            ax0.tick_params(axis='both', labelsize = ASIZE, labelcolor = lcolor)
            ax0.set_yscale('log')
            # speed
            spd = ax2.set_title('\n speed: x1/%d' % SLOW, size = TSIZE, color = lcolor)

            # draw the short oscillogram
            # sort the curves from min to max for proper coloring!!!!    
            tmp = abs(mDD[nump]).max(0) * ((-1) * (mDD[nump].max(0) > -mDD[nump].min(0)) + (+1) * (mDD[nump].max(0) < -mDD[nump].min(0)))
            tmp = tmp.argsort()
            mDD[nump]  = mDD[nump].take(tmp, 1)
            # draw
            ax3.set_axis_on()
            ax3.tick_params(axis='both', labelsize = ASIZE, labelcolor = lcolor)
            osX = linspace(0, 1000 * len(mDD[nump]) / EFREQ, len(mDD[nump]))
            if abs(mDD[nump]).max() > 0.1:
                osY = (10**3) * mDD[nump] / GAIN 
                ax3.set_ylabel('voltage, mV', size = ASIZE, color = lcolor)
            else:
                osY = (10**6) * mDD[nump] / GAIN
                ax3.set_ylabel('voltage, $\mu$V', size = ASIZE, color = lcolor)
            # uncomment here to switch on the coloring!!!
            foS = []
            for p in range(shape(mDD[nump])[1]):
                foS.append(ax3.plot(osX, osY[:,p], '-', lw = 0.5, color = (array(contCM(1 - p / (shape(mDD[nump])[1] - 1))) / 2 + 0.5))[0])
            #foS = ax3.plot(osX, osY, '-', lw = 0.3, color = lcolor)
            # the rest
            #ax3.tick_params(axis='both', labelsize = LSIZE, labelcolor = lcolor)
            ax3.set_xlabel('time, msec', size = ASIZE, color = lcolor)
            mDDmin = osY.min()
            mDDmax = osY.max()

            # draw first PCA component
            ax4.set_axis_on()
            ax4.tick_params(axis='both', labelsize = ASIZE, labelcolor = lcolor)
            sgn   = int(max(abs(comp[nump][:,0])) == max(comp[nump][:,0])) - int(max(abs(comp[nump][:,0])) == -min(comp[nump][:,0]))
            if abs(coef[nump][0,:].max() * comp[nump][:,0]).max() > 0.1:
                #yPCA = sgn * (10**3) * abs(coef[nump][0,:]).max() * (comp[nump][:,0] - comp[nump][:10,0].mean()) / GAIN
                yPCA = sgn * (10**3) * abs(coef[nump][0,:]).max() * comp[nump][:,0] / GAIN
                ax4.set_ylabel('voltage, mV', size = ASIZE, color = lcolor)
            else:
                yPCA = sgn * (10**6) * abs(coef[nump][0,:]).max() * comp[nump][:,0] / GAIN
                ax4.set_ylabel('voltage, $\mu$V', size = ASIZE, color = lcolor)                
            #ax4.tick_params(axis='both', labelsize = LSIZE, labelcolor = lcolor)
            ax4.set_xlabel('time, msec', size = ASIZE, color = lcolor)
            xPCA = linspace(0, 1000 * len(comp[nump]) / EFREQ, len(comp[nump]))
            foP = ax4.plot(xPCA, yPCA, '-', lw = 2.0, color = '#0099ff')
            yPCAmin = yPCA.min()
            yPCAmax = yPCA.max()

            # draw sound if any
            if 'sound' in locals():
                # start and finish of the piece of the sound
                # precise video frame converted to the audio frame
                tmp = indES[r][nump] - limP[r][nump] * EFREQ / 2000 - WINDOW / 2
                ss  = round((ALIGN[1,0] * tmp + ALIGN[1,1]) * (AFREQ / VFREQ))
                tmp = indES[r][nump] + limP[r][nump] * EFREQ / 2000 + WINDOW / 2
                sf  = round((ALIGN[1,0] * tmp + ALIGN[1,1]) * (AFREQ / VFREQ))

                xSnd = linspace(0, 1000 * (sf - ss) / AFREQ, sf - ss)
                ySnd = sound[ss:sf, 1]
                foS = ax5.plot(xSnd, ySnd, '-', lw = 1.0, color = '#00dd00')
                if (ax5.get_ylim()[1] - ax5.get_ylim()[0]) < soundamp:
                    yl = [-10 * sound[:,1].std(), +10 * sound[:,1].std()]
                else:
                    yl = ax5.get_ylim()
                #
                ax5.set_axis_on()
                ax5.tick_params(axis='both', labelsize = ASIZE, labelcolor = lcolor)
                ax5.set_ylim(yl)
                ax5.set_xlabel('time, msec', size = ASIZE, color = lcolor)
                ax5.set_ylabel('sound pressure, a.u.', size = ASIZE, color = lcolor)
                #
                Smin = sound[:,1].min()
                Smax = sound[:,1].max()
           
            while efr < nume:
                print(' vfr %d; efr %d' % (vfr, efr))       
                #W.write(' vfr %d; efr %d\n' % (vfr, efr))
                # red lines
                rln21, = ax3.plot([mint[nump][efr][0], mint[nump][efr][0]], [mDDmin,   mDDmax], '--r', lw = 2.0)
                rln22, = ax3.plot([mint[nump][efr][1], mint[nump][efr][1]], [mDDmin,   mDDmax], '--r', lw = 2.0)
                rln31, = ax4.plot([mint[nump][efr][0], mint[nump][efr][0]], [yPCAmin, yPCAmax], '--r', lw = 2.0)
                rln32, = ax4.plot([mint[nump][efr][1], mint[nump][efr][1]], [yPCAmin, yPCAmax], '--r', lw = 2.0)
                if 'sound' in locals():
                    rln41, = ax5.plot([mint[nump][efr][0], mint[nump][efr][0]], [Smin, Smax], '--r', lw = 2.0)
                    rln42, = ax5.plot([mint[nump][efr][1], mint[nump][efr][1]], [Smin, Smax], '--r', lw = 2.0)
                # field
                fld = interp2d(X, Y, mfld[nump][efr], kind = 'cubic')
                CNR = ax1.contour(XX, YY, fld(XX,YY), linspace(-mlim[nump], mlim[nump], NCONT), linewidths = LW, cmap = contCM)
                # side barr
                barrH  = int(10**6 * (fld(XX,YY).max() - fld(XX,YY).min()) / GAIN)
                if not barrH:
                    barrH += 1
                barrCRV = []
                #print(efr, barrH, 10**6*fld(XX,YY).max()/GAIN, 10**6*fld(XX,YY).min()/GAIN)
                for level in linspace(0, log10(barrH), 200):
                    barrCRV.append(ax0.plot([0,1], [10**level, 10**level], lw = 10, color = barrCM(level / log10(BARR_AMP))))
                # OLD PART FOR NEGATIVE BARR ALSO
                #tmp = 10**6 * sgn * abs(coef[nump][0,:]).max() * comp[nump][:,0] / GAIN
                #barrH = int(tmp[int(mint[nump][efr][0] * EFREQ / 1000.0) : int(mint[nump][efr][1] * EFREQ / 1000.0)].mean()) + 1
                #barrH = fld(XX,YY).max() > -fld(XX,YY).min()                    
                #print(fld(XX,YY).max(), fld(XX,YY).min(), mlim[nump])
                #barrH = barrH * fld(XX,YY).max() + (not barrH) * fld(XX,YY).min()
                #barrH = int(10**6 * barrH / GAIN)
                #print(barrH)
                #if barrH < 0:
                #    ax0.set_ylim(10000,0.0001)
                #    ax0.set_yticks([0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000])
                #    ax0.set_yticklabels(['10$^4$','10$^3$','10$^2$','10','0', '-10', '-10$^2$', '-10$^3$', '-10$^4$'])
                #    barrH = -barrH
                #    barrP = ax0.fill_between([0,1],[1,1],[barrH,barrH], color='#0099ff')
                #    barrH = -barrH
                #elif barrH > 0:
                #    ax0.set_ylim(0.0001, 10000)
                #    ax0.set_yticks([0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000])
                #    ax0.set_yticklabels(['-10$^4$','-10$^3$','-10$^2$','-10','0', '10', '10$^2$', '10$^3$', '10$^4$'])
                #    barrP = ax0.fill_between([0,1],[1,1],[barrH,barrH], color='#cc0000')
                #if barrH < -1000:
                #    writeLCARD(fld(XX,YY), 'jopa.dat')
                #    show()
                #    exit()
                # set size
                fig.subplots_adjust(wspace = 0.30, hspace = 0.50, left=0.01, right=0.99, top=0.95, bottom=0.05)
                fig.set_size_inches(ASPECT * PH / my_dpi, PH / my_dpi)
                #clb = ax1.clabel(CNR, CNR.levels[::1], fonTSIZE = 12, inline_spacing = 5)
                # draw (resterize) the image
                fig.canvas.draw()
                # read image
                image = frombuffer(fig.canvas.tostring_rgb(), dtype = uint8)
                image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                # put the frame into video-file
                outF.write(image)
                # remove
                for cnr in CNR.collections:
                    cnr.remove()
                # remove the red lines
                rln21.remove()
                rln22.remove()
                rln31.remove()
                rln32.remove()
                if 'sound' in locals():
                    rln41.remove()
                    rln42.remove()
                # remode the barr
                try:
                    for barr in barrCRV:
                        barr[0].remove()
                except:
                    print(' - no new barr was plotted!')
                #iterate further
                efr += 1
            # cleaning the plots!!!
            ax0.cla()
            ax0.set_axis_off()
            ax3.cla()
            ax3.set_axis_off()
            ax4.cla()
            ax4.set_axis_off()
            ax5.cla()
            ax5.set_axis_off()
            
        else:
            # speed
            spd = ax2.set_title('\n speed: x1', size = TSIZE, color = lcolor)
            # set size
            fig.subplots_adjust(wspace = 0.30, hspace = 0.50, left=0.01, right=0.99, top=0.95, bottom=0.05)
            fig.set_size_inches(ASPECT * PH / my_dpi, PH / my_dpi)
            # draw (resterize) the image
            fig.canvas.draw()
            # read image
            image = frombuffer(fig.canvas.tostring_rgb(), dtype = uint8)
            image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            # put the frame into video-file
            outF.write(image)

        # if it just finished the pulse
        if lfb < rpb and rfb >= rpb:
            print(' - finilizing the pulse!!!!')
            efr  = 0
            nume = 0
            if nump + 1 < len(limP[r]):
                nump += 1
            print('efr = %d, nump = %d, nume = %d' % (efr, nump, nume))

        # remove the red lines and the previous fram belonging to the processed frame
        img.remove()
        rln11.remove()
        rln12.remove()


        vfr += 1
    # release the handlers
    outF.release()
    outB.release()

    # remove the axes and whatever in the upper-right oscilogramm
    #for obj in foL:
    #    obj.remove()
    close(fig)

    print('   finished')

print(' - buy buy baby - baby is a good buy!!!')


