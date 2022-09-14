#!/usr/bin/python3

import os
import re
import sys
import cv2
import time
import tkinter
from tkinter.filedialog import askopenfilename 
from tkinter.messagebox import askyesno, showerror
from tkinter.simpledialog import askinteger
from struct import *
from numpy import *
from matplotlib.pyplot import *
from scipy.interpolate import *
from scipy.io import wavfile
from moviepy.editor import VideoFileClip
from sklearn.decomposition import PCA

def set_GLOBALS():
    # GLOBALS
    global D, limE, AV, EFREQ, NCOMP, FULL, NCONT, LW, FL, DST, GAIN, THEME, ASPECT
    # Global Settings
    limE   =        1.0     # Time period flanking the pulse (msec)
    NCOMP  =          1     # Number of components to draw. Could be 1 or 2!!!! TODO: Fix it?
    AV     =          1     # Number of frames for movmed, it MUST be ODD as number of frames is always integer!!!
    NCONT  =         30     # Number of Contours
    LW     =        3.0     # Line Width
    FL     =          8     # Focal Length of the lense used
    DST    =   -1.0e-06     # Distorsion correction constant
    GAIN   =       1100 / 100     # Amplifier gain
    THEME  =      'DARK'    # can be DARK or LIGHT, defining the overall appearance of the pictures
    ASPECT =        1.8     # picture aspect ratio
    

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
# numc   - number of the channels => data columns in the file
# itype  - type of the data in the file
# otype  - type of the data in the returned array
# lim    - limits to read from the file, frame numbers
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
    print('   done...') 


# when user pushes the cross on the electrical plot window :)
#def onClose(event):
#    fig.canvas.mpl_disconnect(BPE)
#    fig.canvas.mpl_disconnect( CE)
#    close(fig)
#    doTheJob()

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

# file name and path
name   = pname.split('/')[-1]
path   = pname.replace(name, '')
dname  = dname.split('/')[-1]
dname  = re.sub(name + '-', '', dname)


# video file handler
t = time.time()
if os.path.isfile(path + name + '.avi'):
    cap = cv2.VideoCapture(path + name + '.avi')
else:
    print('There is no target file ' + path + name + '.avi in the working directory!\n   exiting :(')
    root = tkinter.Tk()
    root.withdraw()
    showerror(title = 'ERG Error Message', message = 'There is no target file ' + path + name + '.avi in the working directory!\n\n Execution aborted :(')
    root.destroy()
    exit()
print(' - video file was read in %.1f sec' % (time.time() - t))

PW = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
PH = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# GLOBALS!!!
set_GLOBALS()
VFREQ       = cap.get(cv2.CAP_PROP_FPS)         # Frequency for the video file (Hz)
G           = loadtxt(path + name + '.rect')

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

ind = loadtxt(elogname)
if len(shape(ind)) == 1:
    indE = [ind[1]]
    if len(ind) > 3:
        limE = [ind[3] / 2]
    else:
        limE = [limE]
else:
    indE = ind[:,1]
    if len(ind[0,:]) > 3:
        limE = ind[:,3] / 2 # !!!!!!!!!!!!!
    else:
        limE = limE * ones(len(ind))


# trying either read the alignment from right audio channel or *.align file
if not os.path.isfile(path + name + '.wav'):
    videoclip = VideoFileClip(path + name + '.avi')
    audioclip = videoclip.audio
    if audioclip:
        audioclip.write_audiofile(path + name + '.wav')
        AFREQ, sound = wavfile.read(path + name + '.wav')
else:
    AFREQ, sound = wavfile.read(path + name + '.wav')
    
if 'sound' in locals():
    # NCOMP = 2
    # counter signal MUST be in the right channel!!!!
    tmp     = (sound[:,0] > 300)
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
        start   = (apeaks[0] - aperiod) / AFREQ * VFREQ  # exact beginning of the LCARD series in the video frames scale
        finish  = apeaks[-1] / AFREQ * VFREQ             #       exact end of the LCARD series in the video frames scale
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
            W.write('%30.19f %30.19f\n' % (1, 0))
            W.write('%30.19f %30.19f\n' % (cf[0], cf[1]))
            W.close()
else:
    #NCOMP = 3
    try:
        ALIGN = loadtxt(path + name + '.align')   # Linear regression coefficients for L-CARD to L-CARD and to video frame alignment and conversion
    except:
        print('I did not found either any audio-track in the video file or the corresponding *.align file!!!!\n   exiting... :(')
        exit()

# real video frame rate
realVFREQ = EFREQ * ALIGN[1,0]
print('   real video frame rate is %.5f frames per second' % (realVFREQ))
if (realVFREQ / VFREQ < 0.95) or (realVFREQ / VFREQ > 1.05):
    root = tkinter.Tk()
    root.withdraw()
    showerror('ERROR','Real video frame rate is %.4f fps, which implies a mistake in L-CARD frequency preset\n\nExecution aborted :(' % realVFREQ)
    root.destroy()
    print('   ERROR!!\n   Such a video frame rate implies a mistake in L-CARD frequency preset\n   Execution aborted :(')
    exit()


# sceen dpi definition
root   = tkinter.Tk()
my_dpi = root.winfo_fpixels('1i')
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


# doTheJob code here!!! Just the pictures creation!!!
for event in range(len(indE)):
    indV = int(ALIGN[1,0] * indE[event] + ALIGN[1,1])
    print(' - processing event at %10.3f sec, video frame # %d, electro frame # %d' % (indE[event] / EFREQ, indV, indE[event]))
    
    # extract the full L-CARD data for the pulse neigborhood
    readlim = [int(indE[event] - (limE[event] / 1000 * EFREQ) - ((AV - 1) / 2)), 
               int(indE[event] + (limE[event] / 1000 * EFREQ) + ((AV - 1) / 2))]
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
    t   = linspace(0, len(DD) / EFREQ, len(DD))

    # selected interval data file NOT SMOOTHED!!!!
    if not os.path.isdir(path + name):
        os.mkdir(path + name)
    if not os.path.isdir(path + name + '/' + dname + '-DAT'):
        os.mkdir(path + name + '/' + dname + '-DAT')
    
    tmpname = path + name + '/' + dname + '-DAT/LCARD.%08.3fsec.%dms.dat' % (indE[event] / EFREQ, 2 * limE[event])
    if not os.path.isfile(tmpname):
        writeLCARD(DD, tmpname)
    #
    #++++++++++++++++++++++++++++++++++++++++++++++
    #
    # SMOOOOOOOOOOOOOOOOOOOOTHING!!!!! 
    DD  = movmed(DD, AV)
    # OFFSET REMOVING!!!!!
    DD = DD - DD[:10,:].mean(0)    
    #
    #++++++++++++++++++++++++++++++++++++++++++++++
    #   
    # output files
    if not os.path.isdir(path + name + '/' + dname + '-PNG'):
        os.mkdir(path + name + '/' + dname + '-PNG')
    if not os.path.isdir(path + name + '/' + dname + '-PDF'):
        os.mkdir(path + name + '/' + dname + '-PDF')

    # set the frame in the video file
    cap.set(cv2.CAP_PROP_POS_FRAMES, indV)
    OK, frame = cap.read()
    # convert to grb!!!
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # distortion and perspective
    dst  = cv2.undistort(frame, cam, distCoeff)
    warp = cv2.warpPerspective(dst, M, (PW, PH))

    # save the blank frame
    #fig = figure(666, dpi = my_dpi)
    #ax = gca()
    #ax.set_axis_off()
    #img = imshow(frame, cmap = 'gray')
    #img.axes.get_xaxis().set_visible(False)
    #img.axes.get_yaxis().set_visible(False)
    #fig.tight_layout(pad = 0)
    #fig.set_size_inches(PW / my_dpi, PH / my_dpi)
    # uncomment here to plot the blank frame!!!
    #fig.savefig(path + name + '/' + dname + '-PNG/BLANK.%08.3fsec.png' % (indE[event] / EFREQ))
    #close(fig)

    # PCA
    pca  = PCA(n_components = 3) # TODO: Here it is always 3!!!
    comp = pca.fit_transform(DD)
    coef = pca.components_
    # PCA dat file
    tmpname = path + name + '/' + dname + '-DAT/PCA.%08.3fsec.%dms.%dcomp.dat' % (indE[event] / EFREQ, 2 * limE[event], len(coef))
    if not os.path.isfile(tmpname):
        writeLCARD(comp, tmpname)
    tmpname = path + name + '/' + dname + '-DAT/COEF.%08.3fsec.%dms.%dcomp.dat' % (indE[event] / EFREQ, 2 * limE[event], len(coef))
    if not os.path.isfile(tmpname):
        writeLCARD(coef.T, tmpname)

    tt = linspace(0, 1000 * len(comp) / EFREQ,  len(comp))

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

    fig = figure(777, dpi = my_dpi)
    fig.set_facecolor(bcolor)
    fig.set_size_inches(ASPECT * PH / my_dpi, PH / my_dpi)
    # set size
    gs  = fig.add_gridspec(4, 3)

    # main window
    ax0 = fig.add_subplot(gs[:,:2])
    ax0.set_axis_off()
    ax0.set_xlim(G[4,0], G[4,1])
    ax0.set_ylim(G[4,3], G[4,2])
    img = ax0.imshow(warp, cmap = 'gray')
    img.axes.get_xaxis().set_visible(False)
    img.axes.get_yaxis().set_visible(False)

    # x-axis for oscillogramms
    #tm = linspace(0, 1000 * len(comp) / EFREQ, len(comp))
    tm = linspace(-1000 * len(comp) / EFREQ / 2, 1000 * len(comp) / EFREQ / 2, len(comp))
    # settings
    tsize = (G[2,3] - G[0,3]) //  60  # title text size
    asize = tsize - 2                 # axis label text size
    lsize = asize - 2                 # axis values text size
    csize = lsize - 2                 # in-line contour text size

    # full oscillogramm
    # scaling if the curves ardinates!!!!
    scale = []
    if (abs(DD).max() / GAIN) > 0.0001:
        scale.append(10**3)
    else:
        scale.append(10**6)
    ax1 = fig.add_subplot(gs[0, 2])
    ax1.tick_params(axis='both', labelsize = lsize, labelcolor = lcolor)

    # sort the curves from min to max for proper coloring!!!!    
    tmp = abs(DD).max(0) * ((-1) * (DD.max(0) > -DD.min(0)) + (+1) * (DD.max(0) < -DD.min(0)))
    tmp = tmp.argsort()
    DD  = DD.take(tmp, 1)
    
    cm = get_cmap('bwr')
    oscl = []
    for p in range(shape(DD)[1]):
        #oscl.append(ax1.plot(tm, 1000 * DD[:,p] / GAIN, '-', lw = 0.5, color = cm(1 - p / (shape(DD)[1] - 1)) * array([0.9, 0.9, 0.9, 1]))[0])
        oscl.append(ax1.plot(tm, scale[0] * DD[:,p] / GAIN, '-', lw = 0.5, color = (array(cm(1 - p / (shape(DD)[1] - 1))) / 2 + 0.5))[0]) 
    #oscl = ax1.plot(tm, 1000 * DD / GAIN, '-k', lw = 0.3)

    ax1.spines['top'].set_color(bcolor)
    ax1.spines['right'].set_color(bcolor)
    ax1.spines['bottom'].set_color(lcolor)
    ax1.spines['left'].set_color(lcolor)
    ax1.set_facecolor(bcolor)
    ax1.set_title('Full Oscillogramm, event at %.3fsec' % (indE[event] / EFREQ), size = tsize, color = lcolor)
    ax1.set_xlabel('time, msec', size = asize, color = lcolor)
    if scale[0] == 10**3:
        ax1.set_ylabel('voltage, mV', size = asize, color = lcolor)
    else:
        ax1.set_ylabel('voltage, $\mu$V', size = asize, color = lcolor)


    # first component
    if max(abs((abs(coef[0,:]).max() * comp[:,0]) / GAIN)) > 0.0001:
        scale.append(10**3)
    else:
        scale.append(10**6)
    ax2 = fig.add_subplot(gs[1, 2])
    ax2.tick_params(axis='both', labelsize = lsize, labelcolor = lcolor)
    sgn   = int(max(abs(comp[:,0])) == max(comp[:,0])) - int(max(abs(comp[:,0])) == -min(comp[:,0]))
    PCA1 = scale[1] * sgn * abs(coef[0,:]).max() * (comp[:,0] - comp[:10,0].mean()) / GAIN
    pca1, = ax2.plot(tm, PCA1, '-', lw = 1.0, color = '#000000')
    ax2.set_facecolor(bcolor)
    ax2.spines['top'].set_color(bcolor)
    ax2.spines['right'].set_color(bcolor)
    ax2.spines['bottom'].set_color(lcolor)
    ax2.spines['left'].set_color(lcolor)
    ax2.set_title('First PCA component', size = tsize, color = lcolor)
    ax2.set_xlabel('time, msec', size = asize, color = lcolor)
    if scale[1] == 10**3:
        ax2.set_ylabel('voltage, mV', size = asize, color = lcolor)
    else:
        ax2.set_ylabel('voltage, $\mu$V', size = asize, color = lcolor)

    # second component
    if max(abs((abs(coef[1,:]).max() * comp[:,1]) / GAIN)) > 0.0001:
        scale.append(10**3)
    else:
        scale.append(10**6)
    ax3 = fig.add_subplot(gs[2, 2])
    ax3.tick_params(axis='both', labelsize = lsize, labelcolor = lcolor)
    sgn   = int(max(abs(comp[:,1])) == max(comp[:,1])) - int(max(abs(comp[:,1])) == -min(comp[:,1]))
    PCA2 = scale[2] * sgn * abs(coef[1,:]).max() * (comp[:,1] - comp[:10,1].mean()) / GAIN
    pca2, = ax3.plot(tm, PCA2, '-', lw = 1.0, color = lcolor)
    ax3.set_facecolor(bcolor)
    ax3.spines['top'].set_color(bcolor)
    ax3.spines['right'].set_color(bcolor)
    ax3.spines['bottom'].set_color(lcolor)
    ax3.spines['left'].set_color(lcolor)
    ax3.set_title('Second PCA component', size = tsize, color = lcolor)
    ax3.set_xlabel('time, msec', size = asize, color = lcolor)
    if scale[2] == 10**3:
        ax3.set_ylabel('voltage, mV', size = asize, color = lcolor)
    else:
        ax3.set_ylabel('voltage, $\mu$V', size = asize, color = lcolor)
    
    # third component or sound if present
    if 'sound' not in locals():
        if max(abs((abs(coef[2,:]).max() * comp[:,2]) / GAIN)) > 0.0001:
            scale.append(10**3)
        else:
            scale.append(10**6)
        ax4 = fig.add_subplot(gs[3, 2])
        ax4.tick_params(axis='both', labelsize = lsize, labelcolor = lcolor)
        sgn   = int(max(abs(comp[:,2])) == max(comp[:,2])) - int(max(abs(comp[:,2])) == -min(comp[:,2]))
        PCA3 = scale[3] * sgn * abs(coef[2,:]).max() * (comp[:,2] - comp[:10,2].mean()) / GAIN
        pca3, = ax4.plot(tm, PCA3, '-', lw = 1.0, color = lcolor)
        ax4.set_title('Third PCA component', size = tsize, color = lcolor)
        ax4.set_xlabel('time, msec', size = asize, color = lcolor)
        ax4.set_ylabel('voltage, mV', size = asize, color = lcolor)
        ax4.set_facecolor(bcolor)
        ax4.spines['top'].set_color(bcolor)
        ax4.spines['right'].set_color(bcolor)
        ax4.spines['bottom'].set_color(lcolor)
        ax4.spines['left'].set_color(lcolor)
        if scale[3] == 10**3:
            ax4.set_ylabel('voltage, mV', size = asize, color = lcolor)
        else:
            ax4.set_ylabel('voltage, $\mu$V', size = asize, color = lcolor)

        # subplots and PCAs dictionaries
        sbpl = {0 : ax2,  1 : ax3,  2 : ax4}
        PCAc = {0 : pca1, 1 : pca2, 2 : pca3}
        PCAd = {0 : PCA1, 1 : PCA2, 2 : PCA3}
    else:   
        # start and finish of the piece of the sound
        # precise video frame converted to the audio frame
        AMULT = 2
        tmp = indE[event] - AMULT * limE[event] * EFREQ / 1000  # 1000 means msec!!!
        ss  = round((ALIGN[1,0] * tmp + ALIGN[1,1]) * (AFREQ / VFREQ))
        tmp = indE[event] + AMULT * limE[event] * EFREQ / 1000
        sf  = round((ALIGN[1,0] * tmp + ALIGN[1,1]) * (AFREQ / VFREQ))
        #xSnd = linspace(0, 1000 * (sf - ss) / AFREQ, sf - ss)
        xSnd = linspace(-1000 * (sf - ss) / AFREQ / 2, 1000 * (sf - ss) / AFREQ / 2, sf - ss)
        ySnd = sound[ss:sf, 1]
        # scale is very necessary!!!
        ax4 = fig.add_subplot(gs[3, 2])
        ax4.tick_params(axis='both', labelsize = lsize, labelcolor = lcolor)
        pca3 = ax4.plot(xSnd, ySnd, '-', lw = 1.0, color = '#0099ff')[0]
        if (ax4.get_ylim()[1] - ax4.get_ylim()[0]) < 10 * sound[:,1].std():
            yl = [-10 * sound[:,1].std(), +10 * sound[:,1].std()]
        else:
            yl = ax4.get_ylim()
        ax4.set_title('Audio channel', size = tsize, color = lcolor)
        ax4.set_xlabel('time, msec', size = asize, color = lcolor)
        ax4.set_ylabel('sound pressure, a.u.', size = asize, color = lcolor)
        ax4.set_ylim(yl)
        ax4.set_facecolor(bcolor)
        ax4.spines['top'].set_color(bcolor)
        ax4.spines['right'].set_color(bcolor)
        ax4.spines['bottom'].set_color(lcolor)
        ax4.spines['left'].set_color(lcolor)

        # subplots and PCAs dictionaries
        sbpl = {0 : ax2,  1 : ax3}
        PCAc = {0 : pca1, 1 : pca2}
        PCAd = {0 : PCA1, 1 : PCA2}

    
    # iterate per components
    for c in range(NCOMP):
        # main PCA components' sign
        point = (abs(coef[c,:]).max() == abs(coef[c,:])).nonzero()[0][0]
        sgn   = int(max(abs(comp[:,c])) == max(comp[:,c])) - int(max(abs(comp[:,c])) == -min(comp[:,c]))
        # plotting components
        #figPCA = figure(888)
        #plot(tt, sgn * abs(coef[c,:]).max() * comp[:,c] / 1.1, '-r', lw = 1.0)
        #plot(tt, sgn * abs(coef[c,:]).max() * comp[:,c] / 1.1, '-k', lw = 0.1)
        #xlabel('Time, msec')
        #ylabel('Voltage, mV')
        # uncomment here to plot PCA components themselves!!!
        #savefig(path + name + '/' + dname + '-PNG/CRV.%08.3fsec.%dms.%d.png' % (indE[event] / EFREQ, 2 * limE[event], c + 1))
        #savefig(path + name + '/' + dname + '-PDF/CRV.%08.3fsec.%dms.%d.pdf' % (indE[event] / EFREQ, 2 * limE[event], c + 1))
        #close(figPCA)

        # FIELD DISTRIBUTION: 1   9  17  25    33  41  49  57   
        #                     2  10  18  26    34  42  50  58  
        #                     3  11  19  27    35  43  51  59  
        #                     4  12  20  28    36  44  52  60  
        #                     5  13  21  29    37  45  53  61  
        #                     6  14  22  30    38  46  54  62  
        #                     7  15  23  31    39  47  55  63  
        #                     8  16  24  32    40  48  56  64  
        # matrix must be filled in the reverse order, as in the picture the origin is in the upper-left corner, while in the plot it in the lower-left one!
        F = zeros([8,8])
        for m in range(8):
            for n in range(8):
                F[n,m] = scale[c + 1] * sgn * abs(comp[:,c]).max() * coef[c, 8 * m + n] / GAIN
                # reflected rows and columns
                #F[ 0,1:-1] = -sgn[p] * amp[p][q] * coef[p][q, 0::4]
                #F[-1,1:-1] = -sgn[p] * amp[p][q] * coef[p][q, 3::4]
                #F[1:-1, 0] = -sgn[p] * amp[p][q] * coef[p][q, :4]
                #F[1:-1,-1] = -sgn[p] * amp[p][q] * coef[p][q, -4:]
                # corners filled with the corner value
                #F[ 0, 0] = -sgn[p] * amp[p][q] * coef[p][q, 0]
                #F[ 0,-1] = -sgn[p] * amp[p][q] * coef[p][q,20]
                #F[-1, 0] = -sgn[p] * amp[p][q] * coef[p][q, 3]
                #F[-1,-1] = -sgn[p] * amp[p][q] * coef[p][q,23]

        # field
        fld = interp2d(X, Y, F, kind = 'cubic')
        fld = fld(XX,YY)
        lim = abs(fld).max()
        cnr = ax0.contour(XX, YY, fld, linspace(-lim, lim, NCONT), linewidths = LW, cmap = 'bwr')

        # PCA component recolor
        PCAc[c].remove()
        PCAc[c] = sbpl[c].plot(tm, PCAd[c], '-', lw = 1.0, color = '#ff0000')[0]
        # save the image and clear
        fig.subplots_adjust(wspace = 0.20, hspace = 0.80, left=0.01, right=0.99, top=0.95, bottom=0.05)
        #fig.set_size_inches((3 / 2) * (G[4,1] - G[4,0]) / my_dpi, (G[4,3] - G[4,2]) / my_dpi)
        fig.set_size_inches(ASPECT * PH / my_dpi, PH / my_dpi)
        clb = ax0.clabel(cnr, cnr.levels[::1], fontsize = lsize, inline_spacing = 5)
        #print('PCA # %d' % c)
        #print(fig.subplotpars.hspace, fig.subplotpars.wspace, fig.subplotpars.left, fig.subplotpars.right, fig.subplotpars.top, fig.subplotpars.bottom)
        #print(fig.get_tight_layout())
        fig.savefig(path + name + '/' + dname + '-PNG/FLD.%08.6fsec.%dms.%d.png' % (indE[event] / EFREQ, 2 * limE[event], c + 1), facecolor=fig.get_facecolor())
        fig.savefig(path + name + '/' + dname + '-PDF/FLD.%08.6fsec.%dms.%d.pdf' % (indE[event] / EFREQ, 2 * limE[event], c + 1), facecolor=fig.get_facecolor())
        for cn in cnr.collections:
            cn.remove()
        for cl in clb:
            cl.remove()
        PCAc[c].remove()
        PCAc[c] = sbpl[c].plot(tm, PCAd[c], '-', lw = 1.0, color = lcolor)[0]

        # save the LCARD dat file
        #print(indE[event], indE[event] / 20000, indE[event] / 50000) 
        tmpname = path + name + '/' + dname + '-DAT/FLD.%08.3fsec.%dms.%d.dat' % (indE[event] / EFREQ, 2 * limE[event], c + 1)
        if not os.path.isfile(tmpname):
            writeLCARD(F, tmpname)
    close('all')  
cap.release()

# check the video frame number in EventsList and add them if necessary
D = loadtxt(elogname)
if len(shape(D)) == 2:
    if shape(D)[1] == 2:
        F = open(elogname)
        lines = F.readlines()
        F.close()
        W = open(elogname.replace('.log', '-NoVideoFrames.log'), 'w')
        W.writelines(lines)
        W.close()
        W = open(elogname, 'w')
        W.write('# %d Hz\n' % EFREQ)
        for p in range(len(ind)):
            W.write('%10.3f\t%9d\t%6d\n' % (indE[p] / EFREQ, indE[p], int(ALIGN[1, 0] * indE[p] + ALIGN[1, 1])))
        W.close()
else:
    if len(D) == 2:
        F = open(elogname)
        lines = F.readlines()
        F.close()
        W = open(elogname.replace('.log', '-NoVideoFrames.log'), 'w')
        W.writelines(lines)
        W.close()
        W = open(elogname, 'w')
        W.write('# %d Hz\n' % EFREQ)
        W.write('%10.3f\t%9d\t%6d\n' % (indE[0] / EFREQ, indE[0], int(ALIGN[1, 0] * indE[0] + ALIGN[1, 1])))
        W.close()

print(' - buy buy baby - baby is a good buy!!!')

