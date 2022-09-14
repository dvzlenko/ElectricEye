#!/usr/bin/python3

import sys
import cv2
import tkinter
from numpy import *
from struct import *
from matplotlib.pyplot import *
from tkinter.filedialog import askopenfilename
from tkinter.messagebox import askyesno, showerror

def PointLocation(event):
    global POINT, fig
    if event.dblclick:
        POINT.append([round(getattr(event, 'xdata')), round(getattr(event, 'ydata'))])
        plot(POINT[-1][0],POINT[-1][1], '+r', lw = 0.5, ms = 8)
    if (len(POINT) == 4 and flag == 1) or (len(POINT) == 2 and flag == 2):
        close(fig)

#++++++++++++++++++++++++#
#                        #
#    INPUT FILE          #
#                        #
#++++++++++++++++++++++++#

try:
    name = sys.argv[1]
except:
    root = tkinter.Tk()
    name = askopenfilename(filetypes = [('video files', ['*.avi', '*.mp4']), ('All files','*.*')])
    root.destroy()

if 'avi' not in name and 'mp4' not in name:
    print('   !!!   ERROR   !!!')
    print('   You have provided file ' + name + ', which is not *.avi or *.mp4 file. Rvise the source code of MakeRectangle to continue. Otherwise, the script will ERASE the sourse video!!!!')
    exit()

#++++++++++++++++++++++++#
#                        #
#    MAIN DEFINITIONS    #
#                        #
#++++++++++++++++++++++++#

TTIME = 50          # total duration of the file in minutes
otype = float16     # data type in the output file. This one is mandatory!!!

#++++++++++++++++++++++++#
#                        #
#    PROGRAMM BODY       #
#                        #
#++++++++++++++++++++++++#

# trying to open input file
cap = cv2.VideoCapture(name)
cap.set(cv2.CAP_PROP_POS_FRAMES, 40000)
OK, frame = cap.read()

if not OK:
    print('Error opening video file!!!')
    root = tkinter.Tk()
    showerror('ERROR','Error openeing video file')
    root.destroy()
    exit()

PW = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
PH = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# distortion correction stuff
cam = eye(3)
cam[0,2]    = PW / 2.0  # define center x
cam[1,2]    = PH / 2.0  # define center y
cam[0,0]    = 8.0      # define focal length x
cam[1,1]    = 8.0      # define focal length y
distCoeff   = zeros((4,1))
distCoeff[0]   = -1.0e-06              # 8 mm FishEye 2864x1512

dst  = cv2.undistort(frame, cam, distCoeff)

# electrodes location
flag  = 1
POINT = []
fig = figure(101)
imshow(dst)
title('Please, double-click the corner electrodes!')
BPE = fig.canvas.mpl_connect('button_press_event', PointLocation)
show()
print(' - corner electrodes are:')
print('   %4d x %4d:' % tuple(POINT[0]))
print('   %4d x %4d:' % tuple(POINT[1]))
print('   %4d x %4d:' % tuple(POINT[2]))
print('   %4d x %4d:' % tuple(POINT[3]))

G = zeros([6,4])
for p in range(4):
    G[p,0] = POINT[p][0]
    G[p,1] = POINT[p][1]
for p in range(2):
    G[p,2]   = round(mean([POINT[p][0], POINT[p+2][0]]))
    G[p+2,2] = G[p,2]
    G[2*p,3] = round(mean([POINT[2*p][1], POINT[2*p+1][1]]))
    G[2*p+1,3] = G[2*p,3]

# Perspective correction stuff
M = cv2.getPerspectiveTransform(float32(G[:4,:2]), float32(G[:4,2:]))
warp = cv2.warpPerspective(dst, M, (PW, PH))

# image corners location stuff
flag = 2
POINT = []
fig = figure(102)
img = imshow(warp)
title('Please, double-click the corners of the image!')
BPE = fig.canvas.mpl_connect('button_press_event', PointLocation)
show()
print(' - image corners are:')
print('   %4d x %4d:' % tuple(POINT[0]))
print('   %4d x %4d:' % tuple(POINT[1]))

G[4,0] = POINT[0][0]
G[4,1] = POINT[1][0]
G[4,2] = POINT[0][1]
G[4,3] = POINT[1][1]

# linspaces for contour and imshow
X  = linspace(G[0,2], G[1,2], 8)
Y  = linspace(G[0,3], G[3,3], 8)

# plot the feedback!
figure(103)
imshow(warp)
title('The result whould be like this!')

for a in range(8):
    for b in range(8):
        plot(X[a], Y[b], '+r', ms = 10, lw = 0.1)

plot([G[4,0], G[4,1]], [G[4,2],G[4,2]], '-', color = '#0099ff')
plot([G[4,0], G[4,1]], [G[4,3],G[4,3]], '-', color = '#0099ff')
plot([G[4,0], G[4,0]], [G[4,2],G[4,3]], '-', color = '#0099ff')
plot([G[4,1], G[4,1]], [G[4,2],G[4,3]], '-', color = '#0099ff')

show()

# write the file
W = open(name.replace('avi','rect').replace('mp4','rect'), 'w')
for p in range(5):
    for q in range(4):
        W.write('%6d' % G[p,q])
    W.write('\n')
W.close()


