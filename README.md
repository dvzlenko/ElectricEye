# ElectricEye

This repo contains a set of utilities aimed to process the raw data on the electric activity of fishes, obtained in A.N.Severtsov Institute of Ecology and Evolution. The utilities were developed in the frame of the restrictions of the data type and format, so they are not universal and can be used only with the certain data files.

The raw electric data collected using two 32-channel ADCs, so we do have two raw (\*.dat) files per each experimental session. To distinguish between them, the names of the raw files always contain the number of ADC device collected it. The raw video file is just an \*.avi file. 


1. LocateEvents.py - this script reads the initial electric data, chooses several registration channels and plots them as a function of time. The script requires the actual data sampling rate, which is usually 20 or 50kHz. After the plot was created, user can zoim the curves in or out using standard matplotlib functionality. Double-clicking the interesting events pushes the corresponding real time and number of the sample into the main output file \*-EventsList-XX.log, where XX - is an incrementable number.

2. MakePCA.py - reads  \*-EventsList-XX.log file, than reads the raw \*.dat files and for each event listed in the former plots the spatial distribution of the registered electric potential.

3. MakeRectangle.py and MakeRectangleInn.py - two simple scripts aimed to provide the pixel coordinates of the corner electrodes, which is necessary for electric field plotting over the video frame. The output file is \*.rect file.

4. MakePictures.py - do the same as MakePCA.py but plots the electric field plot over the real image of the experimental tank. The precise synchronization of the electric and video record requires a 1kHz clock signal funneled to the right audio channel of the camera. When ADSs starts data acquisition, the clock signal appears that provides for data synchronization. The coefficients necessary for alignment are stored in \*.align file. The script requires \*-EventsList-XX.log, \*.dat, \*.avi, and also \*.rect and \*.align files. The output is a set of the pictures and some data extracted from the raw electric data files.

5. MakeVideo.py - creates short combined video-electric clips, demonstrating fish behaviour together with their electric activity. The requirements are the same as for MakePictures.py


The raw data are quite large, so we decided to host them in one of the available servers in Moscow State University. Please find them here:
http://caas.ru/stuff/ElectricEyeData/

Please, fill free to mail me: dvzlenko@gmail.com

Dmitry

