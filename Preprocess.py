# -*- coding: utf-8 -*-
import glob
import csv
import numpy as np
from scipy.fftpack import fft
import os
import h5py


# directory setting
path = "./data/raw/*/*/OUT/*"
file_list = glob.glob(path)
file_list_py = [file for file in file_list if file.endswith(".h5")]
omit_count = 0

# processing
for file in file_list_py:
    temp = file.split('/')
    temp_dir = temp[0] + "/" + temp[1] + "/" + "preprocessed"

    # raw data read and reshape
    if temp[3] == "B_U":
        label = "0"
    elif temp[3] == "T_D":
        label = "1"
    elif temp[3] == "L2R":
        label = "2"
    elif temp[3] == "R2L":
        label = "3"
    elif temp[3] == "TOUCH":
        label = "4"
    elif temp[3] == "TOUCH2":
        label = "5"
    elif temp[3] == "PUSH":
        label = "6"

    re_file = temp_dir + "/" + temp[6][0:-3] + "_" + label + ".csv"

    # make directory
    if not(os.path.isdir(temp_dir)):
        os.makedirs(os.path.join(temp_dir))
    print("processing: " + file)

    f = h5py.File(file,'r')
    Frames = list(f.keys())
    Frames = Frames[220:-1]

    # High Pass Filter for DC noise
    HPF = np.ones(20)
    HPF[0:10] = 0.1**((9-np.linspace(0, 9, 10, endpoint=True))/10)
    HPF[0:4] = 0
    
    # kernel for power difference along frame
    kernel = np.array([0.2,0.2,0,-0.2,-0.2])

    # fft and filtering, separation to real and imag part
    # perform along frames
    gesture_status = 0
    onset_idx = 0
    for idx, name in enumerate(Frames):
        if gesture_status == 61:
            break
        else:
            Frame = f[name]
            data = np.array(Frame['TimeData'])
            data = data-2048
            data = data.reshape(-1,256)
            power = np.zeros(1)
            for i, d in enumerate(data):
                re = fft(d).real
                re = (re[0:20]*HPF).reshape(1,20)
                im = fft(d).imag
                im = (im[0:20]*HPF).reshape(1,20)
                power = power + np.sum((re*re)[0][4:9] + (im*im)[0][4:9])
                temp = np.append(re,im, axis=0)
                if i == 0:
                    fft_frame = temp
                else:
                    fft_frame = np.append(fft_frame, temp, axis=0)
            
            fft_frame = fft_frame.reshape(1, fft_frame.shape[0], fft_frame.shape[1])
            
            # power difference calculation for gesture detecting
            if gesture_status == 0:
                if idx == 0:
                    power_data = power
                    buffer_data = fft_frame
                else:
                    if buffer_data.shape[0] == 20:
                        buffer_data = np.delete(buffer_data,0,axis=0)
                    power_data = np.append(power_data, power, axis=0)
                    buffer_data = np.append(buffer_data, fft_frame, axis=0)

                if idx > 3:
                    power_diff = np.convolve(power_data[idx-4:idx+1],kernel,'valid')
                    if power_diff > 3*(10**5):
                        onset_idx = idx
                        fft_data = buffer_data
                        gesture_status = 1
            else:
                fft_data = np.append(fft_data, fft_frame, axis=0) # Nframe X Nchannel X Nrange
                gesture_status += 1
    
    # When there is no gesture detection satisfying threshold, there is no saving
    if onset_idx > 0:
        # flatten to 2d array for saving as scv file
        fft_data = np.transpose(fft_data, (1,0,2))
        if fft_data.shape[1] < 80:
            print("It doesn't have frame length of 80. Its length is " + str(fft_data.shape[1]))
            dif = 80 - fft_data.shape[1]
            fft_data = np.append(fft_data,np.zeros((6,dif,20)),axis=1)            
        fft_data = fft_data.reshape(fft_data.shape[0],-1)

        csvfile = open(re_file,"w",newline="")
        csvwriter = csv.writer(csvfile)
        csvwriter.writerows(fft_data)
    else:
        omit_count += 1
