from numpy import loadtxt
import numpy as np
import scipy.io
from os import listdir
from os.path import isfile, isdir, join
import h5py
from PIL import Image
import os
from generateGT import GT
import cv2
import lutorpy as lua
require('torch')

dsDir = '/home/snf/mohsen/ucf_sports_actions/ucf_action'
classIndex = {'Diving': 1, 'Golf': 2, 'Kicking': 3, 'Riding': 4, 'Run': 5, 'SkateBoarding': 6, 'Swing1': 7, 'Swing2': 8,
              'Lifting': 9, 'Walk': 10}
timeLen = 16
bgClass = 11
means = [0, 0, 0]
cropSize = 112
newHeight = 128
newWidth = 171

videoFile = open('videos.txt', 'r')
videoList = videoFile.read().split('\n')

trainList = open('trainList.txt', 'w')
testList = open('testList.txt', 'w')

for video in videoList:
    if not video:
        break
    frames = []
    flipFrames = []
    gts = []
    flipGts = []
    video = video.split()
    print(video)
    clp = video[6].split('/')
    clp = join(clp[0], clp[1])
    clp = join(dsDir, clp)
    outputPath = clp
    # print(clp)
    actGt = join(clp, 'gt')

    i = 0
    k = 0
    for f in listdir(clp):
        path = join(clp, f)
        if isfile(path) and '.avi' in f:
            startFrame = int(f[len(f)-7:-4])
            preStr = f[:-7]


            cap = cv2.VideoCapture(path)
            while(True):
                ret,frame = cap.read()

                if ret == False:
                    break
                if startFrame>999:
                    preStr = f[:-8]
                tmp = join(actGt, preStr + str(startFrame).zfill(3) + '.tif.txt')

                skip = False
                if not os.path.exists(tmp):
                    tmp = join(actGt, preStr + str(startFrame).zfill(3) + '.jpg.txt')
                    if not os.path.exists(tmp):
                        print ("Skipped:  " + tmp)
                        skip = True
                startFrame = startFrame + 1
                if not skip:
                    i = i+1
                    txt = open(tmp, "r")
                    gtData = txt.read().split()
                    size = frame.shape
                    # print(gtData)
                    #gt = GT(int(gtData[0]), int(gtData[1]), int(gtData[2]), int(gtData[3]), size[1], size[0], classIndex[video[1]], bgClass)
                    gt = GT(int(gtData[0]), int(gtData[1]), int(gtData[2]), int(gtData[3]), size[0], size[1], 1, 2)

                    frame = cv2.resize(frame,(newWidth, newHeight))
                    frame = frame[(newHeight-cropSize)/2:(newHeight+cropSize)/2,(newWidth-cropSize)/2:(newWidth+cropSize)/2]

                    flipFrame = cv2.flip(frame,1)

                    gt = gt.resize((newWidth, newHeight), Image.NEAREST)
                    gt = gt.crop(((newWidth-cropSize)/2,(newHeight-cropSize)/2,(newWidth+cropSize)/2,(newHeight+cropSize)/2))


                    #gt.show()
                    #cv2.imshow('Frame', frame)
                    #cv2.waitKey()


                    flipGt = gt.transpose(Image.FLIP_LEFT_RIGHT)

                    frame = np.asarray(frame).transpose((2, 1, 0))
                    flipFrame = np.asarray(flipFrame).transpose((2, 1, 0))
                    gt = np.asarray(gt)
                    flipGt = np.asarray(flipGt)

                    #print(frame.shape)
                    #frame = frame[:][:][0] - means[0]
                    #frame = frame[:][:][1] - means[1]
                    #frame = frame[:][:][2] - means[2]
                    #print(frame.shape)
                    frames.append(frame)

                    #flipFrame = flipFrame[:][:][0] - means[0]
                    #flipFrame = flipFrame[:][:][1] - means[1]
                    #flipFrame = flipFrame[:][:][2] - means[2]
                    flipFrames.append(flipFrame)

                    gts.append(gt)
                    flipGts.append(flipGt)

                    if i == timeLen:
                        i = 0
                        k = k+1
                        frames = np.asarray(frames)
                        flipFrames = np.asarray(flipFrames)
                        gts = np.asarray(gts)
                        flipGts = np.asarray(flipGts)

                        torch.save(join(outputPath, str(k)+'_frames.t7'), torch.fromNumpyArray(frames))
                        torch.save(join(outputPath, str(k)+'_fframes.t7'), torch.fromNumpyArray(flipFrames))
                        torch.save(join(outputPath, str(k)+'_gt.t7'), torch.fromNumpyArray(gts))
                        torch.save(join(outputPath, str(k)+'_fgt.t7'), torch.fromNumpyArray(flipGts))
                        if video[2] == 'train':
                            trainList.write(outputPath+'/'+str(k)+'_\n')
                            trainList.write(outputPath + '/' + str(k) + '_f\n')
                        else:
                            testList.write(outputPath + '/'+str(k)+'_\n')
                            testList.write(outputPath + '/' + str(k) + '_f\n')

                        frames = []
                        flipFrames = []
                        gts = []
                        flipGts = []


trainList.close()
testList.close()




