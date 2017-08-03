#!/usr/bin/env python2.7

from __future__ import print_function

import os, sys, numpy as np
import argparse
from scipy import misc
import caffe
import tempfile
from math import ceil
import time
#from cv2 import *
import cv2

UNKNOWN_FLOW_THRESH = 1e7

def make_color_wheel():
    """
    Generate color wheel according Middlebury color code
    :return: Color wheel
    """
    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR

    colorwheel = np.zeros([ncols, 3])

    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.transpose(np.floor(255*np.arange(0, RY) / RY))
    col += RY

    # YG
    colorwheel[col:col+YG, 0] = 255 - np.transpose(np.floor(255*np.arange(0, YG) / YG))
    colorwheel[col:col+YG, 1] = 255
    col += YG

    # GC
    colorwheel[col:col+GC, 1] = 255
    colorwheel[col:col+GC, 2] = np.transpose(np.floor(255*np.arange(0, GC) / GC))
    col += GC

    # CB
    colorwheel[col:col+CB, 1] = 255 - np.transpose(np.floor(255*np.arange(0, CB) / CB))
    colorwheel[col:col+CB, 2] = 255
    col += CB

    # BM
    colorwheel[col:col+BM, 2] = 255
    colorwheel[col:col+BM, 0] = np.transpose(np.floor(255*np.arange(0, BM) / BM))
    col += + BM

    # MR
    colorwheel[col:col+MR, 2] = 255 - np.transpose(np.floor(255 * np.arange(0, MR) / MR))
    colorwheel[col:col+MR, 0] = 255

    return colorwheel

colorwheel = make_color_wheel()
ncols = np.size(colorwheel, 0)

def compute_color(u, v):
    """
    compute optical flow color map
    :param u: optical flow horizontal map
    :param v: optical flow vertical map
    :return: optical flow in color code
    """
    [h, w] = u.shape
    img = np.zeros([h, w, 3])
    nanIdx = np.isnan(u) | np.isnan(v)
    u[nanIdx] = 0
    v[nanIdx] = 0

    rad = np.sqrt(u**2+v**2)

    a = np.arctan2(-v, -u) / np.pi

    fk = (a+1) / 2 * (ncols - 1) + 1

    k0 = np.floor(fk).astype(int)

    k1 = k0 + 1
    k1[k1 == ncols+1] = 1
    f = fk - k0

    for i in range(0, np.size(colorwheel,1)):
        tmp = colorwheel[:, i]
        col0 = tmp[k0-1] / 255
        col1 = tmp[k1-1] / 255
        col = (1-f) * col0 + f * col1

        idx = rad <= 1
        col[idx] = 1-rad[idx]*(1-col[idx])
        notidx = np.logical_not(idx)

        col[notidx] *= 0.75
        img[:, :, i] = np.uint8(np.floor(255 * col*(1-nanIdx)))

    return np.uint8(img)


parser = argparse.ArgumentParser()
parser.add_argument('caffemodel', help='path to model')
parser.add_argument('deployproto', help='path to deploy prototxt template')
#parser.add_argument('listfile', help='one line should contain paths "img0.ext img1.ext out.flo"')
parser.add_argument('--gpu',  help='gpu id to use (0, 1, ...)', default=0, type=int)
#parser.add_argument('--verbose',  help='whether to output all caffe logging', action='store_true')

args = parser.parse_args()

if(not os.path.exists(args.caffemodel)): raise BaseException('caffemodel does not exist: '+args.caffemodel)
if(not os.path.exists(args.deployproto)): raise BaseException('deploy-proto does not exist: '+args.deployproto)
#if(not os.path.exists(args.listfile)): raise BaseException('listfile does not exist: '+args.listfile)

width = -1
height = -1

num_blobs = 2
#input_data = []

caffe.set_device(args.gpu)
caffe.set_mode_gpu()
caffe.set_logging_disabled()
proto = open(args.deployproto).readlines()

cam = cv2.VideoCapture(0)
retval, img0mat = cam.read()
img0 = np.asarray(img0mat)     
width = 640
height = 480

vars = {}
vars['TARGET_WIDTH'] = width
vars['TARGET_HEIGHT'] = height

divisor = 64.
vars['ADAPTED_WIDTH'] = int(ceil(width/divisor) * divisor)
vars['ADAPTED_HEIGHT'] = int(ceil(height/divisor) * divisor)

vars['SCALE_WIDTH'] = width / float(vars['ADAPTED_WIDTH']);
vars['SCALE_HEIGHT'] = height / float(vars['ADAPTED_HEIGHT']);

tmp = tempfile.NamedTemporaryFile(mode='w', delete=False)

for line in proto:
    for key, value in vars.items():
        tag = "$%s$" % key
        line = line.replace(tag, str(value))
    tmp.write(line)
tmp.flush()

net = caffe.Net(tmp.name, args.caffemodel, caffe.TEST)


while (True):
    img1 = img0
    start = time.time()
    retval, img0mat = cam.read()
    img0 = np.asarray(img0mat)
    input_data = []
    input_data.append(img0[np.newaxis, :, :, :].transpose(0, 3, 1, 2)[:, [2, 1, 0], :, :])
    input_data.append(img1[np.newaxis, :, :, :].transpose(0, 3, 1, 2)[:, [2, 1, 0], :, :])

    input_dict = {}
    input_dict[net.inputs[0]] = input_data[0]
    input_dict[net.inputs[1]] = input_data[1]
    #for blob_idx in range(num_blobs):
        #input_dict[net.inputs[blob_idx]] = input_data[blob_idx]
    
    net.forward(**input_dict)

    blob = np.squeeze(net.blobs['predict_flow_final'].data).transpose(1, 2, 0)

    #print("it took", time.time() - start, "seconds.")
    #print(blob.size)
    #print(blob.shape)
   
    #print(fl.shape)

    fps = 1.0/(time.time() - start)
    flow = compute_color(blob[:, :, 0]/50, blob[:, :, 1]/50)
    #fl = np.zeros((height, width))
    #flow = (np.dstack((fl, blob)) / 50)
    
    cv2.putText(img1, "{:.2f}".format(fps) + ' fps', (0, 12), cv2.FONT_HERSHEY_PLAIN, 1, (128, 128, 255))
    cv2.putText(img1, "Press 'q' to quit", (0, 24), cv2.FONT_HERSHEY_PLAIN, 1, (128, 128, 255))
    cv2.imshow('Flow', flow)
    cv2.imshow('Input', img1)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    #def readFlow(name):
    #    if name.endswith('.pfm') or name.endswith('.PFM'):
    #        return readPFM(name)[0][:,:,0:2]
#
     #   f = open(name, 'rb')

     #   header = f.read(4)
     #   if header.decode("utf-8") != 'PIEH':
      #      raise Exception('Flow file header does not contain PIEH')
#
      #  width = np.fromfile(f, np.int32, 1).squeeze()
     #   height = np.fromfile(f, np.int32, 1).squeeze()

     #   flow = np.fromfile(f, np.float32, width * height * 2).reshape((height, width, 2))

     #   return flow.astype(np.float32)

def writeFlow(name, flow):  
    f = open(name, 'wb')
    f.write('PIEH'.encode('utf-8'))
    np.array([flow.shape[1], flow.shape[0]], dtype=np.int32).tofile(f)
    flow = flow.astype(np.float32)
    flow.tofile(f)

#writeFlow('/media/cvlnas-ew202/menandro/data/Kitti2012/training/flow.flo', blob)

cam.release()
