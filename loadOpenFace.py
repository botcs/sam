import cv2
import glob
import sys
import numpy
import utils
import os
import time
import argparse
import pathlib
import torch





containing_dir = str(pathlib.Path(__file__).resolve().parent)

fileDir = os.path.dirname(os.path.realpath(__file__))
modelDir = os.path.join(fileDir, 'weights')
dlibModelDir = os.path.join(modelDir, 'dlib')

parser = argparse.ArgumentParser()

parser.add_argument('--dlibFacePredictor', type=str, help="Path to dlib's face predictor.",
                    default=os.path.join(dlibModelDir, "shape_predictor_68_face_landmarks.dat"))
parser.add_argument('--database', type=str, help='Compare query image to pictures found in [database]',
                    default='~/Pictures/Webcam/')
parser.add_argument('--imgDim', type=int,
                    help="Default image dimension.", default=96)
parser.add_argument('--verbose', action='store_true')
parser.add_argument('--dlib_dim', type=int, help='im size for face recognition', default=224)
parser.add_argument('--query_path', help='query image path')
parser.add_argument('--refresh', type=int, help='Refresh output image [sec]')
parser.add_argument('--webcam', action='store_true', help='use webcam')

args = parser.parse_args()
align = utils.AlignDlib(args.dlibFacePredictor)


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
'''
import torch.backends.cudnn as cudnn
from collections import OrderedDict
try:
    from . SpatialCrossMapLRN_temp import SpatialCrossMapLRN_temp
except:
    from SpatialCrossMapLRN_temp import SpatialCrossMapLRN_temp

class LambdaBase(nn.Sequential):
    def __init__(self, fn, *args):
        super(LambdaBase, self).__init__(*args)
        self.lambda_func = fn

    def forward_prepare(self, input):
        output = []
        for module in self._modules.values():
            output.append(module(input))
        return output if output else input


class Lambda(LambdaBase):
    def forward(self, input):
        return self.lambda_func(self.forward_prepare(input))


#
def Conv2d(in_dim, out_dim, kernel, stride, padding):
    l = torch.nn.Conv2d(in_dim, out_dim, kernel, stride=stride, padding=padding)
    return l

def BatchNorm(dim):
    l = torch.nn.BatchNorm2d(dim)
    return l

def CrossMapLRN(size, alpha, beta, k=1.0, gpuDevice=0):
    lrn = SpatialCrossMapLRN_temp(size, alpha, beta, k, gpuDevice=gpuDevice)
    n = Lambda( lambda x,lrn=lrn: Variable(lrn.forward(x.data).cuda(gpuDevice)) if x.data.is_cuda else Variable(lrn.forward(x.data)) )
    return n

def Linear(in_dim, out_dim):
    l = torch.nn.Linear(in_dim, out_dim)
    return l


class Inception(nn.Module):
    def __init__(self, inputSize, kernelSize, kernelStride, outputSize, reduceSize, pool, useBatchNorm, reduceStride=None, padding=True):
        super(Inception, self).__init__()
        #
        self.seq_list = []
        self.outputSize = outputSize

        #
        # 1x1 conv (reduce) -> 3x3 conv
        # 1x1 conv (reduce) -> 5x5 conv
        # ...
        for i in range(len(kernelSize)):
            od = OrderedDict()
            # 1x1 conv
            od['1_conv'] = Conv2d(inputSize, reduceSize[i], (1, 1), reduceStride[i] if reduceStride is not None else 1, (0,0))
            if useBatchNorm:
                od['2_bn'] = BatchNorm(reduceSize[i])
            od['3_relu'] = nn.ReLU()
            # nxn conv
            pad = int(numpy.floor(kernelSize[i] / 2)) if padding else 0
            od['4_conv'] = Conv2d(reduceSize[i], outputSize[i], kernelSize[i], kernelStride[i], pad)
            if useBatchNorm:
                od['5_bn'] = BatchNorm(outputSize[i])
            od['6_relu'] = nn.ReLU()
            #
            self.seq_list.append(nn.Sequential(od))

        ii = len(kernelSize)
        # pool -> 1x1 conv
        od = OrderedDict()
        od['1_pool'] = pool
        if ii < len(reduceSize) and reduceSize[ii] is not None:
            i = ii
            od['2_conv'] = Conv2d(inputSize, reduceSize[i], (1,1), reduceStride[i] if reduceStride is not None else 1, (0,0))
            if useBatchNorm:
                od['3_bn'] = BatchNorm(reduceSize[i])
            od['4_relu'] = nn.ReLU()
        #
        self.seq_list.append(nn.Sequential(od))
        ii += 1

        # reduce: 1x1 conv (channel-wise pooling)
        if ii < len(reduceSize) and reduceSize[ii] is not None:
            i = ii
            od = OrderedDict()
            od['1_conv'] = Conv2d(inputSize, reduceSize[i], (1,1), reduceStride[i] if reduceStride is not None else 1, (0,0))
            if useBatchNorm:
                od['2_bn'] = BatchNorm(reduceSize[i])
            od['3_relu'] = nn.ReLU()
            self.seq_list.append(nn.Sequential(od))

        self.seq_list = nn.ModuleList(self.seq_list)


    def forward(self, input):
        x = input
        ys = []
        target_size = None
        depth_dim = 0
        for seq in self.seq_list:
            #print(seq)
            #print(self.outputSize)
            #print('x_size:', x.size())
            y = seq(x)
            y_size = y.size()
            #print('y_size:', y_size)
            ys.append(y)
            #
            if target_size is None:
                target_size = [0] * len(y_size)
            #
            for i in range(len(target_size)):
                target_size[i] = max(target_size[i], y_size[i])
            depth_dim += y_size[1]

        target_size[1] = depth_dim
        #print('target_size:', target_size)

        for i in range(len(ys)):
            y_size = ys[i].size()
            pad_l = int((target_size[3] - y_size[3]) // 2)
            pad_t = int((target_size[2] - y_size[2]) // 2)
            pad_r = target_size[3] - y_size[3] - pad_l
            pad_b = target_size[2] - y_size[2] - pad_t
            ys[i] = F.pad(ys[i], (pad_l, pad_r, pad_t, pad_b))

        output = torch.cat(ys, 1)

        return output


class netOpenFace(nn.Module):
    def __init__(self, useCuda, gpuDevice=0):
        super(netOpenFace, self).__init__()

        self.gpuDevice = gpuDevice

        self.layer1 = Conv2d(3, 64, (7,7), (2,2), (3,3))
        self.layer2 = BatchNorm(64)
        self.layer3 = nn.ReLU()
        self.layer4 = nn.MaxPool2d((3,3), stride=(2,2), padding=(1,1))
        self.layer5 = CrossMapLRN(5, 0.0001, 0.75, gpuDevice=gpuDevice)
        self.layer6 = Conv2d(64, 64, (1,1), (1,1), (0,0))
        self.layer7 = BatchNorm(64)
        self.layer8 = nn.ReLU()
        self.layer9 = Conv2d(64, 192, (3,3), (1,1), (1,1))
        self.layer10 = BatchNorm(192)
        self.layer11 = nn.ReLU()
        self.layer12 = CrossMapLRN(5, 0.0001, 0.75, gpuDevice=gpuDevice)
        self.layer13 = nn.MaxPool2d((3,3), stride=(2,2), padding=(1,1))
        self.layer14 = Inception(192, (3,5), (1,1), (128,32), (96,16,32,64), nn.MaxPool2d((3,3), stride=(2,2), padding=(0,0)), True)
        self.layer15 = Inception(256, (3,5), (1,1), (128,64), (96,32,64,64), nn.LPPool2d(2, (3,3), stride=(3,3)), True)
        self.layer16 = Inception(320, (3,5), (2,2), (256,64), (128,32,None,None), nn.MaxPool2d((3,3), stride=(2,2), padding=(0,0)), True)
        self.layer17 = Inception(640, (3,5), (1,1), (192,64), (96,32,128,256), nn.LPPool2d(2, (3,3), stride=(3,3)), True)
        self.layer18 = Inception(640, (3,5), (2,2), (256,128), (160,64,None,None), nn.MaxPool2d((3,3), stride=(2,2), padding=(0,0)), True)
        self.layer19 = Inception(1024, (3,), (1,), (384,), (96,96,256), nn.LPPool2d(2, (3,3), stride=(3,3)), True)
        self.layer21 = Inception(736, (3,), (1,), (384,), (96,96,256), nn.MaxPool2d((3,3), stride=(2,2), padding=(0,0)), True)
        self.layer22 = nn.AvgPool2d((3,3), stride=(1,1), padding=(0,0))
        self.layer25 = Linear(736, 128)

        #
        self.resize1 = nn.UpsamplingNearest2d(scale_factor=3)
        self.resize2 = nn.AvgPool2d(4)

        #
        # self.eval()

        if useCuda:
            self.cuda(gpuDevice)


    def forward(self, input):
        x = input

        if x.data.is_cuda and self.gpuDevice != 0:
            x = x.cuda(self.gpuDevice)

        #
        if x.size()[-1] == 128:
            x = self.resize2(self.resize1(x))

        x = self.layer8(self.layer7(self.layer6(self.layer5(self.layer4(self.layer3(self.layer2(self.layer1(x))))))))
        x = self.layer13(self.layer12(self.layer11(self.layer10(self.layer9(x)))))
        x = self.layer14(x)
        x = self.layer15(x)
        x = self.layer16(x)
        x = self.layer17(x)
        x = self.layer18(x)
        x = self.layer19(x)
        x = self.layer21(x)
        x = self.layer22(x)
        x = x.view((-1, 736))

        x_736 = x

        x = self.layer25(x)
        x_norm = torch.sqrt(torch.sum(x**2, 1) + 1e-6)
        x = torch.div(x, x_norm.view(-1, 1).expand_as(x))

        return (x, x_736)

'''
#
# def prepareOpenFace(useCuda=True, gpuDevice=0, useMultiGPU=False):
#     model = netOpenFace(useCuda, gpuDevice)
#     model.load_state_dict(torch.load(os.path.join(containing_dir, 'openface.pth')))
#
#     if useMultiGPU:
#         model = nn.DataParallel(model)
#
#     return model

prepareOpenFace = utils.prepareOpenFace

def ReadImage(imgPath):

    if args.verbose:
        print("Processing {}.".format(imgPath))
    bgrImg = cv2.imread(imgPath)
    if bgrImg is None:
        raise Exception("Unable to load image: {}".format(imgPath))
    return bgrImg

def ProcessImage(bgrImg, max_ratio=1):

    rgbImg = cv2.cvtColor(bgrImg, cv2.COLOR_BGR2RGB)
    orig_rgbImg = rgbImg
    h = bgrImg.shape[0]
    ratio = args.dlib_dim / h
    rgbImg = cv2.resize(rgbImg, None, fx=ratio, fy=ratio)

    start = time.time()
    bb = align.getLargestFaceBoundingBox(rgbImg)
    while bb is None and max_ratio > ratio:
        ratio += 0.1
        rgbImg = cv2.resize(orig_rgbImg, None, fx=ratio, fy=ratio)
        bb = align.getLargestFaceBoundingBox(rgbImg)

    if args.verbose:
        print("  + Original size: {}".format(rgbImg.shape))

    if bb is None:
        raise RuntimeWarning("Unable to find a face!")
    if args.verbose:
        print("  + Face detection took {} seconds.".format(time.time() - start))
    #bb /= ratio

    start = time.time()
    alignedFace = align.align(args.imgDim, rgbImg, bb,
                              landmarkIndices=utils.AlignDlib.OUTER_EYES_AND_NOSE)
    cv2.imwrite('alignedQuery.png', alignedFace)
    if alignedFace is None:
        raise Exception("Unable to align image: {}".format(imgPath))
    if args.verbose:
        print("  + Aligned size: {}".format(alignedFace.shape))
        print("  + Aligned mean: {}".format(alignedFace.mean()))
        print("  + Aligned dev: {}".format(alignedFace.std()))
        print("  + Face alignment took {} seconds.".format(time.time() - start))

    img = numpy.transpose(alignedFace, (2, 0, 1))
    img = img.astype(numpy.float32) / 255.0
    cv2.imshow('preproc', cv2.cvtColor(alignedFace, cv2.COLOR_BGR2RGB))
    #print(numpy.min(img), numpy.max(img))
    #print(numpy.sum(img[0]), numpy.sum(img[1]), numpy.sum(img[2]))
    I_ = torch.from_numpy(img).unsqueeze(0)
    if useCuda:
        I_ = I_.cuda()
        return I_

def find_k_nearest(query_im, k=3):
    q_var = Variable(query_im, requires_grad=False)
    q_f, q_res = model(q_var)
    lin_d = ((f_736-q_res)**2).mean(-1)

    cos_d = torch.mm(f_736, q_res.transpose(0, 1)).squeeze()
    cos_d /= (f_736**2).mean()
    #print(lin_d.shape, cos_d.shape)


    #dist = ((f - q_f)**2).mean(-1)
    dist = lin_d
    ds, idxs = dist.topk(k, largest=False)
    ds = ds.data.cpu()
    idxs = idxs.data.cpu()
    #print(ds)
    #print(idxs)
    if args.verbose:
        for idx, d in zip(idxs, ds):
            print('%30s  distance: %0.4f' % (img_paths[idx].split('/')[-1], d))
            #torch.dot()
            return idxs, ds

if __name__ == '__main__':
    #
    useCuda = True
    if useCuda:
        assert torch.cuda.is_available()
    else:
        assert False, 'Sorry, .pth file contains CUDA version of the network only.'

    model = prepareOpenFace()
    model.load_state_dict(torch.load(os.path.join(containing_dir, 'weights', 'openface.pth')))
    model = model.eval()


    #img_paths = glob.glob('/home/botoscs/Pictures/office_badges/*.jpg', recursive=True)
    #img_paths += glob.glob('/home/botoscs/Pictures/Webcam/*.jpg', recursive=True)
    img_paths = glob.glob(args.database + '/**/*.jpg', recursive=True)
    print(img_paths)
    imgs = []
    for img_path in img_paths:
        try:
            img = ReadImage(img_path)
            img = ProcessImage(img)
        except RuntimeWarning as w:
            continue
        imgs.append(img)

    I_ = torch.cat(imgs, 0)
    I_ = Variable(I_, requires_grad=False)
    start = time.time()
    f, f_736 = model(I_)
    print("  + Forward pass took {} seconds.".format(time.time() - start))

    #print(f)
    '''
    for i in range(f_736.size(0) - 1):
        for j in range(i + 1, f_736.size(0)):
            df = f_736[i] - f_736[j]
            print(img_paths[i].split('/')[-1], img_paths[j].split('/')[-1], torch.dot(df, df))
    '''

    if args.webcam:
        cap = cv2.VideoCapture(1)
        acc_idxs = {}

        imlist = []
        for img_path in img_paths:
            imlist.append(cv2.imread(img_path))

        start_time = time.time()
        while True:
            ret, bgrImg = cap.read()
            cv2.imshow('frame', bgrImg)
            cv2.waitKey(1)
            try:
                query_im = ProcessImage(bgrImg, -1)
                idxs, ds = find_k_nearest(query_im, 5)
                print('\x1b[2J')
                for idx, d in zip(idxs, ds):
                    print('%30s  distance: %0.4f' % (img_paths[idx].split('/')[-2], d))
                for idx in idxs:
                    if acc_idxs.get(idx) is None:
                        acc_idxs[idx] = 1
                    else:
                        acc_idxs[idx] += 1
                #cv2.waitKey(100)
            except Exception:
                pass
            passed_time = time.time() - start_time
            if int(passed_time) % 5 == 4:

                max_idx = -1
                max_score = -1
                for k, v in acc_idxs.items():
                    if v > max_score:
                        max_idx = k
                        max_score = v
                '''if len(acc_idxs) > 0:
                    cv2.imshow('winname', imlist[max_idx])
                    cv2.waitKey(1)'''
                passed_time = 0
                acc_idxs = {}



    else:
        q_path = args.query_path
        query_im = ProcessImage(ReadImage(q_path))
        find_k_nearest(query_im, 3)

    # in OpenFace's sample code, cosine distance is usually used for f (128d).
