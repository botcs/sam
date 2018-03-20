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
from torch.autograd import Variable




containing_dir = str(pathlib.Path(__file__).resolve().parent)

fileDir = os.path.dirname(os.path.realpath(__file__))
modelDir = os.path.join(fileDir, 'weights')

parser = argparse.ArgumentParser()

parser.add_argument('--dlibFacePredictor', type=str, help="Path to dlib's face predictor.",
                    default=os.path.join(modelDir, "shape_predictor_68_face_landmarks.dat"))
parser.add_argument('--database', type=str, help='Compare query image to pictures found in [database]',
                    default='~/Pictures/Webcam/')
parser.add_argument('--imgDim', type=int,
                    help="Default image dimension.", default=96)
parser.add_argument('--verbose', action='store_true')
parser.add_argument('--dlib_dim', type=int, help='im size for face recognition', default=224)
parser.add_argument('--query_path', default='', help='query image path')
parser.add_argument('--refresh', type=int, help='Refresh output image [sec]')
parser.add_argument('--webcam', type=int, default=0, help='use webcam')

args = parser.parse_args()
align = utils.AlignDlib(args.dlibFacePredictor)
prepareOpenFace = utils.prepareOpenFace

def ReadImage(imgPath):

    if args.verbose:
        print("Processing {}.".format(imgPath))
    bgrImg = cv2.imread(imgPath)
    if bgrImg is None:
        raise Exception("Unable to load image: {}".format(imgPath))
    return bgrImg

def ProcessImage(bgrImg, max_ratio=1, returnBB=False):

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
    if returnBB:
        return I_, utils.rect_to_bb(bb, ratio)
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
    #for idx, d in zip(idxs, ds):
    #    print('%30s  distance: %0.4f' % (img_paths[idx].split('/')[-1], d))
        #torch.dot()
    return idxs, ds

def draw_bb(frame, bb):
    x, y, w, h = bb
    cv2.rectangle(frame, (x, y), (x + w, y + h), (200, 200, 200), 2)

def draw_text(frame, bb, text, distance, threshold=0.0050,
              x_offset=0, y_offset=0, font_scale=2, thickness=2):
    x, y = bb[:2]
    color = (0, 200, 0) if distance < threshold else (0, 0, 200)
    cv2.putText(frame, text, (x + x_offset, y + y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, color, thickness, cv2.LINE_AA)



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


    if len(args.query_path) == 0:
        cap = cv2.VideoCapture(args.webcam)
        acc_idxs = {}

        imlist = []
        for img_path in img_paths:
            imlist.append(cv2.imread(img_path))

        cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
        while True:
            ret, bgrImg = cap.read()

            try:
                query_im, rect = ProcessImage(bgrImg, -1, returnBB=True)

                idxs, ds = find_k_nearest(query_im, 5)
                text = img_paths[idxs[0]].split('/')[-2]
                draw_bb(bgrImg, rect)
                draw_text(bgrImg, rect, text, ds[0])
                if args.verbose:
                    #print('\x1b[2J')
                    for idx, d in zip(idxs, ds):
                        print('%30s  distance: %0.4f' % (text, d))
                for idx in idxs:
                    if acc_idxs.get(idx) is None:
                        acc_idxs[idx] = 1
                    else:
                        acc_idxs[idx] += 1
            except Exception as e:
                print(e)
                pass

            finally:
                cv2.imshow('frame', bgrImg)
                cv2.waitKey(1)



    else:
        q_path = args.query_path
        query_im = ProcessImage(ReadImage(q_path))
        find_k_nearest(query_im, 3)

    # in OpenFace's sample code, cosine distance is usually used for f (128d).
