import cv2
import torch
import torch.utils.data
import torch.optim.lr_scheduler as lr_scheduler
import numpy as np
import scipy.io as scio
import os
from PIL import Image
from torch.autograd import Variable
import model as model
import anchor as anchor
from tqdm import tqdm
import random_erasing
import logging
import time
import datetime
import random
import struct

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

 # MSRA15 calibration data
fx = 241.42
fy = 241.42
u0 = 320./2
v0 = 240./2

# DataHyperParms
keypointsNumber = 21
cropWidth = 176
cropHeight = 176
batch_size = 64
learning_rate = 0.00035
Weight_Decay = 1e-4
learning_rate_decay = 0.1
learning_rate_decay_epoch = 7
nepoch = 17
RegLossFactor = 3
spatialFactor = 0.5
RandCropShift = 5
RandshiftDepth = 1
RandRotate = 180
RandScale = (1.0, 0.5)
xy_thres = 90
depth_thres = 500
load_num_workers = 1

randomseed = 12345
random.seed(randomseed)
np.random.seed(randomseed)
torch.manual_seed(randomseed)

total_subject_num = 9
test_subject_id = 3

save_dir = './result/MSRA15_batch_64_12345'

try:
    os.makedirs(save_dir)
except OSError:
    pass

ImgDir = '/home/mahdi/HVR/git_repos/A2J/data/MSRA15' # bin images
center_dir = '/home/mahdi/HVR/git_repos/A2J/data/msra_center' # in 3D coordinates
'''
 we compute mean/std on training set
 we first crop the original depth maps according to center points, which give us
  a hand-centered sub-image, then we compute the mean/std of all of these images.
'''
# put test error model that had been trained here and result file model at model_dir :
model_dir = ''
result_file = 'result_MSRA15.txt'


def pixel2world(x, fx, fy, ux, uy):
    x[:, :, 0] = (x[:, :, 0] - ux) * x[:, :, 2] / fx
    x[:, :, 1] = (x[:, :, 1] - uy) * x[:, :, 2] / fy
    return x
def world2pixel(x, fx, fy, ux, uy):
    x[:, :, 0] = x[:, :, 0] * fx / x[:, :, 2] + ux
    x[:, :, 1] = x[:, :, 1] * fy / x[:, :, 2] + uy
    return x


joint_id_to_name = {
    0: 'Palm',
    1: 'Thumb root',
    2: 'Thumb mid',
    3: 'Thumb tip',
    4: 'Index root',
    5: 'Index mid',
    6: 'Index tip',
    7: 'Middle root',
    8: 'Middle mid',
    9: 'Middle tip',
    10: 'Ring root',
    11: 'Ring mid',
    12: 'Ring tip',
    13: 'Pinky root',
    14: 'Pinky mid',
    15: 'Pinky tip',
}


def transform(img, label, matrix):
    '''
    img: [H, W]  label, [N,2]
    '''
    img_out = cv2.warpAffine(img, matrix, (cropWidth, cropHeight))
    label_out = np.ones((keypointsNumber, 3))
    label_out[:, :2] = label[:, :2].copy()
    label_out = np.matmul(matrix, label_out.transpose())
    label_out = label_out.transpose()

    return img_out, label_out


def dataPreprocess(index, img, lefttop_pixel, rightbottom_pixel,
                   depth_thres=150, augment=True):

    if augment:
        RandomOffset_1 = np.random.randint(-1 * RandCropShift, RandCropShift)
        RandomOffset_2 = np.random.randint(-1 * RandCropShift, RandCropShift)
        RandomOffset_3 = np.random.randint(-1 * RandCropShift, RandCropShift)
        RandomOffset_4 = np.random.randint(-1 * RandCropShift, RandCropShift)
        RandomOffsetDepth = np.random.normal(0, RandshiftDepth, cropHeight * cropWidth).reshape(cropHeight, cropWidth)
        RandomOffsetDepth[np.where(RandomOffsetDepth < RandshiftDepth)] = 0
        RandomRotate = np.random.randint(-1 * RandRotate, RandRotate)
        RandomScale = np.random.rand() * RandScale[0] + RandScale[1]
        matrix = cv2.getRotationMatrix2D((cropWidth / 2, cropHeight / 2), RandomRotate, RandomScale)
    else:
        RandomOffset_1, RandomOffset_2, RandomOffset_3, RandomOffset_4 = 0, 0, 0, 0
        RandomRotate = 0
        RandomScale = 1
        RandomOffsetDepth = 0
        matrix = cv2.getRotationMatrix2D((cropWidth / 2, cropHeight / 2), RandomRotate, RandomScale)

    new_Xmin = max(lefttop_pixel[index,0,0] + RandomOffset_1, 0)
    new_Ymin = max(rightbottom_pixel[index,0,1] + RandomOffset_2, 0)
    new_Xmax = min(rightbottom_pixel[index,0,0] + RandomOffset_3, img.shape[1] - 1)
    new_Ymax = min(lefttop_pixel[index,0,1] + RandomOffset_4, img.shape[0] - 1)
    try:
        imCrop = img[int(new_Ymin):int(new_Ymax), int(new_Xmin):int(new_Xmax)].copy()
    except Exception:
        # print('img=', img)
        # print('imCrop=', imCrop)
        # print('cropWidth=', cropWidth)
        # print('cropHeight=', cropHeight)
        print('corrupt index = ', index)
        # imCrop = img[int(new_Ymax):int(new_Ymin), int(new_Xmin):int(new_Xmax)].copy()
        # imgResize = cv2.resize(imCrop, (cropWidth, cropHeight), interpolation=cv2.INTER_NEAREST)

    return imCrop, index


def _world2pixel(x, y, z, img_width, img_height, fx, fy):
    p_x = x * fx / z + img_width / 2
    p_y = (img_height / 2 - y * fy / z)
    return p_x, p_y
def points2pixels(points, img_width, img_height, fx, fy):
    pixels = np.zeros((points.shape[0], 2))
    pixels[:, 0], pixels[:, 1] = \
        _world2pixel(points[:, 0], points[:, 1], points[:, 2], img_width, img_height, fx, fy)
    return pixels
######################   Pytorch dataloader   #################
class my_dataloader(torch.utils.data.Dataset):
    def __init__(self, ImgDir, center_dir, fx, fy, total_subject_num, mode, test_subject_id, augment=True):
        self.ImgDir = ImgDir
        self.xy_thres = xy_thres
        self.depth_thres = depth_thres

        self.img_width = 320
        self.img_height = 240
        self.max_depth = 0
        self.fx = fx
        self.fy = fy
        self.joint_num = 21
        self.world_dim = 3
        self.folder_list = ['1', '2', '3', '4', '5', '6', '7', '8', '9', 'I', 'IP', 'L', 'MP', 'RP', 'T', 'TIP', 'Y'] # if you change to ['5'] pay attention refined center is read from the beginning of the file per person
        self.total_subject_num = total_subject_num

        self.center_dir = center_dir
        self.mode = mode
        self.test_subject_id = test_subject_id
        self.randomErase = random_erasing.RandomErasing(probability=0.5, sl=0.02, sh=0.4, r1=0.3, mean=[0])
        self.augment = augment

        self._load()

        centre_train_world = np.expand_dims(self.ref_pts, axis=1).copy().astype(np.float32)
        self.centerUVD = np.expand_dims(self.centerUVD, axis=1).astype(np.float32)

        centre_train_world = pixel2world(self.centerUVD.copy(), fx, fy, u0, v0)
        centerlefttop_train = centre_train_world.copy()
        centerlefttop_train[:, 0, 0] = centerlefttop_train[:, 0, 0] - xy_thres
        centerlefttop_train[:, 0, 1] = centerlefttop_train[:, 0, 1] + xy_thres

        centerrightbottom_train = centre_train_world.copy()
        centerrightbottom_train[:, 0, 0] = centerrightbottom_train[:, 0, 0] + xy_thres
        centerrightbottom_train[:, 0, 1] = centerrightbottom_train[:, 0, 1] - xy_thres

        self.lefttop_pixel = world2pixel(centerlefttop_train, fx, fy, u0, v0)
        self.rightbottom_pixel = world2pixel(centerrightbottom_train, fx, fy, u0, v0)

    def __getitem__(self, index):
        def load_depthmap(filename, img_width, img_height, max_depth):
            with open(filename, mode='rb') as f:
                data = f.read()
                _, _, left, top, right, bottom = struct.unpack('I' * 6, data[:6 * 4])
                num_pixel = (right - left) * (bottom - top)
                cropped_image = struct.unpack('f' * num_pixel, data[6 * 4:])
                cropped_image = np.asarray(cropped_image).reshape(bottom - top, -1)
                depth_image = np.zeros((img_height, img_width), dtype=np.float32)
                depth_image[top:bottom, left:right] = cropped_image
                depth_image[depth_image == 0] = max_depth
                return depth_image

        depth = load_depthmap(self.names[index], self.img_width, self.img_height, self.max_depth)

        # self.centerUVD shape is (K,1,3) type float 32 raw center UVD
        imgCrop, index = dataPreprocess(index, depth, self.lefttop_pixel, self.rightbottom_pixel, self.depth_thres, augment=False)

        return imgCrop, index

    def __len__(self):
        return len(self.centerUVD)

    def _load(self):
        self._compute_dataset_size()

        self.num_samples = self.train_size if self.mode == 'train' else self.test_size
        self.joints_world = np.zeros((self.num_samples, self.joint_num, self.world_dim))
        self.keypointsUVD = np.zeros((self.num_samples, self.joint_num, self.world_dim))
        self.centerUVD = np.zeros((self.num_samples, self.world_dim))
        self.ref_pts = np.zeros((self.num_samples, self.world_dim))
        self.names = []

        # Collect reference center points strings
        if self.mode == 'train':
            ref_pt_file = 'center_train_' + str(self.test_subject_id) + '_refined.txt'
        else:
            ref_pt_file = 'center_test_' + str(self.test_subject_id) + '_refined.txt'

        with open(os.path.join(self.center_dir, ref_pt_file)) as f:
            ref_pt_str = [l.rstrip() for l in f]

        #
        file_id = 0
        frame_id = 0

        for mid in range(self.total_subject_num):
            if self.mode == 'train':
                model_chk = (mid != self.test_subject_id)
            elif self.mode == 'test':
                model_chk = (mid == self.test_subject_id)
            else:
                raise RuntimeError('unsupported mode {}'.format(self.mode))

            if model_chk:
                for fd in self.folder_list:
                    annot_file = os.path.join(self.ImgDir, 'P' + str(mid), fd, 'joint.txt')

                    lines = []
                    with open(annot_file) as f:
                        lines = [line.rstrip() for line in f]

                    # skip first line
                    for i in range(1, len(lines)):
                        # referece point
                        splitted = ref_pt_str[file_id].split()
                        if splitted[0] == 'invalid':
                            print('Warning: found invalid reference frame')
                            file_id += 1
                            continue
                        else:
                            self.ref_pts[frame_id, 0] = float(splitted[0])
                            self.ref_pts[frame_id, 1] = float(splitted[1])
                            self.ref_pts[frame_id, 2] = float(splitted[2])

                        self.centerUVD[frame_id, :2] = points2pixels(self.ref_pts[frame_id, :].reshape(1,self.world_dim), self.img_width, self.img_height, self.fx, self.fy)
                        self.centerUVD[frame_id, 2] = self.ref_pts[frame_id, 2]


                        # joint point
                        splitted = lines[i].split()
                        for jid in range(self.joint_num):
                            self.joints_world[frame_id, jid, 0] = float(splitted[jid * self.world_dim])
                            self.joints_world[frame_id, jid, 1] = float(splitted[jid * self.world_dim + 1])
                            self.joints_world[frame_id, jid, 2] = -float(splitted[jid * self.world_dim + 2])

                        self.keypointsUVD[frame_id, :, :2] = points2pixels(self.joints_world[frame_id, :, :], self.img_width, self.img_height, self.fx, self.fy)
                        self.keypointsUVD[frame_id, :, 2] = self.joints_world[frame_id, :, 2]

                        filename = os.path.join(self.ImgDir, 'P' + str(mid), fd, '{:0>6d}'.format(i - 1) + '_depth.bin')
                        self.names.append(filename)

                        frame_id += 1
                        file_id += 1
    def _compute_dataset_size(self):
        self.train_size, self.test_size = 0, 0

        for mid in range(self.total_subject_num):
            num = 0
            for fd in self.folder_list:
                annot_file = os.path.join(self.ImgDir, 'P' + str(mid), fd, 'joint.txt')
                with open(annot_file) as f:
                    num = int(f.readline().rstrip())
                if mid == self.test_subject_id:
                    self.test_size += num
                else:
                    self.train_size += num

train_image_datasets = my_dataloader(ImgDir, center_dir, fx, fy, total_subject_num, mode='train',
                                     test_subject_id=test_subject_id, augment=True)
train_dataloaders = torch.utils.data.DataLoader(train_image_datasets, batch_size=batch_size,
                                                shuffle=True, num_workers=load_num_workers)

def calculator_msra15():
    mean=np.empty(train_image_datasets.num_samples)
    for index in range(0,train_image_datasets.num_samples):
        print('progress: {:.2f} %'.format(index/train_image_datasets.num_samples*100))
        mean[index] = np.mean(train_image_datasets.__getitem__(index)[0])

    MEAN = np.mean(mean)
    STD = np.std(mean)
    np.save('../data/msra15/msra15_mean.npy', MEAN)
    np.save('../data/msra15/msra15_std.npy', STD)

if __name__ == '__main__':
    calculator_msra15()



