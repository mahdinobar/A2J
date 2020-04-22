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
load_num_workers = 8

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
MEAN = np.load('../data/msra15/msra15_mean.npy')
STD = np.load('../data/msra15/msra15_std.npy')

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

# MSRA15
joint_id_to_name = {
    0: 'wrist',
    1: 'index_mcp',
    2: 'index_pip',
    3: 'index_dip',
    4: 'index_tip',
    5: 'middle_mcp',
    6: 'middle_pip',
    7: 'middle_dip',
    8: 'middle_tip',
    9: 'ring_mcp',
    10: 'ring_pip',
    11: 'ring_dip',
    12: 'ring_tip',
    13: 'little_mcp',
    14: 'little_pip',
    15: 'little_dip',
    16: 'little_tip',
    17: 'thumb_mcp',
    18: 'thumb_pip',
    19: 'thumb_dip',
    20: 'thumb_tip',
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


def dataPreprocess(index, img, keypointsUVD, center, mean, std, lefttop_pixel, rightbottom_pixel,
                   depth_thres=150, augment=True):
    imageOutputs = np.ones((cropHeight, cropWidth, 1), dtype='float32')
    labelOutputs = np.ones((keypointsNumber, 3), dtype='float32')

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
        imgResize = cv2.resize(imCrop, (cropWidth, cropHeight), interpolation=cv2.INTER_NEAREST)
    except Exception:
        # print('img=', img)
        # print('imCrop=', imCrop)
        # print('cropWidth=', cropWidth)
        # print('cropHeight=', cropHeight)
        print('corrupt index = ', index)
        # imCrop = img[int(new_Ymax):int(new_Ymin), int(new_Xmin):int(new_Xmax)].copy()
        # imgResize = cv2.resize(imCrop, (cropWidth, cropHeight), interpolation=cv2.INTER_NEAREST)

    imgResize = np.asarray(imgResize, dtype='float32')  # H*W*C

    imgResize[np.where(imgResize >= center[index][0][2] + depth_thres)] = center[index][0][2]
    imgResize[np.where(imgResize <= center[index][0][2] - depth_thres)] = center[index][0][2]
    imgResize = (imgResize - center[index][0][2]) * RandomScale

    imgResize = (imgResize - mean) / std

    ## label
    label_xy = np.ones((keypointsNumber, 2), dtype='float32')
    label_xy[:, 0] = (keypointsUVD[index, :, 0].copy() - new_Xmin) * cropWidth / (new_Xmax - new_Xmin)  # x
    label_xy[:, 1] = (keypointsUVD[index, :, 1].copy() - new_Ymin) * cropHeight / (new_Ymax - new_Ymin)  # y
    # label_xy[:, 0] = (keypointsUVD[validIndex[index], :, 0].copy() - new_Xmin) * cropWidth / (new_Xmax - new_Xmin)  # x
    # label_xy[:, 1] = (keypointsUVD[validIndex[index], :, 1].copy() - new_Ymin) * cropHeight / (new_Ymax - new_Ymin)  # y

    if augment:
        imgResize, label_xy = transform(imgResize, label_xy, matrix)  ## rotation, scale

    imageOutputs[:, :, 0] = imgResize

    labelOutputs[:, 1] = label_xy[:, 0]
    labelOutputs[:, 0] = label_xy[:, 1]
    labelOutputs[:, 2] = (keypointsUVD[index, :, 2] - center[index][0][2]) * RandomScale  # Z
    # labelOutputs[:,2] = (keypointsUVD[validIndex[index],:,2] - center[index][0][2])   # Z


    imageOutputs = np.asarray(imageOutputs)
    imageNCHWOut = imageOutputs.transpose(2, 0, 1)  # [H, W, C] --->>>  [C, H, W]
    imageNCHWOut = np.asarray(imageNCHWOut)
    labelOutputs = np.asarray(labelOutputs)

    data, label = torch.from_numpy(imageNCHWOut), torch.from_numpy(labelOutputs)

    return data, label


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
        self.mean = MEAN
        self.std = STD
        self.xy_thres = xy_thres
        self.depth_thres = depth_thres

        self.img_width = 320
        self.img_height = 240
        self.min_depth = 100
        self.max_depth = 700
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
        data, label = dataPreprocess(index, depth, self.keypointsUVD, self.centerUVD, self.mean, self.std,
                                     self.lefttop_pixel, self.rightbottom_pixel, depth_thres=self.depth_thres, augment=True)
        if self.augment:
            data = self.randomErase(data)

        return data, label

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


# ######################   Pytorch test dataloader   #################
class my_dataloader_test(torch.utils.data.Dataset):
    def __init__(self, ImgDir, center_dir, fx, fy, total_subject_num, mode, test_subject_id, augment=True):
        self.ImgDir = ImgDir
        self.mean = MEAN
        self.std = STD
        self.xy_thres = xy_thres
        self.depth_thres = depth_thres

        self.img_width = 320
        self.img_height = 240
        self.min_depth = 100
        self.max_depth = 700
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

        centre_test_world = np.expand_dims(self.ref_pts, axis=1).copy().astype(np.float32)
        self.centerUVD = np.expand_dims(self.centerUVD, axis=1).astype(np.float32)

        centre_test_world = pixel2world(self.centerUVD.copy(), fx, fy, u0, v0)
        centerlefttop_test = centre_test_world.copy()
        centerlefttop_test[:, 0, 0] = centerlefttop_test[:, 0, 0] - xy_thres
        centerlefttop_test[:, 0, 1] = centerlefttop_test[:, 0, 1] + xy_thres

        centerrightbottom_test = centre_test_world.copy()
        centerrightbottom_test[:, 0, 0] = centerrightbottom_test[:, 0, 0] + xy_thres
        centerrightbottom_test[:, 0, 1] = centerrightbottom_test[:, 0, 1] - xy_thres

        self.lefttop_pixel = world2pixel(centerlefttop_test, fx, fy, u0, v0)
        self.rightbottom_pixel = world2pixel(centerrightbottom_test, fx, fy, u0, v0)


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
        data, label = dataPreprocess(index, depth, self.keypointsUVD, self.centerUVD, self.mean, self.std,
                                     self.lefttop_pixel, self.rightbottom_pixel, self.depth_thres, augment=False)

        return data, label

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
# import nonechucks as nc # to handle none corrupted samples
# train_datasets = nc.SafeDataset(train_image_datasets)
train_dataloaders = torch.utils.data.DataLoader(train_image_datasets, batch_size=batch_size,
                                                shuffle=True, num_workers=load_num_workers)


test_image_datasets = my_dataloader_test(ImgDir, center_dir, fx, fy, total_subject_num, mode='test',
                                     test_subject_id=test_subject_id, augment=False)
test_dataloaders = torch.utils.data.DataLoader(test_image_datasets, batch_size=batch_size,
                                               shuffle=False, num_workers=load_num_workers)


def train():
    net = model.A2J_model(num_classes=keypointsNumber)
    net = net.cuda()

    post_precess = anchor.post_process(shape=[cropHeight // 16, cropWidth // 16], stride=16, P_h=None, P_w=None)
    criterion = anchor.A2J_loss(shape=[cropHeight // 16, cropWidth // 16], thres=[16.0, 32.0], stride=16, \
                                spatialFactor=spatialFactor, img_shape=[cropHeight, cropWidth], P_h=None, P_w=None)
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=Weight_Decay)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=learning_rate_decay_epoch, gamma=learning_rate_decay)

    logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%Y/%m/%d %H:%M:%S', \
                        filename=os.path.join(save_dir, 'train.log'), level=logging.INFO)
    logging.info('======================================================')

    for epoch in range(nepoch):
        net = net.train()
        train_loss_add = 0.0
        Cls_loss_add = 0.0
        Reg_loss_add = 0.0
        timer = time.time()

        # Training loop
        for i, (img, label) in enumerate(train_dataloaders):

            torch.cuda.synchronize()

            img, label = img.cuda(), label.cuda()

            heads = net(img)
            # print(regression)
            optimizer.zero_grad()

            Cls_loss, Reg_loss = criterion(heads, label)

            loss = 1 * Cls_loss + Reg_loss * RegLossFactor
            loss.backward()
            optimizer.step()

            torch.cuda.synchronize()

            train_loss_add = train_loss_add + (loss.item()) * len(img)
            Cls_loss_add = Cls_loss_add + (Cls_loss.item()) * len(img)
            Reg_loss_add = Reg_loss_add + (Reg_loss.item()) * len(img)

            # printing loss info
            if i % 10 == 0:
                print('epoch: ', epoch, ' step: ', i, 'Cls_loss ', Cls_loss.item(), 'Reg_loss ', Reg_loss.item(),
                      ' total loss ', loss.item())

        scheduler.step(epoch)

        # time taken
        torch.cuda.synchronize()
        timer = time.time() - timer
        timer = timer / train_image_datasets.num_samples
        print('==> time to learn 1 sample = %f (ms)' % (timer * 1000))

        train_loss_add = train_loss_add / train_image_datasets.num_samples
        Cls_loss_add = Cls_loss_add / train_image_datasets.num_samples
        Reg_loss_add = Reg_loss_add / train_image_datasets.num_samples
        print('mean train_loss_add of 1 sample: %f, #train_indexes = %d' % (train_loss_add, train_image_datasets.num_samples))
        print('mean Cls_loss_add of 1 sample: %f, #train_indexes = %d' % (Cls_loss_add, train_image_datasets.num_samples))
        print('mean Reg_loss_add of 1 sample: %f, #train_indexes = %d' % (Reg_loss_add, train_image_datasets.num_samples))

        Error_test = 0
        Error_train = 0
        Error_test_wrist = 0

        if (epoch % 1 == 0):
            net = net.eval()
            output = torch.FloatTensor()
            outputTrain = torch.FloatTensor()

            for i, (img, label) in tqdm(enumerate(test_dataloaders)):
                with torch.no_grad():
                    img, label = img.cuda(), label.cuda()
                    heads = net(img)
                    pred_keypoints = post_precess(heads, voting=False)
                    output = torch.cat([output, pred_keypoints.data.cpu()], 0)

            result = output.cpu().data.numpy()
            Error_test = errorCompute(result, test_image_datasets.keypointsUVD, test_image_datasets.centerUVD)
            print('epoch: ', epoch, 'Test error:', Error_test)
            saveNamePrefix = '%s/net_%d_wetD_' % (save_dir, epoch) + str(Weight_Decay) + '_depFact_' + str(
                spatialFactor) + '_RegFact_' + str(RegLossFactor) + '_rndShft_' + str(RandCropShift)
            torch.save(net.state_dict(), saveNamePrefix + '.pth')

        # log
        logging.info('Epoch#%d: total loss=%.4f, Cls_loss=%.4f, Reg_loss=%.4f, Err_test=%.4f, lr = %.6f'
                     % (epoch, train_loss_add, Cls_loss_add, Reg_loss_add, Error_test, scheduler.get_lr()[0]))


def test():
    net = model.A2J_model(num_classes=keypointsNumber)
    net.load_state_dict(torch.load(model_dir))
    net = net.cuda()
    net.eval()

    post_precess = anchor.post_process(shape=[cropHeight // 16, cropWidth // 16], stride=16, P_h=None, P_w=None)

    output = torch.FloatTensor()
    torch.cuda.synchronize()
    for i, (img, label) in tqdm(enumerate(test_dataloaders)):
        with torch.no_grad():
            img, label = img.cuda(), label.cuda()
            heads = net(img)
            pred_keypoints = post_precess(heads, voting=False)
            output = torch.cat([output, pred_keypoints.data.cpu()], 0)

    torch.cuda.synchronize()

    result = output.cpu().data.numpy()
    writeTxt(result, test_image_datasets.centerUVD)
    error = errorCompute(result, test_image_datasets.keypointsUVD, test_image_datasets.centerUVD)
    print('Error:', error)


def errorCompute(source, target, center):
    assert np.shape(source) == np.shape(target), "source has different shape with target"

    Test1_ = source.copy()
    target_ = target.copy()
    Test1_[:, :, 0] = source[:, :, 1]
    Test1_[:, :, 1] = source[:, :, 0]
    Test1 = Test1_  # [x, y, z]

    center_pixel = center.copy()
    centre_world = pixel2world(center.copy(), fx, fy, u0, v0)

    centerlefttop = centre_world.copy()
    centerlefttop[:, 0, 0] = centerlefttop[:, 0, 0] - xy_thres
    centerlefttop[:, 0, 1] = centerlefttop[:, 0, 1] + xy_thres

    centerrightbottom = centre_world.copy()
    centerrightbottom[:, 0, 0] = centerrightbottom[:, 0, 0] + xy_thres
    centerrightbottom[:, 0, 1] = centerrightbottom[:, 0, 1] - xy_thres

    lefttop_pixel = world2pixel(centerlefttop, fx, fy, u0, v0)
    rightbottom_pixel = world2pixel(centerrightbottom, fx, fy, u0, v0)

    for i in range(len(Test1_)):
        Xmin = max(lefttop_pixel[i, 0, 0], 0)
        Ymin = max(rightbottom_pixel[i, 0, 1], 0)
        Xmax = min(rightbottom_pixel[i, 0, 0], 160 * 2 - 1)
        Ymax = min(lefttop_pixel[i, 0, 1], 120 * 2 - 1)

        Test1[i, :, 0] = Test1_[i, :, 0] * (Xmax - Xmin) / cropWidth + Xmin  # x
        Test1[i, :, 1] = Test1_[i, :, 1] * (Ymax - Ymin) / cropHeight + Ymin  # y
        Test1[i, :, 2] = source[i, :, 2] + center[i][0][2]

    labels = pixel2world(target_, fx, fy, u0, v0)
    outputs = pixel2world(Test1.copy(), fx, fy, u0, v0)

    errors = np.sqrt(np.sum((labels - outputs) ** 2, axis=2))

    return np.mean(errors)


def writeTxt(result, center):
    resultUVD_ = result.copy()
    resultUVD_[:, :, 0] = result[:, :, 1]
    resultUVD_[:, :, 1] = result[:, :, 0]
    resultUVD = resultUVD_  # [x, y, z]

    center_pixel = center.copy()
    centre_world = pixel2world(center.copy(), fx, fy, u0, v0)

    centerlefttop = centre_world.copy()
    centerlefttop[:, 0, 0] = centerlefttop[:, 0, 0] - xy_thres
    centerlefttop[:, 0, 1] = centerlefttop[:, 0, 1] + xy_thres

    centerrightbottom = centre_world.copy()
    centerrightbottom[:, 0, 0] = centerrightbottom[:, 0, 0] + xy_thres
    centerrightbottom[:, 0, 1] = centerrightbottom[:, 0, 1] - xy_thres

    lefttop_pixel = world2pixel(centerlefttop, fx, fy, u0, v0)
    rightbottom_pixel = world2pixel(centerrightbottom, fx, fy, u0, v0)

    for i in range(len(result)):
        Xmin = max(lefttop_pixel[i, 0, 0], 0)
        Ymin = max(rightbottom_pixel[i, 0, 1], 0)
        Xmax = min(rightbottom_pixel[i, 0, 0], 160 * 2 - 1)
        Ymax = min(lefttop_pixel[i, 0, 1], 120 * 2 - 1)

        resultUVD[i, :, 0] = resultUVD_[i, :, 0] * (Xmax - Xmin) / cropWidth + Xmin  # x
        resultUVD[i, :, 1] = resultUVD_[i, :, 1] * (Ymax - Ymin) / cropHeight + Ymin  # y
        resultUVD[i, :, 2] = result[i, :, 2] + center[i][0][2]

    resultReshape = resultUVD.reshape(len(result), -1)
    with open(os.path.join(save_dir, result_file), 'w') as f:
        for i in range(len(resultReshape)):
            for j in range(keypointsNumber * 3):
                f.write(str(resultReshape[i, j]) + ' ')
            f.write('\n')

    f.close()


if __name__ == '__main__':
    train()
    # test()



