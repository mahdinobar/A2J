import cv2
import torch
import torch.utils.data
import numpy as np
import scipy.io as scio
import os
from PIL import Image
from torch.autograd import Variable
import model as model
import anchor as anchor
from tqdm import tqdm
import struct


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
 # MSRA15 calibration data
fx = 241.42
fy = 241.42
u0 = 320./2
v0 = 240./2

# DataHyperParms
keypointsNumber = 21
cropWidth = 176
cropHeight = 176
batch_size = 8
xy_thres = 90 # mm
depth_thres = 150

save_dir = './result/MSRA15'

try:
    os.makedirs(save_dir)
except OSError:
    pass

testingImageDir = '/home/mahdi/HVR/git_repos/A2J/data/MSRA15'  # bin images
center_dir = '/home/mahdi/HVR/git_repos/A2J/data/msra_center' # in 3D coordinates
total_subject_num = 9
mode = 'test'
test_subject_id = 3
'''
 we compute mean/std on training set
 we first crop the original depth maps according to center points, which give us
  a hand-centered sub-image, then we compute the mean/std of all of these images.
'''
MEAN = np.load('../data/hands2017/hands2017_mean.npy')
STD = np.load('../data/hands2017/hands2017_std.npy')

# keypoint_file = '/home/mahdi/HVR/git_repos/A2J/data/MSRA15/P3/5/joint.txt' # shape: (K, num_joints, 3)
# put test error model and result file model at model_dir :
model_dir = '../model/HANDS2017.pth'
result_file = 'result_MSRA15.txt'


def pixel2world(x, fx, fy, ux, uy):
    x[:, :, 0] = (x[:, :, 0] - ux) * x[:, :, 2] / fx
    x[:, :, 1] = (x[:, :, 1] - uy) * x[:, :, 2] / fy
    return x


def world2pixel(x, fx, fy, ux, uy):
    x[:, :, 0] = x[:, :, 0] * fx / x[:, :, 2] + ux
    x[:, :, 1] = (x[:, :, 1] * fy / x[:, :, 2] + uy)
    return x


# keypointsUVD_test = scio.loadmat(keypoint_file)['keypoints3D'].astype(np.float32)
# center_test = scio.loadmat(center_file)['centre_pixel'].astype(np.float32)

def dataPreprocess(index, img, keypointsUVD, centerUVD, mean, std, lefttop_pixel, rightbottom_pixel, xy_thres=90,
                   depth_thres=75):
    imageOutputs = np.ones((cropHeight, cropWidth, 1), dtype='float32')
    labelOutputs = np.ones((keypointsNumber, 3), dtype='float32')

    new_Xmin = max(lefttop_pixel[index,0,0], 0)
    new_Ymin = max(rightbottom_pixel[index,0,1], 0)
    new_Xmax = min(rightbottom_pixel[index,0,0], img.shape[1] - 1)
    new_Ymax = min(lefttop_pixel[index,0,1], img.shape[0] - 1)

    imCrop = img[int(new_Ymin):int(new_Ymax), int(new_Xmin):int(new_Xmax)].copy()
    try:
        imgResize = cv2.resize(imCrop, (cropWidth, cropHeight), interpolation=cv2.INTER_NEAREST)
    except Exception:
        print('index=', index)
        print('imCrop', imCrop)
        print(new_Xmin,new_Xmax ,new_Ymin, new_Ymax)
# ########################################################################################################################
#     # plot detection crop
#     import matplotlib.pyplot as plt
#     import matplotlib
#
#     fig, ax = plt.subplots()
#     ax.imshow(imCrop, cmap=matplotlib.cm.jet)
#     plt.title('{:.2f}, {:.2f}, {:.2f}, {:.2f}'.format(new_Xmin,new_Xmax ,new_Ymin, new_Ymax))
#
#     plt.show()
# ########################################################################################################################

    imgResize = np.asarray(imgResize, dtype='float32')  # H*W*C

    imgResize[np.where(imgResize >= centerUVD[index][0][2] + depth_thres)] = centerUVD[index][0][2]
    imgResize[np.where(imgResize <= centerUVD[index][0][2] - depth_thres)] = centerUVD[index][0][2]
    imgResize = (imgResize - centerUVD[index][0][2])

    imgResize = (imgResize - mean) / std

    ## label
    label_xy = np.ones((keypointsNumber, 2), dtype='float32')
    label_xy[:, 0] = (keypointsUVD[index, :, 0].copy() - new_Xmin) * cropWidth / (new_Xmax - new_Xmin)  # x
    label_xy[:, 1] = (keypointsUVD[index, :, 1].copy() - new_Ymin) * cropHeight / (new_Ymax - new_Ymin)  # y

    imageOutputs[:, :, 0] = imgResize

    labelOutputs[:, 1] = label_xy[:, 0]
    labelOutputs[:, 0] = label_xy[:, 1]

    labelOutputs[:, 2] = (keypointsUVD[index, :, 2] - centerUVD[index][0][2])  # Z

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

    def __init__(self, ImgDir, center_dir, fx, fy, total_subject_num, mode, test_subject_id):
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

        # self.keypointsUVD = points2pixels(self.joints_world, self.img_width, self.img_height, self.fx, self.fy)


    def __getitem__(self, index):
        # depth = scio.loadmat(self.ImgDir + str(index+1) + '.mat')['depth']
        # def loadDepthMap(filename):
        #     """
        #     Read a depth-map from png raw data of NYU
        #     :param filename: file name to load
        #     :return: image data of depth image
        #     """
        #     img = Image.open(filename)
        #     # top 8 bits of depth are packed into green channel and lower 8 bits into blue
        #     assert len(img.getbands()) == 3
        #     r, g, b = img.split()
        #     r = np.asarray(r, np.int32)
        #     g = np.asarray(g, np.int32)
        #     b = np.asarray(b, np.int32)
        #     dpt = np.bitwise_or(np.left_shift(g, 8), b)
        #     imgdata = np.asarray(dpt, np.float32)
        #     return imgdata
        #
        # depth = loadDepthMap(self.ImgDir + 'depth_1_{:07d}'.format(index + 1) + '.png')

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

# ########################################################################################################################
#         # plot inputs
#         import matplotlib.pyplot as plt
#         import matplotlib
#
#         fig, ax = plt.subplots()
#         ax.imshow(depth, cmap=matplotlib.cm.jet)
#
#         ax.scatter(self.centerUVD[index, 0, 0], self.centerUVD[index, 0, 1], marker='+', c='yellow', s=150,
#                    label='center')  # initial hand com in IMG
#
#         ax.scatter(self.keypointsUVD[index, :, 0], self.keypointsUVD[index, :, 1], marker='o', c='cyan', s=100,
#                    label='gt joints')  # initial hand com in IMG
#         plt.show()
# ########################################################################################################################

        # self.centerUVD shape is (K,1,3) type float 32 raw center UVD
        data, label = dataPreprocess(index, depth, self.keypointsUVD, self.centerUVD, self.mean, self.std,
                                     self.lefttop_pixel, self.rightbottom_pixel, self.xy_thres, self.depth_thres)

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


test_image_datasets = my_dataloader(ImgDir=testingImageDir, center_dir=center_dir, fx=fx, fy=fy, total_subject_num=total_subject_num,
                                    mode=mode, test_subject_id=test_subject_id)
test_dataloaders = torch.utils.data.DataLoader(test_image_datasets, batch_size=batch_size,
                                               shuffle=False, num_workers=1)


def main():
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


def errorCompute(source, target, centre_world):
    assert np.shape(source) == np.shape(target), "source has different shape with target"

    Test1_ = source.copy()
    target_ = target.copy()
    Test1_[:, :, 0] = source[:, :, 1]
    Test1_[:, :, 1] = source[:, :, 0]
    Test1 = Test1_  # [x, y, z]

    # center_pixel = center.copy()
    # centre_world = pixel2world(center.copy(), fx, fy, u0, v0)

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
        Ymin = max(lefttop_pixel[i, 0, 1], 0)
        Xmax = min(rightbottom_pixel[i, 0, 0], 320 * 2 - 1)
        Ymax = min(rightbottom_pixel[i, 0, 1], 240 * 2 - 1)

        Test1[i, :, 0] = Test1_[i, :, 0] * (Xmax - Xmin) / cropWidth + Xmin  # x
        Test1[i, :, 1] = Test1_[i, :, 1] * (Ymax - Ymin) / cropHeight + Ymin  # y
        Test1[i, :, 2] = source[i, :, 2] + centre_world[i][0][2]

    labels = pixel2world(target_, fx, fy, u0, v0)
    outputs = pixel2world(Test1.copy(), fx, fy, u0, v0)

    errors = np.sqrt(np.sum((labels - outputs) ** 2, axis=2))

    return np.mean(errors)


def writeTxt(result, centre_world):
    resultUVD_ = result.copy()
    resultUVD_[:, :, 0] = result[:, :, 1]
    resultUVD_[:, :, 1] = result[:, :, 0]
    resultUVD = resultUVD_  # [x, y, z]

    # center_pixel = center.copy()
    # centre_world = pixel2world(center.copy(), fx, fy, u0, v0)

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
        Ymin = max(lefttop_pixel[i, 0, 1], 0)
        Xmax = min(rightbottom_pixel[i, 0, 0], 320 * 2 - 1)
        Ymax = min(rightbottom_pixel[i, 0, 1], 240 * 2 - 1)

        resultUVD[i, :, 0] = resultUVD_[i, :, 0] * (Xmax - Xmin) / cropWidth + Xmin  # x
        resultUVD[i, :, 1] = resultUVD_[i, :, 1] * (Ymax - Ymin) / cropHeight + Ymin  # y
        resultUVD[i, :, 2] = result[i, :, 2] + centre_world[i][0][2]

    resultReshape = resultUVD.reshape(len(result), -1)

    with open(os.path.join(save_dir, result_file), 'w') as f:
        for i in range(len(resultReshape)):
            for j in range(keypointsNumber * 3):
                f.write(str(resultReshape[i, j]) + ' ')
            f.write('\n')

    f.close()


def compute_mean_err(pred, gt):
    '''
    pred: (N, K, 3)
    gt: (N, K, 3)

    mean_err: (K,)
    '''
    N, K = pred.shape[0], pred.shape[1]
    err_dist = np.sqrt(np.sum((pred - gt) ** 2, axis=2))  # (N, K)
    return np.mean(err_dist, axis=0)


if __name__ == '__main__':
    main()
    # # load saved results
    # results = np.loadtxt('{}/{}'.format(save_dir, result_file))
    # est_3Djoints = results.reshape(8252, keypointsNumber, 3)
    # gt_3Djoints = test_image_datasets.keypointsUVD
    # print('mean error per joint = {} mm'.format(compute_mean_err(est_3Djoints[:,:,:], gt_3Djoints[:,:,:])))
    # print('overall mean error = {} mm'.format(np.mean(compute_mean_err(est_3Djoints[:,:,:], gt_3Djoints[:,:,:]))))
    # print('ended')

