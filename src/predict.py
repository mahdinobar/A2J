import cv2
import torch
import torch.utils.data
import numpy as np
import scipy.io as scio
import os
from PIL import Image
import model as model
import anchor as anchor
from tqdm import tqdm
import open3d as o3d


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# iPad
h = 480
w = 640
iw = 3088.0
ih = 2316.0
xscale = h / ih
yscale = w / iw
fx = 2883.24 * xscale
fy = 2883.24 * yscale
u0 = 1154.66 * xscale
v0 = 1536.17 * yscale

keypointsNumber = 16
cropWidth = 176
cropHeight = 176
batch_size = 1
xy_thres = 140
depth_thres = 150

save_dir = './result/iPad'

try:
    os.makedirs(save_dir)
except OSError:
    pass

testingImageDir = '/home/mahdi/HVR/hvr/data/iPad/set_1/'
result_file = 'result_iPad.txt'

model_dir = '../model/ICVL.pth'

center_file = '../data/icvl/icvl_center_test.mat'
MEAN = np.load('../data/icvl/icvl_mean.npy')
STD = np.load('../data/icvl/icvl_std.npy')


def pixel2world(x, fx, fy, ux, uy):
    x[:, :, 0] = (x[:, :, 0] - ux) * x[:, :, 2] / fx
    x[:, :, 1] = (x[:, :, 1] - uy) * x[:, :, 2] / fy
    return x


def world2pixel(x, fx, fy, ux, uy):
    x[:, :, 0] = x[:, :, 0] * fx / x[:, :, 2] + ux
    x[:, :, 1] = (x[:, :, 1] * fy / x[:, :, 2] + uy)
    return x



_center_test = scio.loadmat(center_file)['centre_pixel'].astype(np.float32)
center_test = np.expand_dims(_center_test[0], axis=0)

center_test[0, 0, :] = np.array((300., 215., 760.), dtype=np.float32)

del _center_test
centre_test_world = pixel2world(center_test.copy(), fx, fy, u0, v0)

centerlefttop_test = centre_test_world.copy()
centerlefttop_test[:, 0, 0] = centerlefttop_test[:, 0, 0] - xy_thres
centerlefttop_test[:, 0, 1] = centerlefttop_test[:, 0, 1] + xy_thres

centerrightbottom_test = centre_test_world.copy()
centerrightbottom_test[:, 0, 0] = centerrightbottom_test[:, 0, 0] + xy_thres
centerrightbottom_test[:, 0, 1] = centerrightbottom_test[:, 0, 1] - xy_thres

test_lefttop_pixel = world2pixel(centerlefttop_test, fx, fy, u0, v0)
test_rightbottom_pixel = world2pixel(centerrightbottom_test, fx, fy, u0, v0)


def dataPreprocess(index, img, center, mean, std, lefttop_pixel, rightbottom_pixel, depth_thres=150):
    imageOutputs = np.ones((cropHeight, cropWidth, 1), dtype='float32')

    new_Xmin = max(lefttop_pixel[index, 0, 0], 0)
    new_Ymin = max(rightbottom_pixel[index, 0, 1], 0)
    new_Xmax = min(rightbottom_pixel[index, 0, 0], img.shape[1] - 1)
    new_Ymax = min(lefttop_pixel[index, 0, 1], img.shape[0] - 1)

    imCrop = img.copy()[int(new_Ymin):int(new_Ymax), int(new_Xmin):int(new_Xmax)]
    ########################################################################################################################
    # plot cropped
    import matplotlib.pyplot as plt
    import matplotlib

    fig, ax = plt.subplots()
    ax.imshow(imCrop, cmap=matplotlib.cm.jet)

    ax.legend()
    plt.show()
    ########################################################################################################################


    imgResize = cv2.resize(imCrop, (cropWidth, cropHeight), interpolation=cv2.INTER_NEAREST)

    imgResize = np.asarray(imgResize, dtype='float32')  # H*W*C

    imgResize[np.where(imgResize >= center[index][0][2] + depth_thres)] = center[index][0][2]
    imgResize[np.where(imgResize <= center[index][0][2] - depth_thres)] = center[index][0][2]
    imgResize = (imgResize - center[index][0][2])

    imgResize = (imgResize - mean) / std

    imageOutputs[:, :, 0] = imgResize


    imageOutputs = np.asarray(imageOutputs)
    imageNCHWOut = imageOutputs.transpose(2, 0, 1)  # [H, W, C] --->>>  [C, H, W]
    imageNCHWOut = np.asarray(imageNCHWOut)

    data = torch.from_numpy(imageNCHWOut)

    return data

def _pixel2world(x, y, z, img_width, img_height, _fx, _fy, _cx, _cy):
    w_x = (x - _cx) * z / _fx
    w_y = (y - _cy) * z / _fy
    w_z = z
    return w_x, w_y, w_z

def depthmap2points(image, _fx, _fy, _cx, _cy):
    h, w = image.shape
    y, x = np.meshgrid(np.arange(w) + 1, np.arange(h) + 1)
    points = np.zeros((h, w, 3), dtype=np.float32)
    points[:, :, 0], points[:, :, 1], points[:, :, 2] = _pixel2world(x, y, image, w, h, _fx, _fy, _cx, _cy)
    return points
######################   Pytorch dataloader   #################
class my_dataloader(torch.utils.data.Dataset):

    def __init__(self, trainingImageDir, center, lefttop_pixel, rightbottom_pixel):
        self.trainingImageDir = trainingImageDir
        self.mean = MEAN
        self.std = STD
        self.center = center
        self.lefttop_pixel = lefttop_pixel
        self.rightbottom_pixel = rightbottom_pixel
        self.xy_thres = xy_thres
        self.depth_thres = depth_thres

    def __getitem__(self, index):
        print('index=', index)
        # depth = scio.loadmat(self.trainingImageDir + str(self.validIndex[index] + 1) + '.mat')['img']
        depth = np.loadtxt(self.trainingImageDir + 'iPad_1_Depth_1.txt') * 1000  # mm
        depth[depth > 760 + 30] = 1000.
        # _points = depthmap2points(depth, fx, fy, u0, v0)
        # _points = _points.reshape(_points.shape[0] * _points.shape[1], 3)
        # _pcl = o3d.geometry.PointCloud()
        # _pcl.points = o3d.utility.Vector3dVector(_points)
        # pcl, _ = _pcl.remove_statistical_outlier(nb_neighbors=100, std_ratio=.10)

        data = dataPreprocess(index, depth, self.center, self.mean, self.std, self.lefttop_pixel,
                              self.rightbottom_pixel, self.depth_thres)

        return data

    def __len__(self):
        return len(self.center)

    def load_depthmap(self, index):
        idepth = np.loadtxt(self.trainingImageDir + 'iPad_1_Depth_1.txt') * 1000  # mm
        idepth[idepth > 760 + 30] = 1000.
        return idepth


test_image_datasets = my_dataloader(testingImageDir, center_test, test_lefttop_pixel,
                                    test_rightbottom_pixel)
test_dataloaders = torch.utils.data.DataLoader(test_image_datasets, batch_size=batch_size,
                                               shuffle=False, num_workers=8)


def predict_iPad():
    net = model.A2J_model(num_classes=keypointsNumber)
    net.load_state_dict(torch.load(model_dir))
    net = net.cuda()
    net.eval()

    post_precess = anchor.post_process(shape=[cropHeight // 16, cropWidth // 16], stride=16, P_h=None, P_w=None)

    output = torch.FloatTensor()

    for i, (img) in tqdm(enumerate(test_dataloaders)):
        with torch.no_grad():
            img = img.cuda()
            heads = net(img)
            pred_keypoints = post_precess(heads, voting=False)
            output = torch.cat([output, pred_keypoints.data.cpu()], 0)

    result = output.cpu().data.numpy()
    plots(result, center_test)

    print('here!')


def plots(source, center):

    Test1_ = source.copy()
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
        Xmax = min(rightbottom_pixel[i, 0, 0], 320 * 2 - 1)
        Ymax = min(lefttop_pixel[i, 0, 1], 240 * 2 - 1)


        Test1[i, :, 0] = Test1_[i, :, 0] * (Xmax - Xmin) / cropWidth + Xmin  # x
        Test1[i, :, 1] = Test1_[i, :, 1] * (Ymax - Ymin) / cropHeight + Ymin  # y
        Test1[i, :, 2] = source[i, :, 2] + center[i][0][2]
    ########################################################################################################################
    # plot
    import matplotlib.pyplot as plt
    import matplotlib

    idepth = test_image_datasets.load_depthmap(0)

    fig, ax = plt.subplots()
    ax.imshow(idepth, cmap=matplotlib.cm.jet)

    ax.scatter(center[0][0][0], center[0][0][1], marker='+', c='yellow', s=200,
               label='refined center UVD')  # initial hand com in IMG
    ax.scatter(Test1[0, :, 0], Test1[0, :, 1], marker='*', c='lime', s=100,
               label='predicted joints UVD')  # initial hand com in IMG
    ax.legend()
    plt.show()
    ########################################################################################################################

    joints3D = pixel2world(Test1.copy(), fx, fy, u0, v0)


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
    predict_iPad()
    # # load saved results
    # results = np.loadtxt('{}/{}'.format(save_dir, result_file))
    # est_3Djoints = results.reshape(1596, 16, 3)
    # gt_3Djoints = test_image_datasets.keypointsUVD
    # print('mean error per joint = {} mm'.format(compute_mean_err(est_3Djoints[:,:,:], gt_3Djoints[:,:,:])))
    # print('overall mean error = {} mm'.format(np.mean(compute_mean_err(est_3Djoints[:,:,:], gt_3Djoints[:,:,:]))))
    print('ended')