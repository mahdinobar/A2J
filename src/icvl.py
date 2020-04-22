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

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

fx = 240.99
fy = 240.96
u0 = 160
v0 = 120

TestImgFrames = 702+894
#validIndex_train = np.load('../data/icvl/validIndex.npy')
validIndex_test = np.arange(TestImgFrames)
#TrainImgFrames = len(validIndex_train)

keypointsNumber = 16
cropWidth = 176
cropHeight = 176
batch_size = 8
xy_thres = 95
depth_thres = 150

save_dir = './result/ICVL'

try:
    os.makedirs(save_dir)
except OSError:
    pass


testingImageDir = '/home/mahdi/HVR/git_repos/A2J/data/icvl/test_seq_1and2_mat/'
keypointsfile = '../data/icvl/icvl_keypointsUVD_test.mat'
center_file = '../data/icvl/icvl_center_test.mat'
result_file = 'result_ICVL.txt'
model_dir = '../model/ICVL.pth'
MEAN = np.load('../data/icvl/icvl_mean.npy')
STD = np.load('../data/icvl/icvl_std.npy')

def pixel2world(x, fx, fy, ux, uy):
    x[:, :, 0] = (x[:, :, 0] - ux) * x[:, :, 2] / fx
    x[:, :, 1] = (x[:, :, 1] - uy) * x[:, :, 2] / fy
    return x

def world2pixel(x, fx, fy, ux, uy):
    x[:, :, 0] = x[:, :, 0] * fx / x[:, :, 2] + ux
    x[:, :, 1] = x[:, :, 1] * fy / x[:, :, 2] + uy
    return x
    

keypointsUVD_test = scio.loadmat(keypointsfile)['keypoints3D'].astype(np.float32)   

center_test = scio.loadmat(center_file)['centre_pixel'].astype(np.float32)

centre_test_world = pixel2world(center_test.copy(), fx, fy, u0, v0)

centerlefttop_test = centre_test_world.copy()
centerlefttop_test[:,0,0] = centerlefttop_test[:,0,0]-xy_thres
centerlefttop_test[:,0,1] = centerlefttop_test[:,0,1]+xy_thres

centerrightbottom_test = centre_test_world.copy()
centerrightbottom_test[:,0,0] = centerrightbottom_test[:,0,0]+xy_thres
centerrightbottom_test[:,0,1] = centerrightbottom_test[:,0,1]-xy_thres

test_lefttop_pixel = world2pixel(centerlefttop_test, fx, fy, u0, v0)
test_rightbottom_pixel = world2pixel(centerrightbottom_test, fx, fy, u0, v0)


def dataPreprocess(index, img, keypointsUVD, center, mean, std, lefttop_pixel, rightbottom_pixel, validIndex, xy_thres=95, depth_thres=150):
 
    imageOutputs = np.ones((cropHeight, cropWidth, 1), dtype='float32') 
    labelOutputs = np.ones((keypointsNumber, 3), dtype = 'float32') 
 
    new_Xmin = max(lefttop_pixel[index,0,0], 0)
    new_Ymin = max(rightbottom_pixel[index,0,1], 0)  
    new_Xmax = min(rightbottom_pixel[index,0,0], img.shape[1] - 1)
    new_Ymax = min(lefttop_pixel[index,0,1], img.shape[0] - 1)

    
    imCrop = img.copy()[int(new_Ymin):int(new_Ymax), int(new_Xmin):int(new_Xmax)]

    imgResize = cv2.resize(imCrop, (cropWidth, cropHeight), interpolation=cv2.INTER_NEAREST)

    imgResize = np.asarray(imgResize,dtype = 'float32')  # H*W*C

    imgResize[np.where(imgResize >= center[index][0][2] + depth_thres)] = center[index][0][2]
    imgResize[np.where(imgResize <= center[index][0][2] - depth_thres)] = center[index][0][2] 
    imgResize = (imgResize - center[index][0][2])

    imgResize = (imgResize - mean) / std

    label_xy = np.ones((keypointsNumber, 2), dtype = 'float32') 
    
    label_xy[:,0] = (keypointsUVD[validIndex[index],:,0].copy() - new_Xmin)*cropWidth/(new_Xmax - new_Xmin) # x
    label_xy[:,1] = (keypointsUVD[validIndex[index],:,1].copy() - new_Ymin)*cropHeight/(new_Ymax - new_Ymin) # y
    
    
    imageOutputs[:,:,0] = imgResize

    labelOutputs[:,1] = label_xy[:,0]
    labelOutputs[:,0] = label_xy[:,1] 
    
    labelOutputs[:,2] = (keypointsUVD[validIndex[index],:,2] - center[index][0][2])   # Z  
    
    imageOutputs = np.asarray(imageOutputs)
    imageNCHWOut = imageOutputs.transpose(2, 0, 1)  # [H, W, C] --->>>  [C, H, W]
    imageNCHWOut = np.asarray(imageNCHWOut)
    labelOutputs = np.asarray(labelOutputs)

    data, label = torch.from_numpy(imageNCHWOut), torch.from_numpy(labelOutputs)

    return data, label


######################   Pytorch dataloader   #################
class my_dataloader(torch.utils.data.Dataset):

    def __init__(self, trainingImageDir, center, lefttop_pixel, rightbottom_pixel, keypointsUVD, validIndex):

        self.trainingImageDir = trainingImageDir
        self.mean = MEAN
        self.std = STD
        self.center = center
        self.lefttop_pixel = lefttop_pixel
        self.rightbottom_pixel = rightbottom_pixel
        self.keypointsUVD = keypointsUVD
        self.validIndex = validIndex
        self.xy_thres = xy_thres
        self.depth_thres = depth_thres

    def __getitem__(self, index):
        print('index=',index)
        depth = scio.loadmat(self.trainingImageDir + str(self.validIndex[index]+1) + '.mat')['img']       
         
        data, label = dataPreprocess(index, depth, self.keypointsUVD, self.center, self.mean, self.std, \
            self.lefttop_pixel, self.rightbottom_pixel, self.validIndex, self.xy_thres, self.depth_thres)
       

        return data, label
    
    def __len__(self):
        return len(self.center)

      
test_image_datasets = my_dataloader(testingImageDir, center_test, test_lefttop_pixel, 
                                    test_rightbottom_pixel, keypointsUVD_test, validIndex_test)
test_dataloaders = torch.utils.data.DataLoader(test_image_datasets, batch_size = batch_size,
                                             shuffle = False, num_workers = 8)
      
def main():
    
    net = model.A2J_model(num_classes = keypointsNumber)
    net.load_state_dict(torch.load(model_dir)) 
    net = net.cuda()
    net.eval()
    
    post_precess = anchor.post_process(shape=[cropHeight//16,cropWidth//16],stride=16,P_h=None, P_w=None)
    
    output = torch.FloatTensor()
        
    for i, (img, label) in tqdm(enumerate(test_dataloaders)):    
        with torch.no_grad():
    
            img, label = img.cuda(), label.cuda()        
            heads = net(img)  
            pred_keypoints = post_precess(heads,voting=False)
            output = torch.cat([output,pred_keypoints.data.cpu()], 0)
    
    result = output.cpu().data.numpy()
    errTotal = errorCompute(result,keypointsUVD_test, center_test)
    writeTxt(result, center_test)
    
    print('Error:', errTotal)
    

def errorCompute(source, target, center):
    assert np.shape(source)==np.shape(target), "source has different shape with target"

    Test1_ = source.copy()
    target_ = target.copy()
    Test1_[:, :, 0] = source[:,:,1]
    Test1_[:, :, 1] = source[:,:,0]
    Test1 = Test1_  # [x, y, z]
    
    center_pixel = center.copy()
    centre_world = pixel2world(center.copy(), fx, fy, u0, v0)

    centerlefttop = centre_world.copy()
    centerlefttop[:,0,0] = centerlefttop[:,0,0]-xy_thres
    centerlefttop[:,0,1] = centerlefttop[:,0,1]+xy_thres
    
    centerrightbottom = centre_world.copy()
    centerrightbottom[:,0,0] = centerrightbottom[:,0,0]+xy_thres
    centerrightbottom[:,0,1] = centerrightbottom[:,0,1]-xy_thres
    
    lefttop_pixel = world2pixel(centerlefttop, fx, fy, u0, v0)
    rightbottom_pixel = world2pixel(centerrightbottom, fx, fy, u0, v0)


    for i in range(len(Test1_)):
        Xmin = max(lefttop_pixel[i,0,0], 0)
        Ymin = max(rightbottom_pixel[i,0,1], 0)  
        Xmax = min(rightbottom_pixel[i,0,0], 160*2 - 1)
        Ymax = min(lefttop_pixel[i,0,1], 120*2 - 1)

        Test1[i,:,0] = Test1_[i,:,0]*(Xmax-Xmin)/cropWidth + Xmin  # x
        Test1[i,:,1] = Test1_[i,:,1]*(Ymax-Ymin)/cropHeight + Ymin  # y
        Test1[i,:,2] = source[i,:,2] + center[i][0][2]

    # ########################################################################################################################
    # # plot
    # import matplotlib.pyplot as plt
    # import matplotlib
    #
    # idepth = test_image_datasets.load_depthmap(0)
    #
    # fig, ax = plt.subplots()
    # ax.imshow(idepth, cmap=matplotlib.cm.jet)
    #
    # ax.scatter(center[0][0][0], center[0][0][1], marker='+', c='yellow', s=200,
    #            label='refined center UVD')  # initial hand com in IMG
    # ax.scatter(target_[0, :, 0], target_[0, :, 1], marker='o', c='cyan', s=100,
    #            label='gt joints UVD')  # initial hand com in IMG
    # ax.scatter(Test1[0, :, 0], Test1[0, :, 1], marker='*', c='magenta', s=100,
    #            label='predicted joints UVD')  # initial hand com in IMG
    # ax.legend()
    # plt.show()
    # ########################################################################################################################



    labels = pixel2world(target_, fx, fy, u0, v0)
    outputs = pixel2world(Test1.copy(), fx, fy, u0, v0)

    errors = np.sqrt(np.sum((labels - outputs) ** 2, axis=2))

    return np.mean(errors)
   

def writeTxt(result, center):

    resultUVD_ = result.copy()
    resultUVD_[:, :, 0] = result[:,:,1]
    resultUVD_[:, :, 1] = result[:,:,0]
    resultUVD = resultUVD_  # [x, y, z]
    
    center_pixel = center.copy()
    centre_world = pixel2world(center.copy(), fx, fy, u0, v0)

    centerlefttop = centre_world.copy()
    centerlefttop[:,0,0] = centerlefttop[:,0,0]-xy_thres
    centerlefttop[:,0,1] = centerlefttop[:,0,1]+xy_thres
    
    centerrightbottom = centre_world.copy()
    centerrightbottom[:,0,0] = centerrightbottom[:,0,0]+xy_thres
    centerrightbottom[:,0,1] = centerrightbottom[:,0,1]-xy_thres
    
    lefttop_pixel = world2pixel(centerlefttop, fx, fy, u0, v0)
    rightbottom_pixel = world2pixel(centerrightbottom, fx, fy, u0, v0)

    for i in range(len(result)):
        Xmin = max(lefttop_pixel[i,0,0], 0)
        Ymin = max(rightbottom_pixel[i,0,1], 0)  
        Xmax = min(rightbottom_pixel[i,0,0], 160*2 - 1)
        Ymax = min(lefttop_pixel[i,0,1], 120*2 - 1)

        resultUVD[i,:,0] = resultUVD_[i,:,0]*(Xmax-Xmin)/cropWidth + Xmin  # x
        resultUVD[i,:,1] = resultUVD_[i,:,1]*(Ymax-Ymin)/cropHeight + Ymin  # y
        resultUVD[i,:,2] = result[i,:,2] + center[i][0][2]

    resultReshape = resultUVD.reshape(len(result), -1)
    with open(os.path.join(save_dir, result_file), 'w') as f:     
        for i in range(len(resultReshape)):
            for j in range(keypointsNumber*3):
                f.write(str(resultReshape[i, j])+' ')
            f.write('\n') 

    f.close()

def compute_mean_err(pred, gt):
    '''
    pred: (N, K, 3)
    gt: (N, K, 3)

    mean_err: (K,)
    '''
    N, K = pred.shape[0], pred.shape[1]
    err_dist = np.sqrt(np.sum((pred - gt)**2, axis=2))  # (N, K)
    return np.mean(err_dist, axis=0)

if __name__ == '__main__':
    main()
    # # load saved results
    # results = np.loadtxt('{}/{}'.format(save_dir, result_file))
    # est_3Djoints = results.reshape(1596, 16, 3)
    # gt_3Djoints = test_image_datasets.keypointsUVD
    # print('mean error per joint = {} mm'.format(compute_mean_err(est_3Djoints[:,:,:], gt_3Djoints[:,:,:])))
    # print('overall mean error = {} mm'.format(np.mean(compute_mean_err(est_3Djoints[:,:,:], gt_3Djoints[:,:,:]))))
    print('ended')
