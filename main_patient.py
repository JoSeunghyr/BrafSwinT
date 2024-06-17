import os
from PIL import Image
import random
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import math
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn as nn
import cv2
import SimpleITK as sitk
import scipy.io as sio
from timm.data import create_transform
import timm.data.transforms as timm_transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision import datasets, transforms
# from st.models.swin_transformer_v2cponly import SwinTransformerV2
from st.models.brafswin_p import BrafSwin
from st.losses.cls_loss import LabelSmoothCELoss, BCE_LOSS, SoftmaxEQLV2Loss, SoftTargetCrossEntropy

import numpy as np

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
def cosine_anneal_schedule(t, nb_epoch, lr):
    cos_inner = np.pi * (t % (nb_epoch))  # t - 1 is used when t has 1-based indexing.
    cos_inner /= (nb_epoch)
    cos_out = np.cos(cos_inner) + 1
    return float(lr / 2 * cos_out)


def build_transform(typ):
    resize_im = True
    AUG_COLOR_JITTER = 0.4
    AUG_AUTO_AUGMENT = 'rand-m9-mstd0.5-inc1'
    AUG_REPROB = 0.25
    AUG_REMODE = 'pixel'
    AUG_RECOUNT = 1
    DATA_INTERPOLATION = 'bicubic'
    DATA_IMG_SIZE = 256
    TEST_CROP = True

    if typ == 'train':
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=256,
            is_training=True,
            color_jitter=AUG_COLOR_JITTER if AUG_COLOR_JITTER > 0 else None,
            auto_augment=AUG_AUTO_AUGMENT if AUG_AUTO_AUGMENT != 'none' else None,
            re_prob=AUG_REPROB,
            re_mode=AUG_REMODE,
            re_count=AUG_RECOUNT,
            interpolation=DATA_INTERPOLATION,
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(DATA_IMG_SIZE, padding=4)
        return transform

    t = []
    if resize_im:
        if TEST_CROP:
            size = int((256 / 256) * DATA_IMG_SIZE)
            t.append(
                # transforms.Resize(size, interpolation=InterpolationMode.BICUBIC),
                transforms.Resize(size, interpolation=timm_transforms._pil_interp(DATA_INTERPOLATION)),
                # to maintain same ratio w.r.t. 224 images
            )
            t.append(transforms.CenterCrop(DATA_IMG_SIZE))
        else:
            t.append(
                transforms.Resize((DATA_IMG_SIZE, DATA_IMG_SIZE),
                                  interpolation=timm_transforms._pil_interp(DATA_INTERPOLATION))
            )

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.3, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        self.alpha = torch.tensor(alpha).cuda()
        self.gamma = gamma
        self.size_average = size_average

    def forward(self, pred, target):
        pred = torch.softmax(pred,dim=1)
        class_mask = torch.zeros(pred.shape[0], pred.shape[1]).cuda()

        class_mask.scatter_(1, target.view(-1, 1).long(), 1.)

        probs = (pred * class_mask).sum(dim=1).view(-1, 1)
        probs = probs.clamp(min=0.0001, max=1.0)

        log_p = probs.log()

        alpha = torch.ones(pred.shape[0], pred.shape[1]).cuda()
        alpha[:, 0] = alpha[:, 0] * (1 - self.alpha)
        alpha[:, 1] = alpha[:, 1] * self.alpha

        alpha = (alpha * class_mask).sum(dim=1).view(-1, 1)

        batch_loss = -alpha * (torch.pow((1 - probs), self.gamma)) * log_p

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()

        return loss


class KZDataset():
    def __init__(self, path_0=None,path_1=None, ki=0, K=5, typ='train', oridata=None, rand=False):
        self.data_info_0 = self.get_img_info(path_0)
        self.data_info_1 = self.get_img_info(path_1)

        leng_0 = len(self.data_info_0)
        every_z_len_0 = leng_0 / K
        leng_1 = len(self.data_info_1)
        every_z_len_1 = leng_1 / K
        if typ == 'val':
            self.data_info_0 = self.data_info_0[math.ceil(every_z_len_0 * ki) : math.ceil(every_z_len_0 * (ki+1))]
            self.data_info_1 = self.data_info_1[math.ceil(every_z_len_1 * ki) : math.ceil(every_z_len_1 * (ki+1))]

            self.data_info = self.data_info_0 + self.data_info_1
            # self.data_info = self.data_info_0 + self.data_info_1
        elif typ == 'train':
            self.data_info_0 = self.data_info_0[: math.ceil(every_z_len_0 * ki)] + self.data_info_0[math.ceil(every_z_len_0 * (ki+1)) :]
            self.data_info_1 = self.data_info_1[: math.ceil(every_z_len_1 * ki)] + self.data_info_1[math.ceil(every_z_len_1 * (ki+1)) :]
        
            self.data_info = self.data_info_0 + self.data_info_1
        print(len(self.data_info))
        if rand:
	        random.seed(2)
        	random.shuffle(self.data_info)

        self.typ = typ
        self.transform = build_transform(self.typ)
        self.oridata = oridata

    def __getitem__(self, index):

        img_pth, label, cp = self.data_info[index]
        cp = [int(i) for i in cp]
        cp = np.array(cp)
        patient = img_pth[0].split('\\')[-1][:-4].split('ROI')[0]

        fea_path1 = os.path.join(r'', patient + 'ROI' + '1' + '.mat')
        fea_path2 = os.path.join(r'', patient + 'ROI' + '2' + '.mat')
        if self.oridata == True:
            img1 = sitk.ReadImage(img_pth[0])
            img1 = sitk.GetArrayFromImage(img1)
            img1 = img1[:, :, np.newaxis]
            img1 = np.resize(img1, (256, 256, 3))
            h, w, c = img1.shape
            norm = np.zeros((h, w, c), dtype=np.float32)
            img1 = cv2.normalize(img1, norm, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

            img2 = sitk.ReadImage(img_pth[1])
            img2 = sitk.GetArrayFromImage(img2)
            img2 = img2[:, :, np.newaxis]
            img2 = np.resize(img2, (256, 256, 3))
            img2 = cv2.normalize(img2, norm, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        else:
            img1 = Image.open(img_pth[0]).convert('RGB')
            img1 = self.transform(img1)
            img1 = np.array(img1)  # ndarray: 3.224.224
            img1 = img1.transpose(1, 2, 0)  # ndarray: 224.224.3

            img2 = Image.open(img_pth[1]).convert('RGB')
            img2 = self.transform(img2)
            img2 = np.array(img2)  # ndarray: 3.224.224
            img2 = img2.transpose(1, 2, 0)  # ndarray: 224.224.3

        mat1 = sio.loadmat(fea_path1)
        r_features1 = mat1['FeatureAll']
        r_features1 = torch.Tensor(r_features1)

        mat2 = sio.loadmat(fea_path2)
        r_features2 = mat2['FeatureAll']
        r_features2 = torch.Tensor(r_features2)
        return img1, img2, label, cp, patient, r_features1, r_features2

    def __len__(self):
        return len(self.data_info)

    @staticmethod
    def get_img_info(csv_path):

        data_info = []
        data = open(csv_path, 'r')
        data_lines = data.readlines()
        imgnum = len(data_lines)
        for i in range(0, imgnum, 2):
            data_line = data_lines[i].replace(",", " ").replace("\n", "")
            data_line = data_line.split()
            data_line1 = data_lines[i + 1].replace(",", " ").replace("\n", "")
            data_line1 = data_line1.split()
            img1_pth = data_line[0]
            img2_pth = data_line1[0]
            img_cppth = [img1_pth, img2_pth]
            label = int(data_line[1])
            cp = data_line[3:6]
            data_info.append((img_cppth, label, cp))
        return data_info


def train(nb_epoch, batch_size, store_name, resume=False, start_epoch=0):
    # setup output
    exp_dir = store_name
    try:
        os.stat(exp_dir)
    except:
        os.makedirs(exp_dir)
    K = 5
    for ki in range(K):
        model_path = r'' % ki
        
        use_cuda = torch.cuda.is_available()
        print(use_cuda)

        # Data
        print('==> Preparing data..')

        trainset = KZDataset(path_0=r'',
                                path_1=r'',
                                ki=ki, K=K, typ='train', oridata=False, rand=False)
        valset = KZDataset(path_0=r'',
                                path_1=r'',
                                ki=ki, K=K, typ='val', oridata=False, rand=False)

        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)
        if resume:
            v = torch.load(model_path)
        else:
            v = SwinTransformerV2_PF(img_size=256, num_classes=2, in_chans=3, pretrained=True, pretrained_model='swinv2_tiny_patch4_window8_256.pth')

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        v.to(device)
#        cudnn.benchmark = True
        CELoss = FocalLoss()
        # CELoss = nn.CrossEntropyLoss(weight=w)
        # CELoss = SoftmaxEQLV2Loss(num_classes=2)
        optimizer = optim.SGD([
            {'params': v.parameters(), 'lr': 0.004},
        ],
            momentum=0.9, weight_decay=5e-4)
        # optimizer = torch.optim.Adam(v.parameters(), lr=0.002)
    
        max_val_acc = 0
        lr = [0.004, 0.004, 0.004, 0.004, 0.004, 0.004, 0.004, 0.0004]
        for epoch in range(start_epoch, nb_epoch):
            print('\nEpoch: %d' % epoch)
            v.train()
            train_loss = 0
            correct = 0
            total = 0
            idx = 0
            for batch_idx, (inputs1, inputs2, targets, cp, patient, features1, features2) in enumerate(trainloader):
                # targets = 1-targets
                inputs1 = inputs1.permute(0, 3, 1, 2).float()
                inputs2 = inputs2.permute(0, 3, 1, 2).float()
#                print(inputs.shape) #8 32 3 256 256
#                 inputs1 = inputs[:,:,::3,:,256:]
#                 inputs2 = inputs[:,:,::3,:,:256]
#                print(inputs.shape,targets)
                idx = batch_idx
                cp = torch.from_numpy(np.array(cp))
                if inputs1.shape[0] < batch_size:
                    continue
                if use_cuda:
                    inputs1, inputs2, targets, cp, features1, features2 = inputs1.to(device), inputs2.to(device), targets.to(device), cp.to(device), features1.to(device), features2.to(device)
                inputs1, inputs2, targets, cp, features1, features2 = Variable(inputs1), Variable(inputs2), Variable(targets), Variable(cp), Variable(features1), Variable(features2)
                # cp = torch.unsqueeze(cp,0)
                # update learning rate
                for nlr in range(len(optimizer.param_groups)):
                    optimizer.param_groups[nlr]['lr'] = cosine_anneal_schedule(epoch, nb_epoch, lr[nlr])
                optimizer.zero_grad()
                output_concat = v(inputs1, inputs2, cp, features1, features2)
                concat_loss = CELoss(output_concat, targets)
                concat_loss.backward()
                optimizer.step()

                _, predicted = torch.max(output_concat.data, 1)
                total += targets.size(0)
                correct += predicted.eq(targets.data).cpu().sum()
    
                train_loss += concat_loss.item()

    
                if batch_idx % 2 == 0:
                    # viz.line([[train_loss / (batch_idx + 1), 100. * float(correct) / total]],
                    #          [[epoch*400+batch_idx, epoch*400+batch_idx]], win='train_loss', update='append')
                    print(
                        'K-fold %d,Step: %d| Loss: %.3f | Acc: %.3f%% (%d/%d)' % (
                        ki,batch_idx,train_loss / (batch_idx + 1),
                        100. * float(correct) / total, correct, total))
            print('lr:', optimizer.param_groups[0]['lr'])
    
            train_acc = 100. * float(correct) / total
            train_loss = train_loss / (idx + 1)
            with open(exp_dir + '/results_train_np_%d.txt'%ki, 'a') as file:
                file.write(
                    'K-fold %d, Iteration %d | train_acc = %.5f | train_loss = %.5f\n' % (
                    ki,epoch, train_acc, train_loss))
                
            
            torch.cuda.empty_cache()
            testloader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=0)
            test_loss = 0
            correct = 0
            total = 0
            idx = 0
            for batch_idx, (inputs1, inputs2, targets, cp, patient, features1, features2) in enumerate(testloader):
                # targets = 1-targets
                inputs1 = inputs1.permute(0, 3, 1, 2).float()
                inputs2 = inputs2.permute(0, 3, 1, 2).float()
                # inputs1 = inputs[:,:,::3,:,256:]
                # inputs2 = inputs[:,:,::3,:,:256]
                idx = batch_idx
                if use_cuda:
                    inputs1, inputs2, targets, cp, features1, features2 = inputs1.to(device), inputs2.to(device), targets.to(device), cp.to(device), features1.to(device), features2.to(device)
                inputs1, inputs2, targets, cp, features1, features2 = Variable(inputs1), Variable(inputs2), Variable(targets), Variable(cp), Variable(features1), Variable(features2)
                # mask = targets.unsqueeze(1).unsqueeze(2).repeat(1,8,8).bool()
                v.eval()
                output = v(inputs1, inputs2, cp, features1, features2)
                loss = CELoss(output, targets)
        
                test_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                total += targets.size(0)
                correct += predicted.eq(targets.data).cpu().sum()
                print('label:',targets.data.cpu(),'pred:',predicted.cpu())
        
                if batch_idx % 50 == 0:
                    print('Step: %d | Loss: %.3f | Acc: %.3f%% (%d/%d)' % (
                    batch_idx, test_loss / (batch_idx + 1), 100. * float(correct) / total, correct, total))

            test_acc = 100. * float(correct) / total
            test_loss = test_loss / (idx + 1)
            if test_acc > max_val_acc:
                max_val_acc = test_acc
                v.cpu()
                torch.save(v, './' + store_name + '/model_acc_cp_fold_%d.pth' % ki)
                v.to(device)
            elif epoch == nb_epoch-1:
                v.cpu()
                torch.save(v, './' + store_name + '/model_epoch_cp_fold_%d.pth' % ki)
                v.to(device)
            with open(exp_dir + '/results_val_np_%d.txt' % ki, 'a') as file:
                file.write(
                    'K-fold %d, Iteration %d | test_acc/max_acc = %.5f/%.5f | test_loss = %.5f\n' % (
                        ki, epoch, test_acc, max_val_acc, test_loss))

        torch.cuda.empty_cache()

train(nb_epoch=300,             # number of epoch
         batch_size=16,         # batch size
         store_name='bird_bi_aug_fea_pre_1225',     # folder for output
         resume=False,          # resume training from checkpoint
         start_epoch=76)         # the start epoch number when you resume the training

