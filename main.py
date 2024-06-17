import os
from PIL import Image
import random
import torch
import math
import cv2
from torch.autograd import Variable
from tensorboardX import SummaryWriter
import torch.nn as nn
import SimpleITK as sitk
import scipy.io as sio

from st.models.brafswin import BrafSwin
import timm
from timm.data import create_transform
import timm.data.transforms as timm_transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode
from st.losses.cls_loss import LabelSmoothCELoss, BCE_LOSS, SoftmaxEQLV2Loss, SoftTargetCrossEntropy

import numpy as np

os.environ['CUDA_VISIBLE_DEVICES'] = '0'      # 本机GPU内存不够
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
    def __init__(self, alpha=0.4, gamma=2, size_average=True):
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
    def __init__(self, path_0=None,path_1=None, ki=0, K=10, typ=None, oridata=None, rand=False):
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
	        random.seed(1)
        	random.shuffle(self.data_info)

        self.typ = typ
        self.transform = build_transform(self.typ)
        self.oridata = oridata
    def __getitem__(self, index):

        img_pth, label, cp = self.data_info[index]
        cp = [int(i) for i in cp]
        cp = np.array(cp)
        patient = img_pth.split('\\')[-1][:-4].split('ROI')[0]
        mode = img_pth.split('\\')[-1][:-4].split('ROI')[1]
        idx_list = []
        for item in mode:
            idx_list.append(item)
        fea_path = os.path.join(r'', patient + 'ROI' + idx_list[0] + '.mat')
        if self.oridata == True:
            img = sitk.ReadImage(img_pth)
            img = sitk.GetArrayFromImage(img)
            img = img[:, :, np.newaxis]
            img = np.resize(img, (256, 256, 3))
            h, w, c = img.shape
            norm = np.zeros((h, w, c), dtype=np.float32)
            img = cv2.normalize(img, norm, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        else:
            img = Image.open(img_pth).convert('RGB')
            img = self.transform(img)
            img = np.array(img)  # ndarray: 3.224.224
            img = img.transpose(1, 2, 0)  # ndarray: 224.224.3

        mat = sio.loadmat(fea_path)
        r_features = mat['FeatureAll']
        r_features = torch.Tensor(r_features)
        return img, label, cp, patient, r_features

    def __len__(self):
        return len(self.data_info)

    @staticmethod
    def get_img_info(csv_path):

        data_info = []
        data = open(csv_path, 'r')
        data_lines = data.readlines()
        for data_line in data_lines:
            data_line = data_line.replace(",", " ").replace("\n", "")
            data_line = data_line.split()
            img_pth = data_line[0]
            label = int(data_line[1])
            cp = data_line[2:6]
            data_info.append((img_pth, label, cp))
        return data_info   
    
    
def train(nb_epoch, batch_size, store_name, resume=False, start_epoch=0, model_path=None):
    # setup output
    exp_dir = store_name
    try:
        os.stat(exp_dir)
    except:
        os.makedirs(exp_dir)
    K = 10
    for ki in range(K):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
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

            v = BrafSwin(img_size=256, num_classes=2, in_chans=3, pretrained=True, pretrained_model='swinv2_tiny_patch4_window8_256.pth')


        v.to(device)
#        cudnn.benchmark = True
        CELoss = SoftmaxEQLV2Loss(num_classes=2)
        # CELoss = FocalLoss()
        # CELoss = nn.CrossEntropyLoss()

        optimizer = torch.optim.Adam(v.parameters(), lr=0.002)
        CosineLR = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200, eta_min=0)
        max_val_acc = 0
        # lr = [0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.0002] #学习率的动态调整
        lr = [0.002]
        writer = SummaryWriter('./' + store_name + '/LOGS/log_tensorboardX')
        for epoch in range(start_epoch, nb_epoch):
            print('\nEpoch: %d' % epoch)
            print('lr:', optimizer.param_groups[0]['lr'])
            v.train()
            train_loss = 0
            correct = 0
            total = 0
            idx = 0
            for batch_idx, (inputs, targets, cp, patient, features) in enumerate(trainloader): #进行序列化(每一个元素弄上标号)

                inputs = inputs.permute(0,3,1,2).float()

                idx = batch_idx

                if inputs.shape[0] < batch_size:
                    continue
                if use_cuda:
                    inputs, targets, cp, features = inputs.to(device), targets.to(device), cp.to(device), features.to(device)
                inputs, targets, cp, features = Variable(inputs), Variable(targets), Variable(cp), Variable(features)

                optimizer.zero_grad()
                output_concat = v(inputs, cp, features)
                concat_loss = CELoss(output_concat, targets)
                concat_loss.backward()
                optimizer.step()

                _, predicted = torch.max(output_concat.data, 1)
                total += targets.size(0)
                correct += predicted.eq(targets.data).cpu().sum()

                train_loss += concat_loss.item()

                writer.add_scalar('Train/Loss', concat_loss.item(), epoch * len(trainloader) + batch_idx)
                if batch_idx % 10 == 0:

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
            for batch_idx, (inputs, targets, cp, patient, features) in enumerate(testloader):

                inputs = inputs.permute(0,3,1,2).float()
                idx = batch_idx

                if use_cuda:
                    inputs, targets, cp, features = inputs.to(device), targets.to(device), cp.to(device), features.to(device)
                inputs, targets, cp, features = Variable(inputs), Variable(targets), Variable(cp), Variable(features)

                v.eval()
                output = v(inputs, cp, features)
                loss = CELoss(output, targets)
        
                test_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                total += targets.size(0)
                correct += predicted.eq(targets.data).cpu().sum()

                writer.add_scalar('Test/Loss', loss.item(), epoch * len(testloader) + batch_idx)
                print('label:', targets.data.cpu(), 'pred:', predicted.cpu())

                if batch_idx % 10 == 0:
                    print('Step: %d | Loss: %.3f | Acc: %.3f%% (%d/%d)' % (
                    batch_idx, test_loss / (batch_idx + 1), 100. * float(correct) / total, correct, total))
            CosineLR.step()

            test_acc = 100. * float(correct) / total
            test_loss = test_loss / (batch_idx + 1)
            if test_acc > max_val_acc:
                max_val_acc = test_acc
                v.cpu()
                torch.save(v, './' + store_name + '/model_ce_5fold_%d.pth'%ki)
                v.to(device)
            with open(exp_dir + '/results_val_np_%d.txt' % ki, 'a') as file:
                file.write(
                    'K-fold %d, Iteration %d | test_acc/max_acc = %.5f/%.5f | test_loss = %.5f\n' % (
                    ki, epoch, test_acc, max_val_acc, test_loss))

        writer.close()
        torch.cuda.empty_cache()

train(nb_epoch=70,             # number of epoch
         batch_size=16,         # batch size
         store_name='birdaug_fea_pre',     # folder for output
         resume=False,          # resume training from checkpoint
         start_epoch=0,         # the start epoch number when you resume the training
         model_path=None)         # the saved model where you want to resume the training
