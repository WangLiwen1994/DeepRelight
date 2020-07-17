import os
import os.path
from os import listdir
from os.path import join

import cv2
import numpy as np
import torch.utils.data as data
from PIL import Image, ImageOps, ImageEnhance
from torchvision.transforms import ToTensor, ColorJitter, Normalize
import torchvision.transforms as transforms
from data.base_dataset import BaseDataset, get_params, get_transform, normalize
from data.image_folder import make_dataset
import random


class TraindataFromTrack3(data.Dataset):
    def initialize(self, opt):
        return

    def __init__(self):
        super(TraindataFromTrack3, self).__init__()

        self.image_folder = "dataset/track3_train"
        self.image_filenames_input = [join(self.image_folder, x) for x in listdir(self.image_folder) if
                                      self.is_image_file(x)]

        self.transform = transforms.Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.augmentation = ColorJitter(0.2, 0.2, 0.2, 0.2)
        print("data folder: %s; find %d images" % (self.image_folder, len(self.image_filenames_input)))

    def is_image_file(swlf, filename):
        return any(filename.endswith(extension) for extension in [".bmp", ".png", ".jpg", ".jpeg"])

    def is_image_GTfile(self, filename):
        return any(filename.endswith(extension) for extension in ["_4500_E.png"])

    def name(self):
        return "TraindataFromTrack3: any2one"

    def getShadow(self, PIL_img):
        img_BGR = cv2.cvtColor(np.asarray(PIL_img), cv2.COLOR_RGB2BGR)
        RGBsum = np.sum(img_BGR, axis=2) / 3.0
        mask = np.zeros_like(RGBsum)
        mask[RGBsum < 15] = 1
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)
        return mask

    def randomCrop(self, img, mask, width, height):
        img = cv2.cvtColor(np.asarray(img),cv2.COLOR_RGB2BGR)
        mask = cv2.cvtColor(np.asarray(mask),cv2.COLOR_RGB2BGR)
        assert img.shape[0] >= height
        assert img.shape[1] >= width
        assert img.shape[0] == mask.shape[0]
        assert img.shape[1] == mask.shape[1]
        x = random.randint(0, img.shape[1] - width)
        y = random.randint(0, img.shape[0] - height)
        img = img[y:y + height, x:x + width]
        mask = mask[y:y + height, x:x + width]
        img = Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
        mask = Image.fromarray(cv2.cvtColor(mask,cv2.COLOR_BGR2RGB))
        return img, mask

    def get_img_flipAug(self, index, in_setting=None):
        InputFile = self.image_filenames_input[index]
        InputPath = os.path.dirname(InputFile)
        InFileName = os.path.basename(InputFile)
        if in_setting is None:
            in_setting = InFileName[8:]
        else:
            InFileName = InFileName[:8]+in_setting

        if bool(random.getrandbits(1)):
            isUseFlipAug = False
            InTargetName = InFileName[:8] + "_4500_E.png"
        else:
            isUseFlipAug = True
            InTargetName = InFileName[:8] + "_4500_W.png"

        InTargetFile = join(InputPath, InTargetName)
        InputFile = join(InputPath, InFileName)
        img_input = Image.open(InputFile).resize((512, 512))
        img_target = Image.open(InTargetFile).resize((512, 512))
        if isUseFlipAug:
            img_input = ImageOps.mirror(img_input)
            img_target = ImageOps.mirror(img_target)

        return img_input, img_target, in_setting

    def __getitem__(self, index):
        img_input, img_target, img_in_setting = self.get_img_flipAug(index)
        # isRotate = random.uniform(0, 1) < 0.3 # 30% probability for blend image
        # isZoomCrop = random.uniform(0, 1) < 0.5 # 50% probability for zoom and crop
        isAdjBright = random.uniform(0, 1) < 0.5 # 50% probability for adjust brightness
        # ###  Augmentation: blend, zoom
        # # if isBlend:
        # #     alpha = random.uniform(0.3, 0.7)
        # #     img_input_2, img_target_2, _ = self.get_img_flipAug(random.randint(0, self.__len__()-1), img_in_setting)
        # #     img_input = Image.blend(img_input,img_input_2,alpha)
        # #     img_target = Image.blend(img_target,img_target_2,alpha)
        #
        # if bool(random.getrandbits(1)): # 50% probability to adjust brightness
        #     factor_brightness = random.uniform(0.7, 1.2)
        #     enhancer = ImageEnhance.Brightness(img_input)
        #     img_input = enhancer.enhance(factor_brightness)
        #
        # if 1 or bool(random.getrandbits(1)):
        #     factor_color = random.uniform(0.8, 1)
        #     enhancer = ImageEnhance.Color(img_input)
        #     img_input = enhancer.enhance(factor_color)

        #
        # if isZoomCrop:  # 50% probability for zoom and crop image
        #     enlargeratio = random.uniform(1.01, 1.15)
        #     newsize = int(enlargeratio * 512)
        #     img_input, img_target = self.randomCrop(img_input.resize((newsize, newsize)),
        #                                             img_target.resize((newsize, newsize)), width=512, height=512)
        # if isRotate: # 30% probability to rotate image
        #     rotateangle = random.uniform(-5, 5)
        #     img_input = img_input.rotate(rotateangle)
        #     img_target = img_target.rotate(rotateangle)


        A_tensor = self.transform(img_input)
        B_tensor = self.transform(img_target)
        input_dict = {'input': A_tensor, 'target': B_tensor}

        return input_dict  # self.transform(shadow2remove).float().repeat([3, 1, 1])

    def __len__(self):
        return len(self.image_filenames_input)


class AlignedDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot

        ### input A (label maps)
        dir_A = '_A' if self.opt.label_nc == 0 else '_label'
        self.dir_A = os.path.join(opt.dataroot, opt.phase + dir_A)
        self.A_paths = sorted(make_dataset(self.dir_A))

        ### input B (real images)
        if opt.isTrain or opt.use_encoded_image:
            dir_B = '_B' if self.opt.label_nc == 0 else '_img'
            self.dir_B = os.path.join(opt.dataroot, opt.phase + dir_B)
            self.B_paths = sorted(make_dataset(self.dir_B))

        ### instance maps
        if not opt.no_instance:
            self.dir_inst = os.path.join(opt.dataroot, opt.phase + '_inst')
            self.inst_paths = sorted(make_dataset(self.dir_inst))

        ### load precomputed instance-wise encoded features
        if opt.load_features:
            self.dir_feat = os.path.join(opt.dataroot, opt.phase + '_feat')
            print('----------- loading features from %s ----------' % self.dir_feat)
            self.feat_paths = sorted(make_dataset(self.dir_feat))

        self.dataset_size = len(self.A_paths)

    def __getitem__(self, index):
        ### input A (label maps)
        A_path = self.A_paths[index]
        A = Image.open(A_path)
        params = get_params(self.opt, A.size)
        if self.opt.label_nc == 0:
            transform_A = get_transform(self.opt, params)
            A_tensor = transform_A(A.convert('RGB'))
        else:
            transform_A = get_transform(self.opt, params, method=Image.NEAREST, normalize=False)
            A_tensor = transform_A(A) * 255.0

        B_tensor = inst_tensor = feat_tensor = 0
        ### input B (real images)
        if self.opt.isTrain or self.opt.use_encoded_image:
            B_path = self.B_paths[index]
            B = Image.open(B_path).convert('RGB')
            transform_B = get_transform(self.opt, params)
            B_tensor = transform_B(B)

        ### if using instance maps        
        if not self.opt.no_instance:
            inst_path = self.inst_paths[index]
            inst = Image.open(inst_path)
            inst_tensor = transform_A(inst)

            if self.opt.load_features:
                feat_path = self.feat_paths[index]
                feat = Image.open(feat_path).convert('RGB')
                norm = normalize()
                feat_tensor = norm(transform_A(feat))

        input_dict = {'label': A_tensor, 'inst': inst_tensor, 'image': B_tensor,
                      'feat': feat_tensor, 'path': A_path}

        return input_dict

    def __len__(self):
        return len(self.A_paths) // self.opt.batchSize * self.opt.batchSize

    def name(self):
        return 'AlignedDataset'
