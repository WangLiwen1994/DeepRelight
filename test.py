import os
from os import listdir
from os.path import join, basename
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from models.models import create_model
from options.test_options import TestOptions
from PIL import Image, ImageOps, ImageEnhance
import cv2
import numpy as np


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".bmp", ".png", ".jpg", ".jpeg"])


def tensor2im(image_tensor, imtype=np.uint8, normalize=True):
    if isinstance(image_tensor, list):
        image_numpy = []
        for i in range(len(image_tensor)):
            image_numpy.append(tensor2im(image_tensor[i], imtype, normalize))
        return image_numpy
    image_numpy = image_tensor.cpu().float().numpy()
    if normalize:
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    else:
        image_numpy = np.transpose(image_numpy, (1, 2, 0)) * 255.0
    image_numpy = np.clip(image_numpy, 0, 255)
    if image_numpy.shape[2] == 1 or image_numpy.shape[2] > 3:
        image_numpy = image_numpy[:, :, 0]
    return image_numpy.astype(imtype)


opt = TestOptions().parse(save=False)
opt.nThreads = 1  # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip
model = create_model(opt)

# =============== test settings ================================
folder_in = "datasets/test"
in_img_size = 1024
out_img_size = 1024
save_dir = "./output/"
checkpoint = "checkpoints/best_model/latest_net_G.pth"
# =============================================================

model.netG.load_state_dict(
    torch.load(checkpoint, map_location=lambda storage, loc: storage),
    strict=True)
try:
    os.stat(save_dir)
except:
    os.makedirs(save_dir)

imgs_list = [join(folder_in, x) for x in sorted(listdir(folder_in)) if is_image_file(x)]
trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


def spiltImg(PIL_img):
    img_cv = cv2.cvtColor(np.asarray(PIL_img), cv2.COLOR_RGB2BGR)
    img_h1 = img_cv[::2, :, :]
    img_v1 = img_cv[:, ::2, :]
    img_1 = img_h1[:, ::2, :]
    img_2 = img_h1[:, 1::2, :]
    img_3 = img_v1[::2, :, :]
    img_4 = img_v1[1::2, :, :]

    PIL_img_1 = Image.fromarray(cv2.cvtColor(img_1, cv2.COLOR_BGR2RGB))
    PIL_img_2 = Image.fromarray(cv2.cvtColor(img_2, cv2.COLOR_BGR2RGB))
    PIL_img_3 = Image.fromarray(cv2.cvtColor(img_3, cv2.COLOR_BGR2RGB))
    PIL_img_4 = Image.fromarray(cv2.cvtColor(img_4, cv2.COLOR_BGR2RGB))

    return [PIL_img_1, PIL_img_2, PIL_img_3, PIL_img_4]


def imgs2tensor(PIL_imgs_4):
    tensor_imgs = torch.cat(
        [trans(PIL_imgs_4[0]).unsqueeze(0), trans(PIL_imgs_4[1]).unsqueeze(0), trans(PIL_imgs_4[2]).unsqueeze(0),
         trans(PIL_imgs_4[3]).unsqueeze(0), ], dim=0)
    return tensor_imgs


def tensorList2img(prediction_4, input_refshape):  ##resize and blur
    pred_0 = cv2.resize(prediction_4[0], (1023, 1023))
    pred_1 = cv2.resize(prediction_4[1], (1023, 1023))
    pred_2 = cv2.resize(prediction_4[2], (1023, 1023))
    pred_3 = cv2.resize(prediction_4[3], (1023, 1023))

    pred_0 = cv2.copyMakeBorder( pred_0, 0, 1, 0, 1, cv2.BORDER_REPLICATE)
    pred_1 = cv2.copyMakeBorder( pred_1, 0, 1, 1, 0, cv2.BORDER_REPLICATE)
    pred_2 = cv2.copyMakeBorder( pred_2, 0, 1, 0, 1, cv2.BORDER_REPLICATE)
    pred_3 = cv2.copyMakeBorder( pred_3, 1, 0, 1, 0, cv2.BORDER_REPLICATE)
    prediction = (1.0*pred_0+1.0*pred_1+1.0*pred_2+1.0*pred_3)/4.0
    prediction = np.clip(prediction,0,255)
    return prediction

import time
model = model.eval()
tStart = time.time()
for in_img_file in imgs_list:
    base_name = basename(in_img_file)
    print(in_img_file)
    Input_img_original = Image.open(in_img_file).convert('RGB').resize((in_img_size, in_img_size))
    Input_img_4 = spiltImg(Input_img_original)
    predictions_4 = []
    for Input_img in Input_img_4:
        Input_img = trans(Input_img).unsqueeze(0).cuda()
        # inference
        with torch.no_grad():
            prediction = model.inference(Input_img)
        prediction = tensor2im(prediction.data[0])
        predictions_4 += [prediction]

    prediction = tensorList2img(predictions_4, input_refshape=Input_img_original)
    output = Image.fromarray(np.uint8(prediction)).resize((out_img_size, out_img_size))
    output.save(join(save_dir, base_name))
print("--- %s seconds ---" % ((time.time() - tStart)/len(imgs_list)))
