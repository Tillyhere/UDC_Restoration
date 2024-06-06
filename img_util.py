import os
import glob
from PIL import Image
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms

def crop_imgs(source_path, save_path, hstep, wstep):
    imgs = glob.glob(source_path)
    for x in imgs:
        name,ext = os.path.splitext(x.split("\\")[-1])
        im = Image.open(x)
        w,h = im.size
        h_space = np.arange(0, h - hstep + 1, hstep)
        w_space = np.arange(0, w - wstep + 1, wstep)
        index = 0
        for x in w_space:
            for y in h_space:
                index +=1
                region = (x, y, x+wstep,y+hstep)
                print(region)
                cropImg = im.crop(region)
                cropImg.save(os.path.join(save_path,name+"_"+str(index)+ext))

def get_lr(source_path,save_path,scale):
    imgs = glob.glob(source_path)
    for im in imgs:
        name, ext = os.path.splitext(im.split("\\")[-1])
        im = Image.open(im)
        img = im.resize((im.size[0] // scale, im.size[1] // scale), Image.BICUBIC)
        img.save(os.path.join(save_path,name+"_lr"+ext))




def crop_five(source_path,save_path,h,w):
    size = (h,w)
    trans = transforms.FiveCrop(size)
    # retrans = transforms.ToPILImage()
    imgs = glob.glob(source_path)
    for im in imgs:
        name, ext = os.path.splitext(im.split("\\")[-1])
        im = Image.open(im)
        imgtensor = trans(im)
        print(imgtensor)
        imgtensor[0].save(os.path.join(save_path,name+"_1"+ext))
        imgtensor[1].save(os.path.join(save_path, name + "_2" + ext))
        imgtensor[2].save(os.path.join(save_path, name + "_3" + ext))
        imgtensor[3].save(os.path.join(save_path, name + "_4" + ext))
        imgtensor[4].save(os.path.join(save_path, name + "_5" + ext))

def random_crop(source_gt_path,source_in_path,save_gt_path,save_in_path,patch_size,per_num):

    # retrans = transforms.ToPILImage()
    gt_imgs = sorted(glob.glob(source_gt_path))
    in_imgs = sorted(glob.glob(source_in_path))
    index = 0
    for index in range(len(gt_imgs)):
        name, ext = os.path.splitext(gt_imgs[index].split("\\")[-1])
        gt_im = Image.open(gt_imgs[index])
        in_im = Image.open(in_imgs[index])
        w,h = gt_im.size
        for k in range(per_num):
            i = torch.randint(0, h - patch_size + 1, size=(1,)).item()
            j = torch.randint(0, w - patch_size + 1, size=(1,)).item()
            region = (j,i,j+patch_size,i+patch_size)
            gt_cropImg = gt_im.crop(region)
            gt_cropImg.save(os.path.join(save_gt_path, name + "_" + str(k) + ext))
            in_cropImg = in_im.crop(region)
            in_cropImg.save(os.path.join(save_in_path, name + "_" + str(k) + ext))

def check_file():
    dir_a = '/home/ubuntu/tyhere/OLED_challenge/Poled_train/LQ/*.png'
    dir_b = '/home/ubuntu/tyhere/OLED_challenge/Poled_train/HQ/*.png'
    fs_a = glob.glob(dir_a)
    fs_b = glob.glob((dir_b))
    if(fs_a==fs_a):
        print("ok")
    else:
        print("error")

source_path = r'E:\CodeHome\dataset\OLED_challenge\Poled_val\LQ\*.png'
# save_path = r'E:\CodeHome\dataset\OLED_challenge\cropped\POLED_train\LQ'
save_path = r'E:\CodeHome\dataset\OLED_challenge\cropped\cropped_5\POLED_val\LQ'
# crop_imgs(source_path,save_path,512,1024)
# crop_five(source_path,save_path,512,1024)
source_gt_path =r'E:\CodeHome\dataset\OLED_challenge\Toled_train\HQ\*.png'
source_in_path =r'E:\CodeHome\dataset\OLED_challenge\Toled_train\LQ\*.png'
save_gt_path = r'E:\CodeHome\dataset\OLED_challenge\cropped\random_cropped\Toled_train\HQ'
save_in_path = r'E:\CodeHome\dataset\OLED_challenge\cropped\random_cropped\Toled_train\LQ'
# random_crop(source_gt_path,source_in_path,save_gt_path,save_in_path,512,16)
check_file()
