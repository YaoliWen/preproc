import cv2
import os
import dlib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torchvision import transforms as transforms


predicter_path = '/home/lab-wen.yaoli/CODE/AFEW/test/model/shape_predictor_68_face_landmarks.dat'
# 导入人脸检测模型
detector = dlib.get_frontal_face_detector()
# 导入检测人脸特征点的模型
sp = dlib.shape_predictor(predicter_path)


class Img_read(object):
    def __call__(self, img_dir):
        bgr_img = cv2.imread(img_dir)
        print(img_dir)
        return bgr_img


class Img_contrast_histeqClahe(object):
    def __call__(self, bgr_img, limit=2.0, size=(8,8)):
        img_yuv = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2YUV)
        clahe = cv2.createCLAHE(clipLimit=limit, tileGridSize=size)
        img_yu = img_yuv[:,:,0]
        img_yuv[:,:,0] = clahe.apply(img_yu)
        bgr_img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
        return bgr_img


class Img_face_alignment(object):
    def __call__(self, bgr_img, max_face=True):
        # opencv的颜色空间是BGR，需要转为RGB才能用在dlib中
        rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        # 检测图片中的人脸，第二个为上采样率，第三个为阈值
        dets, scores, idx = detector.run(rgb_img, 2, -1.5)
        # 检测到的人脸数量
        num_faces = len(dets)
        # print(num_faces)
        if num_faces == 0:
            print("Sorry, there were no faces found")
            return bgr_img
        # 识别人脸特征点，并保存下来
        # Find the 5 face landmarks we need to do the alignment
        faces = dlib.full_object_detections()
        if max_face:
            #取最大面部
            s_max = 0 
            det_max = None
            for det in dets:
                s = (det.right()-det.left())*(det.bottom()-det.top())
                if s > s_max:
                    s_max = s
                    det_max = det
                    faces.append(sp(rgb_img, det_max))
        else:
            for detection in dets:
                faces.append(sp(rgb_img, detection))

        # 人脸对齐
        rgb_images = dlib.get_face_chips(rgb_img, faces, size=224, padding=0.25)
        # print(type(rgb_images[0]))
        bgr_images = [cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR) for rgb_image in rgb_images]
        return bgr_images


class Img_cv2PIL(object):
    def __call__(self, img_cv):
        img_PIL = Image.fromarray(cv2.cvtColor(img_cv,cv2.COLOR_BGR2RGB))
        return img_PIL 


class Img_PIL2cv(object):
    def __call__(self, img_PIL):
        img_cv = cv2.cvtColor(np.asarray(img_PIL),cv2.COLOR_RGB2BGR) 
        return img_cv


class Img_crop():  
    def __init__(self, scale, crop_type):
        self.scale = scale
        self.crop_type = crop_type
        self.cv2PIL = Img_cv2PIL()
        self.PIL2cv = Img_PIL2cv()
    def __call__(self, img_cv):
        if img_cv is None:
            return None
        img_PIL = self.cv2PIL(img_cv)
        ori_width, ori_height = img_PIL.size
        delta_width = ori_width*self.scale
        delta_height =  ori_height*self.scale
        mid_width = ori_width/2
        mid_height = ori_height/2
        # print(self.crop_type)
        if self.crop_type == 'right':
            crop_box = (ori_width-delta_width, mid_height-delta_height/2,
                        ori_width, mid_height+delta_height/2)
        elif self.crop_type == 'left':
            crop_box = (0, mid_height-delta_height/2,
                        0+delta_width, mid_height+delta_height/2)
        elif self.crop_type == "top":
            crop_box = (mid_width-delta_width/2, 0,
                        mid_width+delta_width/2, 0+delta_height)
        elif self.crop_type == "bottom":
            crop_box = (mid_width-delta_width/2, ori_height-delta_height,
                        mid_width+delta_width/2, ori_height)
        elif self.crop_type == 'right_top':
            crop_box = (ori_width-delta_width, 0,
                        ori_width, 0+delta_height)         
        elif self.crop_type == 'right_bottom':
            crop_box = (ori_width-delta_width, ori_height-delta_height,
                        ori_width, ori_height)
        elif self.crop_type == 'left_top':
            crop_box = (0, 0,
                    0+delta_width, 0+delta_height)
        elif self.crop_type == 'left_bottom':
            crop_box = (0, ori_height-delta_height,
                        0+delta_width, ori_height)
        elif self.crop_type == 'center':
            crop_box = (mid_width-delta_width/2, mid_height-delta_height/2,
                    mid_width+delta_width/2, mid_height+delta_height/2)
        else:
            print('error')
            return None
        img_crop_PIL = img_PIL.crop(crop_box)
        img_crop_cv = self.PIL2cv(img_crop_PIL)
        return img_crop_cv



class Img_show(object):   
    def __call__(self, bgr_images):
        if isinstance(bgr_images,list):
            for bgr_image in bgr_images:
                #image_cnt += 1
                cv_bgr_image = np.array(bgr_image).astype(np.uint8)# 先转换为numpy数组
                rgb_img = cv2.cvtColor(cv_bgr_image, cv2.COLOR_BGR2RGB)
                plt.imshow(rgb_img)
        else:
            cv_bgr_image = np.array(bgr_images).astype(np.uint8)# 先转换为numpy数组
            rgb_img = cv2.cvtColor(cv_bgr_image, cv2.COLOR_BGR2RGB)
            plt.imshow(rgb_img)


class Img_save(object):
    def __init__(self, save_dir):
        self.save_dir = save_dir
    def __call__(self, img):
        return
        if img is None:
            print(self.save_dir)
        else:
            # print(self.save_dir)
            cv2.imwrite(self.save_dir, img)
        return img


class Img_selector(object):
    def __call__(self, images, i=0):
        if images is None:
            img = None
        else:
            img = images[0]
        return img


class Img_Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms
    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string

