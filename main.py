import os
import argparse
from img_process import *
from file_process import *

# nohup python main.py /157_Dataset_in150/data-wen.yaoli/affectnet/affectnet/Manually_Annotated_Images/ --root-dir /157_Dataset_in150/data-wen.yaoli/Data/affectnet >affectnet 2>&1 &
class Save_picture_dir(object):
    def __init__(self, root_dir):
        self.root_dir = root_dir
    def __call__(self, file_path, output_dir):
        Is_img = file_path.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff'))
        Is_video = file_path.lower().endswith(('.mp4', '.mkv', '.avi', '.wmv', '.iso'))
        if not Is_img:
            return
        name_body = output_dir.rsplit(".", 1)[0]
        name_end = os.path.splitext(output_dir)[-1]
        name_body = os.path.join(self.root_dir, name_body)
        if  not os.path.exists(name_body):#路径不存在
            os.makedirs(name_body)
        processes = []
        processes.append(Img_Compose([Img_read(), Img_save(save_dir=os.path.join(name_body, '0'+name_end))]))
        processes.append(Img_Compose([Img_read(), Img_crop(scale=0.75, crop_type='left_top'), Img_save(save_dir=os.path.join(name_body, '1'+name_end))]))
        processes.append(Img_Compose([Img_read(), Img_crop(scale=0.75, crop_type='right_top'), Img_save(save_dir=os.path.join(name_body, '2'+name_end))]))
        processes.append(Img_Compose([Img_read(), Img_crop(scale=0.75, crop_type='bottom'), Img_save(save_dir=os.path.join(name_body, '3'+name_end))]))
        processes.append(Img_Compose([Img_read(), Img_crop(scale=0.9, crop_type='center'), Img_save(save_dir=os.path.join(name_body, '4'+name_end))]))
        processes.append(Img_Compose([Img_read(), Img_crop(scale=0.85, crop_type='center'), Img_save(save_dir=os.path.join(name_body, '5'+name_end))]))
        for process in processes:
            process(file_path)


class Save_picture_alignment_dir(object):
    def __init__(self, root_dir):
        self.root_dir = root_dir
    def __call__(self, file_path, output_dir):
        Is_img = file_path.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff'))
        Is_video = file_path.lower().endswith(('.mp4', '.mkv', '.avi', '.wmv', '.iso'))
        if not Is_img:
            return
        name_body = output_dir.rsplit(".", 1)[0]
        name_end = os.path.splitext(output_dir)[-1]
        name_body = os.path.join(self.root_dir, name_body)
        if  not os.path.exists(name_body):#路径不存在
            os.makedirs(name_body)
        processes = []
        # processes.append(Img_Compose([Img_read(), Img_face_alignment(), Img_selector(), Img_save(save_dir=os.path.join(name_body, '0'+name_end))]))
        # processes.append(Img_Compose([Img_read(), Img_face_alignment(), Img_selector(), Img_crop(scale=0.75, crop_type='left_top'), Img_save(save_dir=os.path.join(name_body, '1'+name_end))]))
        # processes.append(Img_Compose([Img_read(), Img_face_alignment(), Img_selector(), Img_crop(scale=0.75, crop_type='right_top'), Img_save(save_dir=os.path.join(name_body, '2'+name_end))]))
        # processes.append(Img_Compose([Img_read(), Img_face_alignment(), Img_selector(), Img_crop(scale=0.75, crop_type='bottom'), Img_save(save_dir=os.path.join(name_body, '3'+name_end))]))
        # processes.append(Img_Compose([Img_read(), Img_face_alignment(), Img_selector(), Img_crop(scale=0.9, crop_type='center'), Img_save(save_dir=os.path.join(name_body, '4'+name_end))]))
        # processes.append(Img_Compose([Img_read(), Img_face_alignment(), Img_selector(), Img_crop(scale=0.85, crop_type='center'), Img_save(save_dir=os.path.join(name_body, '5'+name_end))]))
        processes.append(Img_Compose([Img_read(), Img_face_alignment()]))
        for process in processes:
            process(file_path)


class Save_Function(object):
    def __init__(self, root_dir):
        # self.save_picture_dir = Save_picture_dir(os.path.join(root_dir, "Crop"))
        self.save_picture_alignment_dir = Save_picture_alignment_dir(os.path.join(root_dir,"Alignment"))
    def __call__(self, file_path, output_dir):
        # self.save_picture_dir(file_path, output_dir)
        self.save_picture_alignment_dir(file_path, output_dir)
        # print('x')


def parse_arg():
    parser = argparse.ArgumentParser(u"Python批量转换 视频 为 音频MP3（即提取音频文件）")
    parser.add_argument(u"dir_path", help=u"输入文件、目录路径，如果为目录，则遍历目录下的文件")
    parser.add_argument(u"--root-dir", help=u"(可选)输出目录路径，如果不传，则使用输入文件目录")
    parser.add_argument(u"--traverse", action=u'store_true',
                        help=u"(可选)src-path为目录是，是否遍历子目录，默认False")

    return parser.parse_args()

if __name__ == '__main__':
    # 解析输入参数
    command_param = parse_arg()
    dir_path = command_param.dir_path
    root_dir = command_param.root_dir
    save_function = Save_Function(root_dir=root_dir)
    traverse_file(dir_path=dir_path, save_function=save_function)