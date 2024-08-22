import os
from glob import glob
from shutil import copy

data_path = "flood_dataset/images"
anno_path = "flood_dataset/annotations"
new_path = "5cls_flood_dataset"
img_files = sorted(os.listdir(data_path))
for sub_folder in img_files:
    img_list = glob(os.path.join(data_path, sub_folder, '*.jpg'))
    val_list = img_list[-int(len(img_list)*0.2):]
    train_list = img_list[:-int(len(img_list)*0.2)]
    for img_path in train_list:
        img_name = img_path.split('/')[-1]
        anno_img_name = img_name.replace('jpg','png')
        save_path = os.path.join(new_path, 'img_dir', 'train')
        copy(img_path, save_path)
        anno_img_path = os.path.join(anno_path,sub_folder,anno_img_name)
        save_anno_path = os.path.join(new_path, 'ann_dir', 'train')
        copy(anno_img_path,save_anno_path)
        
    for img_path in val_list:
        img_name = img_path.split('/')[-1]
        anno_img_name = img_name.replace('jpg','png')
        save_path = os.path.join(new_path, 'img_dir', 'val')
        copy(img_path, save_path)
        anno_img_path = os.path.join(anno_path,sub_folder,anno_img_name)
        save_anno_path = os.path.join(new_path, 'ann_dir', 'val')
        copy(anno_img_path,save_anno_path)
