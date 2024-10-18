from PIL import Image
import random
import os
import glob
from numpy.random import default_rng
import numpy as np
import cv2
from matplotlib import pyplot as plt
from tqdm import tqdm
from skimage.transform import radon

theta = np.linspace(0., 180., 224, endpoint=False)


def multi_hot_encoding(labels):
    one_hot_labels = np.zeros((labels.shape[0], 12))
    for i, label in enumerate(labels):
        one_hot_labels[i, label] = 1
    return one_hot_labels

def spatial_filtering(binary_image):
    result_image = np.copy(binary_image)
    rows, cols = binary_image.shape
    
    # directions = [(2, 0), (-2, 0), (0, 2), (0, -2), (2, 2), (-2, -2), (2, -2), (-2, 2)]
    directions = [(2, 0), (-2, 0), (0, 2), (0, -2), (2, 2), (-2, -2), (2, -2), (-2, 2),
                  (2, 1), (2, -1), (-2, 1), (-2, -1), (1, 2), (-1, 2), (1, -2), (-1, -2)]
    
    for i in range(rows):
        for j in range(cols):
            all_zero_neighbors = True
            if binary_image[i, j] != 0:
                for dx, dy in directions:
                    x, y = i + dx, j + dy
                    if 0 <= x < rows and 0 <= y < cols:
                        if binary_image[x, y] == 1:
                            all_zero_neighbors = False                    
                            break
                if all_zero_neighbors:
                    result_image[i, j] = 0
    return result_image

# cl = ["102", "110", "115", "120", "130", "138", "139", "140", "155", "156"]
cl = ["102", "110", "115", "120", "130", "138", "139", "140", "155", "156"] 
train_mode = False

label = []
image_list = []

for i, wafer_class in enumerate(cl):
    
    path = "/home/cocel/sh/samsung/1106_new_data/train/{}/".format(wafer_class)
    for image_path in glob.glob(path + "*.png"):
        image_list.append(image_path)
        label.append(i+1)

image_list = np.array(image_list)
label = np.array(label)

data2 = []
new_label2 = []

# ### 2-Mixed Data

# for i in tqdm(range(10000)):
#     path2 = "/home/cocel/sh/samsung/1106_new_data/new_multi_label/train2/"
#     os.makedirs(path2, exist_ok=True)
    
#     while True:
#         random_integers = np.random.choice(np.arange(0, len(label)), size=2, replace=False)  
#         if label[random_integers][0] != label[random_integers][1]:
#             break
#     if train_mode == True:
#         with Image.open(image_list[random_integers][0]).convert('L') as im:
#             im1 = spatial_filtering(np.array(im.rotate(np.random.choice(np.arange(0, 359), size=1, replace=False), fillcolor='black')) / 255.)
#         with Image.open(image_list[random_integers][1]).convert('L') as im:
#             im2 = spatial_filtering(np.array(im.rotate(np.random.choice(np.arange(0, 359), size=1, replace=False), fillcolor='black')) / 255.)
#     else:
#         with Image.open(image_list[random_integers][0]).convert('L') as im:
#             im1 = spatial_filtering(np.array(im) / 255.)
#         with Image.open(image_list[random_integers][1]).convert('L') as im:
#             im2 = spatial_filtering(np.array(im) / 255.)            
#     # im1 = spatial_filtering(cv2.imread(image_list[random_integers][0], 0) / 255.)
#     # im2 = spatial_filtering(cv2.imread(image_list[random_integers][1], 0) / 255.)
#     im3 = (cv2.bitwise_or(im1, im2, mask = None) * 255.).astype(np.uint8)
    
#     save_name = path2 + str(i) + '.png'
    
#     cv2.imwrite(save_name, im3)
#     data2.append(im3)
#     new_label2.append([label[random_integers][0], label[random_integers][1]])  
                          
# data2 = np.array(data2)
# new_label2 = multi_hot_encoding(np.array(new_label2))

data3 = []
new_label3 = []

### 3-Mixed Data    
for i in tqdm(range(10000)):
    path3 = "/home/cocel/sh/samsung/1106_new_data/new_multi_label/train3/"
    
    os.makedirs(path3, exist_ok=True)
    
    while True:
        random_integers = np.random.choice(np.arange(0, len(label)), size=3, replace=False)  
        if label[random_integers][0] != label[random_integers][1] and label[random_integers][0] != label[random_integers][2] and label[random_integers][1] != label[random_integers][2]:
            break
    if train_mode == True:
        with Image.open(image_list[random_integers][0]).convert('L') as im:                              
            im1 = spatial_filtering(np.array(im.rotate(np.random.choice(np.arange(0, 359), size=1, replace=False), fillcolor='black')) / 255.)
        with Image.open(image_list[random_integers][1]).convert('L') as im:
            im2 = spatial_filtering(np.array(im.rotate(np.random.choice(np.arange(0, 359), size=1, replace=False), fillcolor='black')) / 255.)
        with Image.open(image_list[random_integers][2]).convert('L') as im:
            im3 = spatial_filtering(np.array(im.rotate(np.random.choice(np.arange(0, 359), size=1, replace=False), fillcolor='black')) / 255.)    
    else:
        with Image.open(image_list[random_integers][0]).convert('L') as im:
            o_im1 = np.array(im)
            r_im1 = radon(o_im1/255., theta=theta).astype(np.uint8)
            s_im1 = (spatial_filtering(np.array(im) / 255.)*255.).astype(np.uint8)
            rs_im1 = radon(s_im1/255., theta=theta).astype(np.uint8)
            im1 = spatial_filtering(np.array(im) / 255.)
        with Image.open(image_list[random_integers][1]).convert('L') as im:
            o_im2 = np.array(im)
            r_im2 = radon(o_im2/255., theta=theta).astype(np.uint8)
            s_im2 = (spatial_filtering(np.array(im) / 255.)*255.).astype(np.uint8)
            rs_im2 = radon(s_im2/255., theta=theta).astype(np.uint8)
            im2 = spatial_filtering(np.array(im) / 255.)
        with Image.open(image_list[random_integers][2]).convert('L') as im:
            o_im3 = np.array(im)
            r_im3 = radon(o_im3/255., theta=theta).astype(np.uint8)
            s_im3 = (spatial_filtering(np.array(im) / 255.)*255.).astype(np.uint8)
            rs_im3 = radon(s_im3/255., theta=theta).astype(np.uint8)
            im3 = spatial_filtering(np.array(im) / 255.)           
                  
    # im1 = spatial_filtering(cv2.imread(image_list[random_integers][0], 0) / 255.)
    # im2 = spatial_filtering(cv2.imread(image_list[random_integers][1], 0) / 255.)
    # im3 = spatial_filtering(cv2.imread(image_list[random_integers][2], 0) / 255.)
    im4 = spatial_filtering(cv2.bitwise_or(im1, im2, mask = None))
    im5 = (cv2.bitwise_or(im4, im3, mask = None) * 255.).astype(np.uint8)


    plt.subplot(3, 3, 1)
    plt.imshow(o_im1)
    plt.axis('off')
    
    plt.subplot(3, 3, 4)
    plt.imshow(o_im2)
    plt.axis('off')
    
    plt.subplot(3, 3, 7)
    plt.imshow(o_im3)
    plt.axis('off')
    
    plt.subplot(3, 3, 2)
    plt.imshow(s_im1)
    plt.axis('off')
    
    plt.subplot(3, 3, 5)
    plt.imshow(s_im2)
    plt.axis('off')
    
    plt.subplot(3, 3, 8)
    plt.imshow(s_im3)
    plt.axis('off')
 
    plt.subplot(3, 3, 6)
    plt.imshow((im4*255.).astype(np.uint8))
    plt.axis('off')

    plt.subplot(3, 3, 9)
    plt.imshow(im5)
    plt.axis('off')
           
    plt.subplot(3, 3, 3)
    plt.imshow((cv2.bitwise_or(im1, im2, mask = None)*255.).astype(np.uint8))
    plt.axis('off')
    
    plt.subplots_adjust(
        wspace=0.1, 
        hspace=0.1)
    plt.axis('off')

    plt.savefig('6.png', bbox_inches='tight')
    print([label[random_integers][0], label[random_integers][1], label[random_integers][2]])
    plt.show()
    
    raise

    
    train_save_name = path3 + str(i) + '.png'
    

    cv2.imwrite(train_save_name, im5)
    data3.append(im5)
    new_label3.append([label[random_integers][0], label[random_integers][1], label[random_integers][2]]) 
          
                          
data3 = np.array(data3)
new_label3 = multi_hot_encoding(np.array(new_label3))

data = np.concatenate((data2, data3), 0)
new_label = np.concatenate((new_label2, new_label3), 0)
np.save("/home/cocel/sh/samsung/1106_new_data/new_multi_label/train_data.npy", data)
np.save("/home/cocel/sh/samsung/1106_new_data/new_multi_label/train_label.npy", new_label)
