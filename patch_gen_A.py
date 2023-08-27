
import os
import numpy as np
import skimage.io
import scipy.io
# len(os.listdir('part_A/test_data/images/'))


def crop_gen(img, points):
    rn = np.random.choice(4, size=1, replace=True, p=None)[0]
    tar = np.zeros([img.shape[0], img.shape[1]])
    for co in range(len(points)):
        if (np.rint(points[co,1]).astype(int) < img.shape[0]) and (np.rint(points[co,0]).astype(int) < img.shape[1]):
            tar[  np.rint(points[co,1]).astype(int), np.rint(points[co,0]).astype(int)] = 1
        else: 
            tar[  points[co,1].astype(int), points[co,0].astype(int)] = 1

    if rn == 0:
        temp_img = img[:img.shape[0]//128 * 128, :img.shape[1]//128 * 128]
        temp_tar = tar[:img.shape[0]//128 * 128, :img.shape[1]//128 * 128]
    elif rn == 1:
        temp_img = img[-(img.shape[0]//128 * 128):, :img.shape[1]//128 * 128]
        temp_tar = tar[-(img.shape[0]//128 * 128):, :img.shape[1]//128 * 128]
    elif rn == 2:
        temp_img = img[:img.shape[0]//128 * 128, -(img.shape[1]//128 * 128):]
        temp_tar = tar[:img.shape[0]//128 * 128, -(img.shape[1]//128 * 128):]
    elif rn == 3:
        temp_img = img[-(img.shape[0]//128 * 128):, -(img.shape[1]//128 * 128):]
        temp_tar = tar[-(img.shape[0]//128 * 128):, -(img.shape[1]//128 * 128):]


    crop_tar = []
    crop_list = []
    for x in range(temp_img.shape[0]//128):
        for y in range(temp_img.shape[1]//128):
            crop_list.append(temp_img[x*128:(x*128+128), y*128:(y*128+128)])
            crop_tar.append( temp_tar[x*128:(x*128+128), y*128:(y*128+128)] )

    # for x in range(img.shape[0]//128):
    #     crop_list.append( img[x*128:(x*128+128), (img.shape[1] - 128):img.shape[1]] )
    # crop_list.append( img[(img.shape[0] - 128):img.shape[0], (img.shape[1] - 128):img.shape[1]] )

    # for y in range(img.shape[1]//128):
    #     crop_list.append( img[ (img.shape[0] - 128):img.shape[0], y*128:(y*128+128)] )
    return crop_list, crop_tar
# crop_list.append( img[(img.shape[0] - 128):img.shape[0], (img.shape[1] - 128):img.shape[1]] )

imglist = os.listdir('part_A/train_data/images/')
shape_list = []
for i in range(len(imglist)):
    img = skimage.io.imread('part_A/train_data/images/' + imglist[i])
    mat = scipy.io.loadmat('part_A/train_data/ground-truth/' + 'GT_' + imglist[i].split('.jpg')[0] + '.mat')
    ppp = mat['image_info'][0][0][0][0][0]
    c_list = crop_gen(img, ppp)

    for nk in range(len(c_list[0])):
        skimage.io.imsave( 'part_A/uncertain_data/'  + imglist[i].split('.jpg')[0] + '_' + str(nk) + '.png',   c_list[0][nk] )
        skimage.io.imsave( 'part_A/uncertain_label/'  + imglist[i].split('.jpg')[0] + '_' + str(nk) + '_label.png', c_list[1][nk] )