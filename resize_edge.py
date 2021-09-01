import os
import pandas as pd
import numpy as np
from PIL import Image, ImageOps
import cv2

DESIRED_SIZE = 224
MARGIN = 100
MODEL_VIEWS_DIR = 'dataset_3d_model_view_white'
data_dir = os.path.join('.', 'data', 'blender-off-multiview-tool')
sk_resize_dir = 'SHREC14LSSTB_SKETCHES_RESIZED'
cad_resize_dir = os.path.join('SHREC14', 'SHREC14LSSTB_TARGET_MODELS_RESIZED')
cad_edge_dir = os.path.join('SHREC14', 'SHREC14LSSTB_TARGET_MODELS_EDGE')


def transform_sk(im):
    if im.mode == 'RGBA':
        im = im.convert('RGB')
    # [width, height]
    old_size = np.asarray(im.size)
    ratio = float(DESIRED_SIZE) / max(old_size)
    new_size = map(int, old_size * ratio)
    # Image.NEAREST：Nearest neighbors(最近邻)
    # Image.BILINEAR：Bilinear difference(双线性插值)
    # Image.BICUBIC：Bicubic interpolation(双三次插值)
    # Image.ANTIALIAS：Area interpolation(Lanczos kernel)
    im = im.resize(new_size, Image.ANTIALIAS)
    return im


def transform_cad(im, margin=0):
    if im.mode == 'RGBA':
        im = im.convert('RGB')

    # get roi coordinates
    mask = np.asarray(im) != 0

    min_w = np.min(np.argwhere(mask)[:, 1]) - margin
    max_w = np.max(np.argwhere(mask)[:, 1]) + margin

    min_h = np.min(np.argwhere(mask)[:, 0]) - margin
    max_h = np.max(np.argwhere(mask)[:, 0]) + margin

    bbox = [min_w, min_h, max_w, max_h]

    # cropping
    im = im.crop(bbox)

    # resize
    old_size = np.asarray(im.size)
    ratio = float(DESIRED_SIZE) / max(old_size)
    new_size = list(map(int, old_size * ratio))
    im = im.resize(new_size, Image.ANTIALIAS)

    # padding
    delta_w = DESIRED_SIZE - new_size[0]
    delta_h = DESIRED_SIZE - new_size[1]
    padding = (delta_w // 2, delta_h // 2, delta_w - (delta_w // 2),
               delta_h - (delta_h // 2))
    black = (0, 0, 0)
    im = ImageOps.expand(im, padding, black)

    return im


def process_im(path, mode):
    im = Image.open(path)
    # saving
    im_dir = os.path.dirname(path)
    fname = os.path.basename(path)
    split = im_dir.split(os.path.sep)[-1]
    clsname = im_dir.split(os.path.sep)[-2]

    save_dir = ''
    if mode == 'sketch':
        save_dir = os.path.join(*im_dir.split(os.path.sep)[:-4])
        save_dir = os.path.join(save_dir, sk_resize_dir, clsname, split)
        im = transform_sk(im)
    else:
        save_dir = os.path.join(data_dir, cad_resize_dir, clsname, split)
        # im = transform_cad(im, MARGIN)
        im = transform_sk(im)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # The quality of the saved image, with values ranging from 1 (worst) to 95 (best)
    im.save(os.path.join(save_dir, fname), quality=95)


def getEdge(path):
    im = cv2.imread(path)
    # grayscale
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    # Gaussian smoothing,
    # the gaussian convolution kernel size: 3 * 3
    # standard deviation: 0
    im = cv2.GaussianBlur(im, (3, 3), 0)
    canny = cv2.Canny(im, 50, 150)
    canny = cv2.bitwise_not(canny)

    save_edge = path.replace(cad_resize_dir, cad_edge_dir)
    im_dir = os.path.dirname(save_edge)
    if not os.path.exists(im_dir):
        os.makedirs(im_dir)
    # [int(cv2.IMWRITE_PNG_COMPRESSION), 0]：High quality compression
    cv2.imwrite(save_edge, canny, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])


if __name__ == '__main__':
    df = pd.read_hdf(
        os.path.join('labels', 'REORGANIZE-PART-SHREC14', 'sk_orig.hdf5'))
    paths = df.index.values
    sk_paths = []
    for p in paths:
        sk_paths.append(os.path.join(data_dir, p))
    sk_cat = df['cat'].values
    sk_split = df['split'].values
    sk_idx = df['id'].values
    print('paths:', paths)
    print('sk_cat:', sk_cat)
    print('sk_split:', sk_split)
    print('sk_idx:', sk_idx)
    print('total number of sketches: ', len(sk_paths))

    cad_im_path = os.path.join(data_dir, MODEL_VIEWS_DIR)
    cad_class = os.listdir(cad_im_path)
    cad_paths = []
    for c in cad_class:
        class_path = os.path.join(cad_im_path, c)
        class_item = os.listdir(class_path)
        for item in class_item:
            item_path = os.path.join(class_path, item)
            views = os.listdir(item_path)
            for v in views:
                cad_paths.append(os.path.join(item_path, v))
    print('total number of 3D models\' views: ', len(cad_paths))

    # print('----- Start sketches resizing -----')
    # for i in range(len(sk_paths)):
    #     process_im(sk_paths[i], 'sketch')
    #     if i and i % 500 == 0:
    #         print('done:', i)
    # print('total:', len(sk_paths))
    # print('----- sketches resizing all be done -----')

    print('----- start 3D models\' views resizing -----')
    edge_paths = []
    for i in range(len(cad_paths)):
        # process_im(cad_paths[i], 'cad')
        edge_paths.append(cad_paths[i].replace(MODEL_VIEWS_DIR,
                                               cad_resize_dir))
        if i and i % 5000 == 0:
            print('done:', i)
    print('total:', len(cad_paths))
    print('----- resizing 3D model views all be done -----')

    print('----- start edges extracting -----')
    for i in range(len(edge_paths)):
        getEdge(edge_paths[i])
        if i and i % 5000 == 0:
            print('done:', i)
    print('total:', len(edge_paths))
    print('----- edges extracting all be done -----')