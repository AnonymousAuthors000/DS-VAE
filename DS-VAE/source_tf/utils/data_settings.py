import numpy as np
import os
import json
import scipy.misc
import tensorflow_hub as hub
from disentanglement_lib.data.ground_truth import cars3d, faces3d, dsprites, norb


def imread(path, is_grayscale = False):
    if (is_grayscale):
        return scipy.misc.imread(path, flatten = True).astype(np.float)
    else:
        return scipy.misc.imread(path).astype(np.float)
    
def center_crop(x, crop_h, crop_w, resize_h=64, resize_w=64):
    if crop_w is None:
        crop_w = crop_h
    h, w = x.shape[:2]
    j = int(round((h - crop_h)/2.))
    i = int(round((w - crop_w)/2.))
    return scipy.misc.imresize(x[j:j+crop_h, i:i+crop_w], [resize_h, resize_w])

def transform(image, input_height, input_width, 
              resize_height=64, resize_width=64, is_crop=True):
    if is_crop:
        cropped_image = center_crop(image, input_height, input_width, resize_height, resize_width)
    else:
        cropped_image = scipy.misc.imresize(image, [resize_height, resize_width])
    return np.array(cropped_image)#/127.5 - 1.

def get_image(image_path, input_height, input_width, resize_height=64, resize_width=64, is_crop=True, is_grayscale=False):
    image = imread(image_path, is_grayscale)
    return transform(image, input_height, input_width, resize_height, resize_width, is_crop)


def get_dataset(dataname, dataroot, numsamples):
    '''
    inputs: x
    targets: c or y (y if using dlib)
    '''
    
    if numsamples==-1:
        inds = slice(0, None, 1)
    else:
        inds = slice(0, numsamples, 1)
    
    if dataname=='dSprites':
        # Dataroot is the directory containing the dsprites .npz file
        dataset_zip = np.load(os.path.join(dataroot, 'dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz'), encoding='bytes')
        inputs = np.expand_dims(dataset_zip['imgs'], axis=4)[inds] 
        targets = dataset_zip['latents_values'][inds]
    elif dataname=='faces':
        # Dataroot is the directory containing the images
        filenames = np.array(json.load(open(os.path.join(dataroot, 'img_store')))[inds])
        inputs = np.array([os.path.join(dataroot, '/'.join(filenames[i].split('/')[1:])) for i in range(len(filenames))])
        targets = np.load(os.path.join(dataroot, 'z_store'))[inds]
    elif dataname=='faces2':
        # Dataroot is the directory containing the images
        filenames = np.array(json.load(open(os.path.join(dataroot, 'img_store')))[inds])
        inputs = np.array([os.path.join(dataroot, '/'.join(filenames[i].split('/')[1:])) for i in range(len(filenames))])
        targets = np.load(os.path.join(dataroot, 'z_store_full'))[inds]
    elif dataname=='planes':
        # Dataroot is the directory containing the images
        filenames = np.array(json.load(open(os.path.join(dataroot, 'img_store_planes')))[inds])
        inputs = np.array([os.path.join(dataroot, 'planes', '/'.join(filenames[i].split('/')[-2:])) for i in range(len(filenames))])
        targets = np.load(os.path.join(dataroot, 'z_store_trans'))[inds]
    elif dataname=='cars':
        # Dataroot is the directory containing the images
        filenames = np.array(json.load(open(os.path.join(dataroot, 'img_store_cars')))[inds])
        inputs = np.array([os.path.join(dataroot, 'cars_for_transfer', '/'.join(filenames[i].split('/')[-2:])) for i in range(len(filenames))])
        targets = np.load(os.path.join(dataroot, 'z_store_new_cars'))[inds]
    elif dataname=='chairs':
        # Dataroot is the directory containing the images
        filenames = np.array(json.load(open(os.path.join(dataroot, 'img_store_chairs')))[inds])
        inputs = np.array([os.path.join(dataroot, 'chairs_for_transfer', '/'.join(filenames[i].split('/')[-2:])) for i in range(len(filenames))])
        targets = np.load(os.path.join(dataroot, 'z_store_new_chairs'))[inds]
    elif dataname=='dlib_cars3d':
        ## dataroot: path to the tensorflow_hub directory

        dta = cars3d.Cars3D()
        with hub.eval_function_for_module(dataroot) as f:
            # Save reconstructions.
            inputs = dta.images
            targets = f(
                dict(images=inputs), signature="reconstructions",
                as_dict=True)["images"]
    elif dataname=='dlib_smallnorb':
        ## dataroot: path to the tensorflow_hub directory

        dta = norb.SmallNORB()
        with hub.eval_function_for_module(dataroot) as f:
            # Save reconstructions.
            inputs1 = np.expand_dims(dta.images, 3)[:25000]
            targets1 = f(
                dict(images=inputs1), signature="reconstructions",
                as_dict=True)["images"]
            inputs2 = np.expand_dims(dta.images, 3)[25000:]
            targets2 = f(
                dict(images=inputs2), signature="reconstructions",
                as_dict=True)["images"]
            inputs = np.concatenate((inputs1, inputs2), axis=0)
            targets = np.concatenate((targets1, targets2), axis=0)
        print(inputs.shape)
        print(targets.shape)
    elif dataname=='dlib_faces3d':
        ## dataroot: path to the tensorflow_hub directory

        dta = faces3d.Faces3D()
        with hub.eval_function_for_module(dataroot) as f:
            # Save reconstructions.
            inputs = dta.images
            targets = f(
                dict(images=inputs), signature="reconstructions",
                as_dict=True)["images"]
        print(dataname)
        print(inputs.shape)
        print(targets.shape)
    else:
        raise NotImplementedError
    return (inputs, targets)
