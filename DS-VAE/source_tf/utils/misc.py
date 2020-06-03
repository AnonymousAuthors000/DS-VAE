import numpy as np
from fid import calculate_fid_given_paths

def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    if (images.shape[3] in (3,4)):
        c = images.shape[3]
        img = np.zeros((h * size[0], w * size[1], c))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w, :] = image
        return img
    elif images.shape[3]==1:
        img = np.zeros((h * size[0], w * size[1]))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w] = image[:,:,0]
        return img
    else:
        raise ValueError('in merge(images,size) images parameter '
                     'must have dimensions: HxW or HxWx3 or HxWx4')

def get_cars_factors():
    factors = np.zeros((17568,3))
    for i in range(factors.shape[0]):
        first, remainder = divmod(i, 24*183)
        second, third = divmod(remainder, 183)
        factors[i] = np.array([first,second,third])
    return factors
    
def get_norb_factors():
    factors = np.zeros((dta.images.shape[0],4))
    for i_1 in range(5):
    for i_2 in range(10):
        for i_3 in range(9):
            for i_4 in range(18):
                for i_5 in range(6):
                    all_factor = np.array([i_1,i_2,i_3,i_4,i_5])
                    idx = dta.index.features_to_index(all_factor)
                    factors[idx] = all_factor[np.array([0,2,3,4])]
    return factors


def sigmoid(x):
    z = 1/(1 + np.exp(-x)) 
    return z

def remove_files(folder):
    filelist = os.listdir(folder)
    for the_file in filelist:
        file_path = os.path.join(folder, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(e)
            
            

def get_fid_from_array(arrays, convert2image_path, originaldata_path, path2inceptionnet):
    fid_value_list = []
    for arr in arrays:
        i = 0
        # convert array
        if args.dataset == 'norb':
            new_arr = np.zeros((arr.shape[0],arr.shape[1],arr.shape[2],3))
            for j in range(arr.shape[0]):
                new_arr[j] = cv2.cvtColor((arr[j]*255).astype(np.uint8),cv2.COLOR_GRAY2RGB)
            assert new_arr.shape == (arr.shape[0],arr.shape[1],arr.shape[2],3)
            arr = new_arr
        
        # clear converted image folder
        remove_files(convert2image_path)
        for img in arr:
            ims(convert2image_path+str(i)+".jpg",img)
            i+=1
        fid_value = calculate_fid_given_paths([convert2image_path,originaldata_path],path2inceptionnet)
        remove_files(convert2image_path)
        fid_value_list.append(fid_value)
    return fid_value_list
