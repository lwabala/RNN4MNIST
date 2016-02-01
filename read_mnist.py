import numpy as np
import struct
import matplotlib.pyplot as plt

#file name
name_tr_lab = r'MNIST\train-LABels.idx1-ubyte'
name_tr_img = r'MNIST\train-images.idx3-ubyte'
name_te_lab = r'MNIST\t10k-labels.idx1-ubyte'
name_te_img = r'MNIST\t10k-images.idx3-ubyte'

file_tr_lab = open(name_tr_lab, 'rb')
file_tr_img = open(name_tr_img, 'rb')
file_te_lab = open(name_te_lab, 'rb')
file_te_img = open(name_te_img, 'rb')

stream_tr_lab = file_tr_lab.read()
stream_tr_img = file_tr_img.read()
stream_te_lab = file_te_lab.read()
stream_te_img = file_te_img.read()

lab_titlesize = struct.calcsize('>II')
img_titlesize = struct.calcsize('>IIII')
lab_subsize = struct.calcsize('>1B')
img_subsize = struct.calcsize('>784B')

def lab2target(lab):
    '''Transform a numeral label to one-hot code.'''
    tar = np.zeros([1,1,10], dtype=np.float32)
    tar[0,0,lab] = 1.
    return tar

def img_subset(img):
    '''Transform a whole img to a set of patches.'''
    local_img = img
    img_set = []
    # 3*3 patches in total , each patch is 14pix*14pix
    subnum = np.array([3,3])
    subsize = np.array([14,14])
    subdist = (local_img.shape - subsize) / (subnum - 1)
    for srow in range(subnum[0]):
        for scol in range(subnum[1]):
            row, col = subdist * (srow, scol)
            sub_img = local_img[row : row + subsize[0], col : col + subsize[1]]
            img_set.append(sub_img.reshape(1,-1))
    return np.array(img_set)

def read_one_mnist(index, ifshow=False, file='train'):
    '''Read an image and its label from MNIST with its order
        
        index = the order of the image you want to unpack
        ifshow = True or False , if show the image
        file = 'train' , for train data 
            or 'test'  , for test data.'''
    if file == 'train': LABstream, IMGstream = stream_tr_lab, stream_tr_img
    elif file == 'test': LABstream, IMGstream = stream_te_lab, stream_te_img
    index_LAB = lab_titlesize + lab_subsize * index
    index_IMG = img_titlesize + img_subsize * index
    lab = struct.unpack_from('>1B', LABstream, index_LAB)[0]
    img = struct.unpack_from('>784B', IMGstream, index_IMG)
    # norm img data to [0~1] & reshape
    img = np.asarray(img,dtype=np.float32).reshape(28,28)/255
    if ifshow:
        fig = plt.figure()
        plotwindow = fig.add_subplot(111)
        plt.imshow(img, cmap='gray')
        plt.show()
    return lab , img


def read_all_mnist(img_num, file='train',mod='local'):
    '''Using in train_rnn, file='train' for train data or 'test' for test data.'''
    labs, imgs = [], []
    if mod == 'local':
        for imgiter in xrange(img_num):
            lab, img = read_one_mnist(imgiter, ifshow=0, file=file)
            labs.append(lab2target(lab))
            imgs.append(img_subset(img))
        lab_set = np.concatenate(labs, axis=0)
        img_set = np.concatenate(imgs, axis=1)
    elif mod == 'full':
        for imgiter in xrange(img_num):
            lab, img = read_one_mnist(imgiter, ifshow=0, file=file)
            labs.append(lab2target(lab))
            imgs.append([img])
        lab_set = np.concatenate(labs, axis=0)
        img_set = np.concatenate(imgs, axis=0)
    return lab_set, img_set, img_num