import numpy as np
from numpy import array as npa
import struct
import matplotlib.pyplot as plt

#file name
name_train_LAB = r'MNIST\train-LABels.idx1-ubyte'
name_train_IMG = r'MNIST\train-images.idx3-ubyte'
name_test_LAB= r'MNIST\t10k-labels.idx1-ubyte'
name_test_IMG= r'MNIST\t10k-images.idx3-ubyte'

trLABfile = open(name_train_LAB , 'rb')
trIMGfile = open(name_train_IMG , 'rb')
teLABfile = open(name_test_LAB , 'rb')
teIMGfile = open(name_test_IMG , 'rb')

trLABstream = trLABfile.read()
trIMGstream = trIMGfile.read()
teLABstream = teLABfile.read()
teIMGstream = teIMGfile.read()

LAB_titlesize=struct.calcsize('>II')
IMG_titlesize=struct.calcsize('>IIII')
LAB_subsize=struct.calcsize('>1B')
IMG_subsize=struct.calcsize('>784B')

# transform a numeral label to one-hot code
def lab2target(lab):
    tar=np.zeros([1,10],dtype=np.float32)
    tar[0,lab]=1.
    return tar

# transform a whole img to a set of patches 
def img_subset(img):
    localimg=img
    # 3*3 patches in total , each patch is 14pix*14pix
    ((rows,cols),subnum,subsize)=(localimg.shape,npa([3,3]),npa([14,14]))
    IMG=[]
    subdist=([rows,cols]-subsize)/(subnum-1)
    for srow in range(subnum[0]):
        for scol in range(subnum[1]):
            (row,col)=subdist*[srow,scol]
            sub_img=localimg[row:row+subsize[0],col:col+subsize[1]]
            IMG.append(sub_img.reshape(1,-1))
    return npa(IMG)

def readMNIST(index,ifshow=0,file='train'):
    if file == 'train':
        LABstream = trLABstream
        IMGstream = trIMGstream
    elif file == 'test':
        LABstream = teLABstream
        IMGstream = teIMGstream
    
    index_LAB = LAB_titlesize + LAB_subsize * index
    index_IMG = IMG_titlesize + IMG_subsize * index

    lab = struct.unpack_from('>1B',LABstream,index_LAB)[0]
    img = struct.unpack_from('>784B',IMGstream,index_IMG)
    # norm img data to 0~1
    img = np.asarray(img,dtype=np.float32).reshape(28,28)/255

    if ifshow:
        fig = plt.figure()
        plotwindow = fig.add_subplot(111)
        plt.imshow( img , cmap='gray' )
        plt.show()
    return lab , img

# using in RNN
def readMNIST_all(file='train'):
    if file == 'train':
        # the image number will be taken
        imgnum = 10240
    elif file == 'test':
        imgnum = 5120

    lab,img = readMNIST(0,ifshow=0,file=file)
    LAB,IMG = lab2target(lab),img_subset(img)
    
    for imgiter in xrange(1,imgnum):
        lab , img = readMNIST( imgiter , ifshow=0 , file=file )
        LAB = np.concatenate( ( LAB,lab2target(lab) ) , axis=0 )
        IMG = np.concatenate( ( IMG,img_subset(img) ) , axis=1 )
    return IMG , LAB , imgnum
