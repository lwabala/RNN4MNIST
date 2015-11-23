import autograd.numpy as np

def lab2target(lab):
    tar=np.zeros(10,dtype=np.float32)
    tar[lab]=1.
    return tar

def output2lab(output):
    outlist=list(output.reshape(-1))
    return outlist.index(max(outlist))


def img_subset(img):
    localimg=img
    ((rows,cols),subnum,subsize)=(localimg.shape,[3,3],[14,14])
    IMG=[]
    subdist=([rows,cols]-subsize)/(subnum-1)
    for srow in range(subnum[0]):
        for scol in range(subnum[1]):
            (row,col)=subdist*[srow,scol]
            sub_img=localimg[row:row+subsize[0],col:col+subsize[1]]
            #IMG[srow].append(sub_img)
            IMG.append(sub_img.reshape(1,-1))
    return IMG