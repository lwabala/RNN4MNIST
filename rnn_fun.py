import autograd.numpy as np
import autograd.scipy as sp
from autograd.numpy import array as npa
from autograd.numpy import tensordot as tdot


def trans(x, w, b, axes=False):
    dim_x, dim_w = len(x.shape), len(w.shape)
    ax_x, ax_w = [dim_x-2,dim_x-1], [0,1]
    if ~axes: axes = (ax_x, ax_w)
    x_dot_w = np.tensordot(x, w, axes=axes)
    if np.any(b): y = x_add_bias(x_dot_w, b)
    else:         y = x_dot_w
    return y

def x_add_bias(x,b):
    '''Add the bias to the input X'''
    try: batch_size = x.shape[0]
    except: print "Error@ x_add_bias(x,b): x.shape=%s" %x.shape
    if isinstance(b, (int,float)): x_with_b = x + b        
    else: x_with_b = x + b.repeat(batch_size, axis=0)
    return x_with_b

def init_acts_in_order():
    act_set=[]
    for ii in range(9): actset.append(np.zeros([1,5,5]))
    actset[0][0,1,1]=1
    actset[1][0,1,2]=1
    actset[2][0,1,3]=1
    actset[3][0,2,1]=1
    actset[4][0,2,2]=1
    actset[5][0,2,3]=1
    actset[6][0,3,1]=1
    actset[7][0,3,2]=1
    actset[8][0,3,3]=1
    return act_set

def rmsprop_grad_method(decay_rate):
    global r, dr
    r = 0.
    dr = decay_rate
    def rmsprop_grad(grad):
        global r, dr
        r *= dr
        r += (grad**2) * (1 - dr)
        rms_grad = grad / np.sqrt(r)
        return rms_grad
    return rmsprop_grad