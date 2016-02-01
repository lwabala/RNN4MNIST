import autograd.numpy as np
import autograd.scipy as sp

def sigmoid(*args, **kwargs):
    if args:
        if kwargs.has_key('scale'): 
            scale = kwargs['scale']
        else:
            scale = 1.
        y = 1 / (1 + np.exp(-args[0] * scale))
        return y
    elif kwargs.has_key('scale'):
        def sigmoid_with_scale(x):
            return 1 / (1 + np.exp(-x * kwargs['scale']))
        return sigmoid_with_scale

def tanhn(*args, **kwargs):
    if args:
        if kwargs.has_key('scale'): 
            scale = kwargs['scale']
        else:
            scale = 1.
        y = (np.tanh(args[0] * scale) + 1.0) / 2.
        return y
    elif kwargs.has_key('scale'):
        def tanhn_with_scale(x):
            return 0.5 * (np.tanh(x * kwargs['scale']) + 1.0)
        return tanhn_with_scale

def relu(x):
    return np.maximum(x, 0.)

def softmax(x, axis=1):
    expx=np.exp(x)
    if isinstance(axis, int):
        ax_sum = np.sum(expx, axis=axis, keepdims=True)
    elif isinstance(axis, (list,tuple)):
        ax_sum = expx
        for ax in axis:
            ax_sum = np.sum(ax_sum, axis=ax, keepdims=True)
    return np.divide(expx, ax_sum)

def matrix_argmax(x, value=False):
    # get the max sub from a matrix
    mm = np.argmax(x,axis=1)
    ax = x[range(len(x)),mm]
    mx = np.argmax(ax,axis=0)
    my = mm[mx]
    xy = [mx,my]
    if value: return xy, ax[mx]
    else:     return xy

def matrix_argmax_batch(matrix_batch):
    # get the max subs from a matrix batcch
    batch_size = matrix_batch.shape[0]
    argmax_batch = np.zeros([batch_size, 2])
    for order in range(batch_size):
        argmax_batch[order,:] = matrix_argmax(matrix_batch[order])
    return argmax_batch