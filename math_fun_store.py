import autograd.numpy as np
EPS=1e-5

def sigmoid(*args, **kwargs):
    if args:
        if kwargs.has_key('scale'): scale = kwargs['scale']
        else:                       scale = 1.
        return            1 / (1 + np.exp(-args[0] * scale))
    elif kwargs.has_key('scale'):
        return lambda x : 1 / (1 + np.exp(-x * kwargs['scale']))

def tanhn(x):
    return 0.5*(np.tanh(x) + 1.0)   # Output ranges from 0 to 1.

def relu(x):
    return np.maximum(x,0.)

def softmax(x, axis=1):
    expx=np.exp(x)
    if isinstance(axis, int):
        ax_sum = np.sum(expx, axis=axis, keepdims=True)
    elif isinstance(axis, (list,tuple)):
        ax_sum = expx
        for ax in axis:
            ax_sum = np.sum(ax_sum, axis=ax, keepdims=True)
    return np.divide(expx, ax_sum)

def log_loss(pred, targ):
    ax = len(targ.shape)
    batch_size = targ.shape[0]
    likelihood = np.multiply(targ, pred) + np.multiply(1.-targ, 1.-pred)
    likelihood_norm = likelihood + EPS
    log_likelihood = np.log(likelihood_norm)
    lost = -np.sum(log_likelihood, axis=ax-1).reshape([batch_size])
    return lost

def matrix_argmax(x, value=False):
    mm = np.argmax(x,axis=1)
    ax = x[range(len(x)),mm]
    mx = np.argmax(ax,axis=0)
    my = mm[mx]
    xy = [mx,my]
    if value:
        v = ax[mx]
        return xy, v
    else:
        return xy

def matrix_argmax_batch(x_batch):
    batch_size = x_batch.shape[0]
    y_batch = np.zeros([batch_size, 2])
    for order in range(batch_size):
        y_batch[order,:] = matrix_argmax(x_batch[order])
    return y_batch

def dot(x, w):
    if x.shape[-2:] == w.shape[:2]:
        ndim = len(x.shape)
        return np.tensordot(x,w,axes=[[ndim-2,ndim-1],[0,1]])
    elif w.shape[-2:] == x.shape[:2]:
        ndim = len(w.shape)
        return np.tensordot(w,x,axes=[[ndim-2,ndim-1],[0,1]])
    elif x.shape[1] == w.shape[0]:
        return np.dot(x, w)
    elif x.shape[0] == w.shape[1]:
        return np.dot(w, x)
    else:print "ERRFROMME:The shapes of Weight and Input are not suit"

def gauss_kernel_fun(shape, sigma, mu=False):
    ndim = len(shape)
    axs = []
    gauss_x2 = 0
    for dim in range(ndim):
        reshape_tmp = np.ones(ndim)
        reshape_tmp[dim] = shape[dim]
        ax_r = (shape[dim] - 1) / 2.
        ax = np.linspace(-ax_r, ax_r, shape[dim]).reshape(reshape_tmp)
        if mu:
            if isinstance(mu, (float,int)):
                ax += mu
            elif ndim == len(mu):
                ax += mu[dim]
            else: print "@gauss_kernel_fun: Error @ input 'mu'..."
        if isinstance(sigma, (float,int)):
            gauss_x2 = gauss_x2 + ((ax/sigma)**2)/2
        elif ndim == len(sigma):
            gauss_x2 = gauss_x2 + ((ax/sigma[dim])**2)/2
        else: print "@gauss_kernel_fun: Error @ input 'sigma'..."
    gaussmf = np.exp(-gauss_x2)
    gaussmf_norm = gaussmf / gaussmf.max()
    return gaussmf_norm