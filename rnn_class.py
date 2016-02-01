import autograd.numpy.random as npr
from rnn_fun import *

class Weights(object):
    def __init__(self, w_config, w_init_scale):
        self.w_num = 0
        self.idxs_and_shapes = {}
        self.config = w_config
        self.init_scale = w_init_scale
        self.data = False
        for config in self.config:
            name, lyr_size, has_bias = config
            if not self.idxs_and_shapes.has_key(name):
                self.add_shape(config, add_to_config=False)
        if not self.data:
            self.data = npr.RandomState().randn(self.w_num) * self.init_scale

    def add_shape(self, config, add_to_config=True):
        name, lyr_size, has_bias = config
        start = self.w_num
        shape = lyr_size[0] + lyr_size[1]
        self.w_num += np.prod(shape)    #prod:product of all members
        slice_weight = [slice(start, self.w_num),]
        if has_bias: 
            start = self.w_num
            self.w_num += np.prod(lyr_size[1])
            slice_bias = slice(start, self.w_num)
            slice_weight.append(slice_bias)
        self.idxs_and_shapes[name] = (slice_weight, lyr_size, has_bias)
        if add_to_config: self.w_config.append(config)

    def get(self, w_data, name):
        idxs, lyr_size, has_bias = self.idxs_and_shapes[name]
        shape = lyr_size[0] + lyr_size[1]
        weight = [w_data[idxs[0]].reshape(shape),]
        if has_bias:
            weight.append(w_data[idxs[1]].reshape([1,]+lyr_size[1]))
        return weight

    def set_weights_from_data(self, w_data, w_config):
        for name, lyr_size, has_bias in w_config:
            setattr(self, name, self.get(w_data, name))
        return self
    
    def change_weights(self, w_data, name, weight, is_bias=False):
        if not is_bias: w_slice = self.idxs_and_shapes[name][0][0]
        else:           w_slice = self.idxs_and_shapes[name][0][1]
        w_data[w_slice] = weight.reshape(-1)

    def find(self, idx):
        '''Find the name a weight scalar belongs to with its index'''
        for name in self.idxs_and_shapes:
            idxs, _ = self.idxs_and_shapes[name]
            for idx_slice in idxs:
                if idx in range(self.w_num)[idxs]:
                    return name

    def show(self, w_data, grad=False, fun=np.linalg.norm):
        norm_l1 = lambda x: np.sum(np.abs(x))
        norm_fun = norm_l1
        for name, lyr_size, has_bias in self.config:
            weight = self.get(w_data, name)
            num = np.prod(lyr_size[1])
            w = norm_fun(weight[0])
            b = norm_fun(weight[1]) if has_bias else np.nan
            if not isinstance(grad, bool): 
                gg = self.get(grad, name)
                gw = norm_fun(gg[0])
                gb = norm_fun(gg[1]) if has_bias else np.nan
                print_form =  "%15s\t=> |\t"
                print_form += "w = %.6s <- %.6s @ %.4s%%\t|\t"
                print_form += "b = %.6s <- %.6s @ %.4s%%"
                print print_form %(
                    name, w/num, gw/num, gw/w*100, b/num, gb/num, gb/b*100)
            else: print "%15s\t=> |\tw = %.6s\t| b = %.6s" %(name, w, b)

class Layer(object):
    def __init__(self, size, actfun=lambda x:x):
        self.size = size
        self.actfun = actfun if actfun else lambda x:x
        self.out = False

    def __mul__(self, weight):
        '''Method of multiply layer's output signal with weight'''
        x, w = self.out, weight[0]
        axes = (range(1, 1+len(self.size)), range(len(self.size)))
        w_dot_x = np.tensordot(x, w, axes=axes)
        if len(weight)==2:   return x_add_bias(w_dot_x, weight[1])
        elif len(weight)==1: return w_dot_x

    def __call__(self, signal, actfun=True):
        '''Method of produce the layer's output signal'''
        if actfun: self.out = self.actfun(signal)
        else:      self.out = signal
        return self

    def __imul__(self, gate):
        gate_size = [gate.out.shape[0]] + [1] * (len(self.out.shape) - 1)
        self.out = np.multiply(self.out, gate.out.reshape(gate_size))
        return self

class RnnLayers(object):
    def __init__(self, layer_config):
        self.config = layer_config
        for name, size, actfun in self.config:
            setattr(self, name, Layer(size, actfun=actfun))

class RMSPropGrad(object):
    def __init__(self, decay_rate):
        self.r = 1.
        self.dr = decay_rate
        self.scale = 1.
    def grad(self, grad):
        self.r *= self.dr
        self.r += (grad**2) * (1 - self.dr)
        rms_grad = np.divide(grad, np.sqrt(self.r))
        self.scale = np.linalg.norm(grad) / np.linalg.norm(rms_grad)
        return rms_grad * self.scale

class MomentumGrad(object):
    def __init__(self, decay_rate):
        self.r = 0.
        self.dr = decay_rate
    def grad(self, grad):
        self.r *= self.dr
        self.r += grad * (1 - self.dr)
        return self.r