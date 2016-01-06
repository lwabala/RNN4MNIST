import autograd.numpy as np
import autograd.numpy.random as npr
from math_fun_store import *
from rnn_fun import *

class Weights(object):
    def __init__(self, param_scale):
        self.w_num = 0
        self.idxs_and_shapes = {}
        self.w_config = []
        self.init_scale = param_scale
        self.data = False

    def init(self,w_config):
        for name,shape in w_config:
            if ~self.idxs_and_shapes.has_key(name):
                self.add_shape(name, shape, add_to_config=False)
        if ~self.data:
            self.data = npr.RandomState().randn(self.w_num) * self.init_scale

    def add_shape(self, name, shape, add_to_config=True):
        start = self.w_num
        self.w_num += np.prod(shape)    #prod:product of all members
        self.idxs_and_shapes[name] = (slice(start, self.w_num), shape)
        if add_to_config: self.w_config.append((name,shape))

    def get(self, w_data, name):
        idxs, shape = self.idxs_and_shapes[name]
        return np.reshape(w_data[idxs], shape)

    def find(self, idx):
        for name in self.idxs_and_shapes:
            idxs, _ = self.idxs_and_shapes[name]
            if idx in range(self.w_num)[idxs]:
                return name
                break


class Layer(object):
    def __init__(self, size, actfun=lambda x:x):
        self.size = size
        self.actfun = actfun
        self.input = False
        self.output = False
        self.axes = range(1, 1 + len(size))

    def trans(self, layer, w, b, axes=False):
        if ~axes: axes = (layer.axes, range(len(layer.axes)))
        x_dot_w = np.tensordot(layer.output, w, axes=axes)
        if np.any(b): self.input = x_add_bias(x_dot_w, b)
        else:         self.input += x_dot_w
        return self

    def act(self):
        self.output = self.actfun(self.input)


class RnnLayers(object):
    def __init__(self):
        pass
    '''
    def __init__(self, layer_config, weights_config):
        self.w = Weights(weights_config)
        for name,shape,actfun in layer_config:
            setattr(self, name, Layer(shape, actfun=actfun))
    def layer(self, name):pass
    '''

class QLearning(object):
    def __init__(self, w, qnet_size, act_size, data_length=256, w_update_step=64):
        self.dataset = []
        self.data_length = data_length
        self.w_update_step = w_update_step
        self.step = 0
        self.qnet = Layer(qnet_size, actfun=relu)
        self.action = Layer(act_size, actfun=sigmoid)
        self.w_prev = w
        self.state_prev = []

    def update_prev_weight(self, w):
        if self.step >= self.w_update_step: 
            self.w_prev = w
            self.step = 0
        self.step += 1

    def select_action_and_value(self, w_now=False):
        if isinstance(w_now, Weights):
            w,b = w_now.qnet_act, w_now.qnet_act_b
        elif ~w_now:
            w,b = self.w_prev.qnet_act, self.w_prev.qnet_act_b
        else: print"Error@select_action_and_value"
        self.action.trans(self.qnet, w, b).act()
        action = self.action.output
        ax = len(self.action.output.shape)-2
        value = np.max(np.max(self.action.output, axis=ax), axis=ax)
        return action, value.reshape([self.action.output.shape[0],-1])

    def pack_to_experience_dataset(self, act_and_value, reward, glimpse):
        if len(self.dataset) > self.data_length: self.dataset.pop(0)
        state_prev,state = self.state_prev, self.qnet.output
        e = (state_prev, act_and_value, reward, state, glimpse)
        self.dataset.append(e)
        self.state_prev = self.qnet.output