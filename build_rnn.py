import autograd.numpy as np
import autograd.numpy.random as npr
from autograd.numpy import array as npa
EPS=1e-5

def loss(pred, targ):
    # pred=pred/np.sum(pred)
    likelihood = np.multiply(targ, pred) + np.multiply(1.-targ, 1.-pred)
    likelihood_norm = likelihood + EPS
    log_likelihood = np.sum(np.log(likelihood_norm))
    lost = -log_likelihood
    # dist=pred-targ
    # lost=np.sum(np.linalg.norm(dist))
    return lost

def sigmoid(x):
    return 1/(1+np.exp(-x))

def tanhn(x):
    return 0.5*(np.tanh(x) + 1.0)   # Output ranges from 0 to 1.

def relu(x):
    return np.maximum(x,0.)

def softmax(x):
    expx=np.exp(x)
    return expx


class WeightsParser(object):
    '''A Parser to select the weights we need from the whole Weights Vector'''
    def __init__(self):
        self.num_weights = 0
        self.idxs_and_shapes = {}
        self.weights_name = []

    def add_shape(self, name, shape):
        start = self.num_weights
        self.num_weights += np.prod(shape)    #prod:product of all members
        self.idxs_and_shapes[name] = (slice(start, self.num_weights), shape)
        self.weights_name.append(name)

    def get(self, vect, name):
        idxs, shape = self.idxs_and_shapes[name]
        return np.reshape(vect[idxs], shape)


def rnn_for_mnist(input_size, h1_size, h2_size, output_size):
    parser = WeightsParser()
    parser.add_shape('pre_h2_out', (1, h2_size))
    parser.add_shape('input_2_h1', (input_size+h2_size+1, h1_size))
    parser.add_shape('h1_2_h2', (h1_size+1, h2_size))
    parser.add_shape('h2_2_output', (h2_size+1, output_size))

    hidden_actfun = relu
    output_actfun = sigmoid

    def w_act_on_x(x, w):
        if x.shape[1] == w.shape[0]:
            return np.dot(x, w)
        else:
            print "ERRFROMME:The shapes of Weight and Input are not suit"

    def x_with_bias(x):
        '''Add a row of vector which value=1 for the bias to the input X

               x.shape=(batch_size, input_vector_length) 
            => x_with_bias(x).shape=(batch_size, input_vector_length + 1)
        '''
        batch_size = x.shape[0]
        return np.concatenate((x, np.ones([batch_size, 1])), axis=1)

    def process_one_batch(inputs, weights):
        '''Process one batch of image sets

        Recurrent Network process:
            h1_out = H( W[x_h1]*X(t) + W[h2_h1]*h2_out + bias[h1] )
            h2_out = H( W[h1_h2]*h1_out + bias[h2] )
            output = H( W[h2_out]*h2_out + bias[out] )
        '''
        batch_size = inputs.shape[1]
        w_in_2_h1 = parser.get(weights, 'input_2_h1')
        w_h1_2_h2 = parser.get(weights, 'h1_2_h2')
        w_h2_2_out = parser.get(weights, 'h2_2_output')
        h2_out = np.repeat(
            parser.get(weights, 'pre_h2_out'),
            batch_size,
            axis=0)
        outputs=[]
        for sub_input in inputs:
            input_x = x_with_bias(np.concatenate((sub_input, h2_out), axis=1))
            h1_in = w_act_on_x(input_x, w_in_2_h1)
            h1_out = hidden_actfun(h1_in)
            h2_in = w_act_on_x(x_with_bias(h1_out), w_h1_2_h2)
            h2_out = hidden_actfun(h2_in)
            output_in = w_act_on_x(x_with_bias(h2_out), w_h2_2_out)
            output_out = output_actfun(output_in)
            outputs.append(output_out)
        return outputs

    def clac_loss(inputs, weights, targets):
        ''' claculate loss of one batch
        # clac loss of each prediction of sub img and add them together
        # but the earlier prediction donate less loss
        '''
        outputs = process_one_batch(inputs, weights)
        lost = 0
        loss_memory_rate = 0.8

        for out in outputs:
            lost *= loss_memory_rate
            lost += loss(out, targets)
        lost /= 4
        # lost=loss(outputs[-1],targets)
        return lost

    def accuracy(inputs, weights, targets):
        lab_targs = np.argmax(targets, axis=1)
        lab_preds = np.argmax(process_one_batch(inputs, weights)[-1], axis=1)
        ac = (lab_targs == lab_preds)
        return np.mean(ac)

    return  clac_loss, parser.num_weights, process_one_batch, accuracy, parser
    