import autograd.numpy as np
import autograd.numpy.random as npr
from autograd.numpy import array as npa

def loss(pred,targ):
    # pred=pred/np.sum(pred)
    # log_likelihood = np.sum(np.log(np.multiply(targ,pred)+np.multiply(1.-targ,1.-pred)))
    # cost = -log_likelihood
    dist=pred-targ
    cost=np.sum(np.linalg.norm(dist))
    return cost

def sigmoid(x):
    return 0.5*(np.tanh(x) + 1.0)   # Output ranges from 0 to 1.
def relu(x):
    return np.maximum(x,0.)
def softmax(x):
    expx=np.exp(x)
    return expx

# a Parser to select the index range in which the weights 
# between two layers from the whole weights vector
class WeightsParser(object):
    def __init__(self):
        self.idxs_and_shapes = {}
        self.num_weights = 0

    def add_shape(self, name, shape):
        start = self.num_weights
        self.num_weights += np.prod(shape)    #prod:product of all members
        self.idxs_and_shapes[name] = (slice(start, self.num_weights), shape)

    def get(self, vect, name):
        idxs, shape = self.idxs_and_shapes[name]
        return np.reshape(vect[idxs], shape)


def RNN4MNIST(Input_size,H_1_size,H_2_size,Output_size):
    parser = WeightsParser()
    parser.add_shape( 'PreInput' , (1,H_2_size) )
    parser.add_shape( 'Input_2_H1' , (Input_size+H_2_size+1,H_1_size) )
    parser.add_shape( 'H1_2_H2' , (H_1_size+1,H_2_size) )
    parser.add_shape( 'H2_2_Output' , (H_2_size+1,Output_size) )

    Hidden_actfun=relu
    Output_actfun=sigmoid

    def Weight_ActOn_Input(Input,Weight):
        #print Input.shape,Weight.shape # just for debug
        if Input.shape[1]==Weight.shape[0]:
            return np.dot(Input,Weight)
        else:
            print "ERRFROMME:The shapes of Weight and Input are not suit"

    def addbias(inputs):
        #print inputs.shape
        return np.concatenate((inputs,np.ones([inputs.shape[0],1])),axis=1)

    def ProcessOneBatch(Weights,Inputs):
        Weight_In2H1=parser.get(Weights,'Input_2_H1')
        Weight_H12H2=parser.get(Weights,'H1_2_H2')
        Weight_H22Out=parser.get(Weights,'H2_2_Output')
        PreInput=np.repeat(parser.get(Weights,'PreInput'),Inputs.shape[1],axis=0)
        OUTPUTS=[]
        for Input_unit in Inputs:
            Input_all=np.concatenate((Input_unit,addbias(PreInput)),axis=1)
            H1_in=Weight_ActOn_Input(Input_all,Weight_In2H1)
            H1_out=Hidden_actfun(H1_in)
            H2_in=Weight_ActOn_Input(addbias(H1_in),Weight_H12H2)
            H2_out=Hidden_actfun(H2_in)
            PreInput=H2_out
            Output_in=Weight_ActOn_Input(addbias(H2_out),Weight_H22Out)
            Output_out=Output_actfun(Output_in)
            OUTPUTS.append(Output_out)
        return OUTPUTS

    def ClacCost(Weights,Inputs,Target):
        OUTPUTS=ProcessOneBatch(Weights,Inputs)
        cost=0
        
        for out in OUTPUTS:
            cost*=0.9
            cost+=loss(out,Target)
        
        cost=loss(OUTPUTS[-1],Target)
        return cost

    def accuracy(Weights,Inputs,Target):
        ac = np.argmax(Target,axis=1) == np.argmax(ProcessOneBatch(Weights,Inputs)[-1],axis=1)
        return np.mean( ac )

    return  ClacCost , parser.num_weights , ProcessOneBatch , accuracy
    