from autograd import grad
from autograd import value_and_grad
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from buildRNN import *
from readmnist import *

if ~('train_images' in dir()):
    # MNIST training data and labels
    train_images, train_labels, N_data = readMNIST_all(file='train')
    # MNIST test data and labels
    test_images, test_labels, _ = readMNIST_all(file='test')

if __name__ == '__main__':
    # Network parameters
    Input_size,H_1_size,H_2_size,Output_size=14*14,200,80,10
    print Input_size,H_1_size,H_2_size,Output_size

    # Training parameters
    param_scale=0.1
    learning_rate = 0.01
    momentum = 0.8
    batch_size = 512
    num_epochs = 500

    # training function & backword gradient
    ClacCost , num_weights , POB , ac = RNN4MNIST( Input_size,H_1_size,H_2_size,Output_size )
    loss_and_grad = value_and_grad(ClacCost)

    # set batches index
    def make_batches(N_data, batch_size):
        return [ slice(i, min(i+batch_size, N_data)) for i in range(0, N_data, batch_size) ]
    batch_idxs = make_batches( train_images.shape[1] , batch_size )

    # init random weights
    rs = npr.RandomState()
    W = rs.randn(num_weights) * param_scale
    # init backforword gradient matrix
    W_back = np.zeros(num_weights)


    # print out information
    def print_perf(epoch, W,lag):
        test_perf  = ac(W, test_images, test_labels)
        train_perf = ac(W, train_images, train_labels)
        print "%5s|%.6s|%.6s|cost=%.8s|gdnorm=%.8s" \
        %(epoch, train_perf, test_perf,lag[0],np.sum(np.abs(lag[1])))
        return train_perf,test_perf

    # record the loss , training & test accuracy
    LOSS=[]
    TRAIN_ac=[]
    TEST_ac=[]

    for epoch in range(num_epochs):
        for idxs in batch_idxs:
            #print train_images.shape
            lag=loss_and_grad(W,train_images[:,idxs,:], train_labels[idxs,:])
            W_back = momentum * W_back + (1.0 - momentum) * lag[1]
            W -= learning_rate * W_back
            
        train_perf,test_perf=print_perf(epoch, W,lag)
        LOSS.append(lag[0])
        TRAIN_ac.append(train_perf)
        TEST_ac.append(test_perf)

    # output pic & save data
    x=range(len(LOSS))
    
    plt.plot(x,LOSS)
    plt.savefig("LOSS.png")
    plt.clf()

    plt.plot(x,TRAIN_ac , x,TEST_ac)
    plt.savefig("Accuracy.png")
    plt.clf()
    
    np.save('LOSS_TRAINING_TEST',(LOSS,TRAIN_ac,TEST_ac))




