from autograd import grad
from autograd import value_and_grad
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from build_rnn import *
from read_mnist import *

global train_images, train_labels, img_all_num,test_images, test_labels

if not 'train_images' in dir():
    print "Training MNIST DATA not exist,now unpacking training data..."
    # MNIST training data and labels
    train_labels, train_images, img_all_num = read_all_mnist(file='train')
    # MNIST test data and labels
    test_labels, test_images, _ = read_all_mnist(file='test')
else:
    print "Training MNIST DATA exist,now start building RNN..."

if __name__ == '__main__':
    # Network parameters
    input_size, h1_size, h2_size, output_size = 14*14, 200, 80, 10
    print input_size, h1_size, h2_size, output_size

    # Training parameters
    param_scale = 0.1
    learning_rate = 0.01 / img_all_num
    momentum = 0.9
    batch_size = 512
    num_epochs = 5000

    # training function & backword gradient
    clac_loss, num_weights, p_o_b, accuracy, parser = \
        rnn_for_mnist(input_size, h1_size, h2_size, output_size)
    loss_and_grad = value_and_grad(clac_loss, argnum=1)

    # set batches index
    def make_batches(img_all_num, batch_size):
        return [ slice(i, min(i+batch_size, img_all_num)) 
                for i in range(0, img_all_num, batch_size) ]
    batch_idxs = make_batches( train_images.shape[1] , batch_size )

    # init random weights
    rs = npr.RandomState()
    weights = rs.randn(num_weights) * param_scale
    
    # init backforword gradient matrix
    weights_back = np.zeros(num_weights)


    # print out information
    def print_perf(epoch, weights, l_a_g):
        test_perf  = accuracy(test_images, weights, test_labels)
        train_perf = accuracy(train_images, weights, train_labels)
        print "%5s|%.6s|%.6s|cost=%.8s|gdnorm=%.8s" \
        %(epoch, train_perf, test_perf, l_a_g[0], np.sum(np.abs(l_a_g[1])))
        return train_perf, test_perf

    # record the loss , training & test accuracy
    lost = []
    train_ac = []
    test_ac = []
    # learning_rate_down = 0

    for epoch in range(num_epochs):
        for idxs in batch_idxs:
            #print train_images.shape
            imgs_in_batch = train_images[:,idxs,:]
            labs_in_batch = train_labels[idxs,:]
            l_a_g = loss_and_grad(imgs_in_batch, weights, labs_in_batch)
            weights_back = momentum * weights_back + (1.0 - momentum) * l_a_g[1]
            # weights_back = weights_back - np.mean( weights_back ) * 0.01
            # weights_back = weights_back * ( weights_back > 0 )
            weights -= learning_rate * weights_back
            
        train_perf, test_perf = print_perf(epoch, weights, l_a_g)
        lost.append(l_a_g[0])
        train_ac.append(train_perf)
        test_ac.append(test_perf)
        '''
        # learning_rate adjust
        if epoch > 10:
            ac_down_num = np.sum(np.diff(test_ac[-5:],2) < 0)
            if learning_rate_down >= 15:
                learning_rate *= 4.5
                learning_rate_down = 0
            if ac_down_num > 2 :
                learning_rate *= 0.95
                learning_rate_down += 1
        '''
    
    # output pic & save data
    x = range(len(lost))
    
    plt.plot(x, lost)
    plt.savefig("lost.png")
    plt.clf()

    plt.plot(x, train_ac, x, test_ac)
    plt.savefig("Accuracy.png")
    plt.clf()
    
    np.save('LOSS_TRAINING_TEST',(lost, train_ac, test_ac))
    np.save('RNN_SETTINGS_WEIGHTS',(
        (input_size, h1_size, h2_size, output_size),
        weights,
        parser))



(input_size, h1_size, h2_size, output_size),weights,parser=np.load('RNN_SETTINGS_WEIGHTS.npy')
