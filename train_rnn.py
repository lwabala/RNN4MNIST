import winsound
from autograd import grad
from autograd import value_and_grad
from autograd.util import quick_grad_check
from build_rnn import *
from read_mnist import *

if not 'train_imgs' in dir():
    print "Training MNIST DATA not exist,now unpacking training data..."
    # MNIST training data and labels
    train_labs, train_imgs, train_num = read_all_mnist(1024*2, file='train', mod='full')
    # MNIST test data and labels
    test_labs, test_imgs, test_num = read_all_mnist(1024*1, file='test', mod='full')
    print "Training img num = %s & Test img num = %s" %(train_num, test_num)
else:
    print "Training MNIST DATA exist,now start building RNN..."

if __name__ == '__main__':
    global w, layer, w_config, actset, step, w_data_prev, q_dataset
    # Network parameters
    layer_config  = ([8,8], [16,16], [12,12], [1,10])
    q_net_config = ([16,16], [5,5])
    
    # Training parameters
    w_scale = 0.1
    momentum = 0.8
    batch_size = 256
    num_epochs = 5000
    learning_rate = 10
    print "W.scale=%s|mom=%s|bs=%s|ep=%s|lr=%s" %(
        w_scale,momentum,batch_size,num_epochs,learning_rate)
    
    (clac_loss, w, process_one_batch, accuracy
        ) = rnn_for_mnist(layer_config, q_net_config, w_init_scale=w_scale)
    weights = w.data
    # init backforword gradient matrix
    grad_back = np.zeros(weights.shape)

    # set batches index
    def make_batches(train_num, batch_size):
        return [ slice(ii, ii+batch_size) 
                for ii in range(0, train_num-batch_size, batch_size) ]
    batch_idxs = make_batches( train_imgs.shape[0] , batch_size )
    
    # training function & backword gradient
    def check_loss(weights, train_imgs, train_labs):
        return clac_loss(train_imgs, weights, train_labs, mode='test')
    idxs = npr.RandomState().rand(train_num) < 0.1
    training_grad = value_and_grad(clac_loss, argnum=1)
    try: 
        quick_grad_check(check_loss, 
            weights, (train_imgs[idxs], train_labs[idxs]), verbose=False)
        print "Grad check OK ! ..."
    except: print "Grad check failed ..."

    # print out information
    #def print_perf(epoch, weights, loss_and_grad):
    def print_and_store(learning_rate):
        train_result = accuracy(train_imgs, weights, train_labs)
        test_result  = accuracy(test_imgs, weights, test_labs)
        tr_ac, tr_mean, tr_min, tr_max, tr_std, tr_wl = train_result
        te_ac, te_mean, te_min, te_max, te_std, te_wl = test_result
        print ("%5s|%.2f|%.2f|cost=%.6s|w2=%.6s|gd2=%.8s|lr=%3.3f|"+
            "tr=%.2f %.1f %.1f %.1f %.1f|te=%.2f %.1f %.1f %.1f %.1f") %(
            epoch,
            tr_ac*100, te_ac*100,
            np.mean(costs),
            np.linalg.norm(weights),
            np.mean(grad_norm),
            learning_rate,
            tr_mean, tr_min, tr_max, tr_std, tr_wl*100,
            te_mean, te_min, te_max, te_std, te_wl*100)
        return train_result, test_result
    
    for epoch in range(num_epochs):
        costs,grad_norm = [],[]
        for idxs in batch_idxs:
            imgs_batch, labs_batch = train_imgs[idxs], train_labs[idxs]
            loss_and_grad = training_grad(imgs_batch, weights, labs_batch,mode='train')
            grad_back *= momentum 
            grad_back += loss_and_grad[1] * (1 - momentum)
            weights -= learning_rate * grad_back
            costs.append(loss_and_grad[0])
            grad_norm.append(np.linalg.norm(loss_and_grad[1]))
        print_and_store(learning_rate)