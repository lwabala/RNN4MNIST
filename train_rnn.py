from autograd import value_and_grad
from autograd.util import quick_grad_check
from build_rnn import *
from read_mnist import read_all_mnist

if not 'train_imgs' in dir():
    print "Training MNIST DATA not exist,now unpacking training data..."
    # MNIST training data and labels
    train_labs, train_imgs, train_num = read_all_mnist(50000, file='train',mod='full')
    # MNIST test data and labels
    test_labs, test_imgs, test_num = read_all_mnist(10000, file='test',mod='full')
    print "Training img num = %s & Test img num = %s" %(train_num, test_num)
else:
    print "Training MNIST DATA exist,now start building RNN..."

if __name__ == '__main__':
    global w, layer, w_config, actset, step, w_data_prev, q_dataset
    
    # Training parameters
    w_scale = 0.1
    momentum = 0.9
    batch_size = 1000
    num_epochs = 5000
    learning_rate = 0.2
    print "W.scale=%s|momentum=%s|batch_size=%s|epoch=%s|learning_rate=%s" %(
                    w_scale, momentum, batch_size, num_epochs, learning_rate)
    
    rnn = Rnn4Mnist(w_init_scale=w_scale)
    weights = rnn.w.data
    
    # training function & backword gradient
    qnet_gradient = value_and_grad(rnn.qnet_loss, argnum=0)

    def print_result():
        # print out information
        train_result, action = rnn.cls_accuracy(weights, train_imgs, train_labs)
        test_result, _       = rnn.cls_accuracy(weights, test_imgs, test_labs)
        train_final_accuracy = np.mean(train_result[-1]) * 100
        test_final_accuracy  = np.mean(test_result[-1])  * 100
        #scale = np.devide(mom_grad_method.scale,mom_grad_method.r)
        #print np.mean(scale), np.min(scale), np.max(scale), np.std(scale)
        print ("epoch%5s | acccuraccy train:%.5s%% | test:%.5s%% | "
             + "cost=%.6s | l2_norm(weight)= %.6s | l2_norm(gardient)=%.6s | "
             + "learning_rate=%.6s") %(
            epoch, train_final_accuracy, test_final_accuracy,
            np.mean(cost), np.linalg.norm(weights), np.mean(grad_norm),
            learning_rate)
        return train_final_accuracy, test_final_accuracy

    # dataset to record the loss , training & test accuracy
    lost, train_ac, test_ac = [], [], []
    mom_grad_method = MomentumGrad(momentum)
    stop = False
    
    batch_idxs = make_batches(train_imgs.shape[0], batch_size)
    imgs, labs = [], []
    for idxs in batch_idxs:  # create or update experience of Q-net
        imgs.append(train_imgs[idxs])
        labs.append(train_labs[idxs])
    
    for epoch in range(num_epochs):
        cost, grad_norm = [], []
        batch_num = np.int(np.round(train_num / batch_size))
        for order in range(batch_num):
            rand_batch = np.random.rand(train_num) < (1. / batch_num)
            imgs_batch = train_imgs[rand_batch]
            labs_batch = train_labs[rand_batch]
            gradient = qnet_gradient(weights, imgs_batch, labs_batch)
            if np.isnan(gradient[1]).any():
                print "ERROR@epoch:Exist NaN in grad"
                stop = True
                rnn.w.show(weights, grad=gradient[1])
                break
            else:
                cost.append(gradient[0])
                grad_norm.append(np.linalg.norm(gradient[1]))
                qnet_grad = mom_grad_method.grad(gradient[1])
                weights -= learning_rate * qnet_grad
        if stop: break
        train_accuracy, test_accuracy = print_result()
        lost.append(np.mean(cost))
        train_ac.append(train_accuracy)
        test_ac.append(test_accuracy)