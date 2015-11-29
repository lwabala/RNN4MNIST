# Network parameters
Input_size,H_1_size,H_2_size,Output_size=14*14,200,80,10
print Input_size,H_1_size,H_2_size,Output_size

# Training parameters
param_scale=0.1
learning_rate = 0.01/N_data
momentum = 0.9
batch_size = 256
num_epochs = 500

# training function & backword gradient
ClacCost , num_weights , POB , ac , P = RNN4MNIST( Input_size,H_1_size,H_2_size,Output_size )
loss_and_grad = value_and_grad(ClacCost)

# set batches index
batch_idxs = make_batches( train_images.shape[1] , batch_size )

# init random weights
rs = npr.RandomState()
W = rs.randn(num_weights) * param_scale
# init backforword gradient matrix
W_back = np.zeros(num_weights)

LOSS=[]
TRAIN_ac=[]
TEST_ac=[]

#learning_rate = 0.0001
for epoch in range(num_epochs):
    for idxs in batch_idxs:
        #print train_images.shape
        lag=loss_and_grad(W,train_images[:,idxs,:], train_labels[idxs,:])
        W_back = momentum * W_back + (1.0 - momentum) * lag[1]
        # W_back = W_back - np.mean( W_back ) * 0.01
        # W_back = W_back * ( W_back > 0 )
        W -= learning_rate * W_back
        
    train_perf,test_perf=print_perf(epoch, W,lag)
    LOSS.append(lag[0])
    TRAIN_ac.append(train_perf)
    TEST_ac.append(test_perf)