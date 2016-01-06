import matplotlib as mpl
import matplotlib.pyplot as plt
from memory_profiler import profile
import autograd.numpy as np
import autograd.scipy as sp
from autograd.numpy import array as npa
from autograd.numpy import tensordot as tdot
from math_fun_store import *
from rnn_class import *
from rnn_fun   import *

def rnn_for_mnist(layer_config, q_net_config, w_init_scale=0.050):
    #layer_config = tuple([x_size, h1_size, h2_size, cls_size])
    x_size, h1_size, h2_size, cls_size = layer_config
    qnet_size, act_size = q_net_config
    global w, layer, w_config, act_in_order_set, step, w_data_prev, q_dataset

    w = Weights(w_init_scale)
    w_config=[('pre_h2_out', ([1,] + h2_size)),
                ('x_h1',       (x_size + h1_size)),
                ('h2_re_h1',   (h2_size + h1_size)),
                ('x_h1_b',     ([1,] + h1_size)),
                ('h1_h2',      (h1_size + h2_size)),
                ('h1_h2_b',    ([1,] + h2_size)),
                ('h2_cls',     (h2_size + cls_size)),
                ('h2_cls_b',   ([1,] + cls_size)),
                ('g_re',       (h2_size + [1,1])),
                ('g_re_b',     ([1,] + [1,1])),
                ('h2_qnet',    (h2_size + qnet_size)),
                ('h2_qnet_b',  ([1,] + qnet_size)),
                ('qnet_act',   (qnet_size + act_size)),
                ('qnet_act_b', ([1,] + act_size))]
    w.config = w_config
    w.init(w_config)
    w_data_prev = w.data
    print "layer settings: %s, %s & num(weights) = %s"%(layer_config, q_net_config, w.w_num)

    def action_to_slice_img(input_imgs, slice_size, actions):
        batch_size = len(input_imgs)
        act_arg = matrix_argmax_batch(actions)
        act_2_pos = lambda a: np.array(a)*5
        out_imgs = np.zeros([batch_size,]+slice_size)
        for order in range(batch_size):
            act = act_arg[order]
            xx,yy = act_2_pos(act)
            slice_x = slice(xx, xx + slice_size[0])
            slice_y = slice(yy, yy + slice_size[1])
            out_imgs[order] = input_imgs[order, slice_x, slice_y]
        return out_imgs
    
    def rnn_process(in_x, w_data, h2_out):
        global layer, w
        w_g_re = w.get(w_data,'g_re')
        w_g_re_b = w.get(w_data,'g_re_b')
        w_h2_re_h1 = w.get(w_data,'h2_re_h1')
        w_x_h1 = w.get(w_data,'x_h1')
        w_x_h1_b = w.get(w_data,'x_h1_b')
        w_h1_h2 = w.get(w_data,'h1_h2')
        w_h1_h2_b = w.get(w_data,'h1_h2_b')
        w_h2_cls = w.get(w_data,'h2_cls')
        w_h2_cls_b = w.get(w_data,'h2_cls_b')

        # gate @ h2 recurrent to input
        g_re_in = trans(h2_out, w_g_re, w_g_re_b)
        g_re = sigmoid(g_re_in, scale=0.3)
        h2_out = np.multiply(h2_out, g_re)
        in_x  = np.multiply(in_x, 1-g_re)
        # input x -> h1
        h1_in = trans(in_x, w_x_h1, w_x_h1_b)
        h1_in += trans(h2_out, w_h2_re_h1, 0)
        h1_out = relu(h1_in)
        # h1 -> h2
        h2_in = trans(h1_out, w_h1_h2, w_h1_h2_b)
        h2_out = relu(h2_in)
        # h2 -> classify prediction
        cls_in = trans(h2_out, w_h2_cls, w_h2_cls_b)
        cls_out = sigmoid(cls_in)
        return cls_out, h2_out

    step = 0
    q_dataset = []
    def qnet_process(input_imgs, state_prev, w_data, targets, order, mode='train'):
        global layer, w, act_in_order_set, batch_size, step, w_data_prev, q_dataset
        # update weights to Q-net every N step
        if mode=='train': 
            if step > 128:
                w_data_prev = w_data
                step = 0
            step += 1
            data = w_data_prev
        elif mode=='test': 
            data = w_data

        w_h2_qnet   = w.get(data,'h2_qnet')
        w_h2_qnet_b = w.get(data,'h2_qnet_b')
        w_qnet_act   = w.get(data,'qnet_act')
        w_qnet_act_b = w.get(data,'qnet_act_b')

        # get input of Q-net
        h2_out = state_prev
        # update state in Q-net
        q_net_in = trans(h2_out, w_h2_qnet, w_h2_qnet_b)
        q_net_out = relu(q_net_in)
        # clac action & action-value in Q-net
        q_act_in = trans(q_net_out, w_qnet_act, w_qnet_act_b)
        q_act_out = sigmoid(q_act_in)
        act = q_act_out
        ax = len(act.shape)-2
        value = np.max(np.max(act, axis=ax), axis=ax)

        # pick up local img from origin img
        in_x = action_to_slice_img(input_imgs, x_size, act)
        # run in rnn
        cls_out, h2_out = rnn_process(in_x, data, h2_out)
        # clac reward
        reward = reward_for_cls(cls_out, targets, order)
        state  = h2_out
        if (mode=='train') & (order>1):
            # record transition to q-net's experience dataset
            while len(q_dataset) >= 1024: q_dataset.pop(0)
            experience = (state_prev, act, value, reward, state, order)
            # weights' grad don't depend on experience
            # clear the tapes of primitive nodes recorded by Autograd
            for member in experience:
                if hasattr(member, 'tapes'):
                    member.tapes = {}
            q_dataset.append(experience)
        return act, value, cls_out, state
            
    # act_in_order_set = init_acts_in_order()
    
    def process_one_batch(input_imgs, w_data, targets, mode='train'):
        global layer, w, batch_size
        batch_size = input_imgs.shape[0]
        h2_out = w.get(w_data,'pre_h2_out').repeat(batch_size, axis=0)
        state_prev = h2_out

        classify, action = [], []
        for order in range(9):
            (act, value, cls_out, state_prev
                ) = qnet_process(
                input_imgs, state_prev, w_data, targets, order, mode=mode)
            classify.append(cls_out)
            action.append(act)
        return classify, action
    
    def clac_loss(inputs, w_data, targets, mode='train'):
        global layer, w, batch_size, q_dataset
        batch_size = targets.shape[0]
        classify, action = process_one_batch(inputs, w_data, targets, mode=mode)                
        lost = np.float64(0.)
        # select experience transition randomly from q_dataset
        rand_experience_batch = []
        for experience in q_dataset:
            if np.random.rand() < 0.5:
                rand_experience_batch.append(experience)
        # loss by q-net
        train_num = len(rand_experience_batch)
        for (state, act, value, reward, state_next, order
            ) in rand_experience_batch:
            targ_value = target_value(
                inputs, reward, w_data, targets, state_next, order)
            lost += np.linalg.norm(targ_value - value)# / train_num
        lost = lost / batch_size
        # prevent self-excitation in recurrent (h2->h1->h2)
        norm_set = []
        w_h2_re_h1 = w.get(w_data,'h2_re_h1')
        w_h1_h2    = w.get(w_data,'h1_h2')
        norm_set.append(np.abs(dot(w_h2_re_h1, w_h1_h2)))
        norm_set.append(np.abs(dot(w_h1_h2, w_h2_re_h1)))
        for norm in norm_set:
            lost += np.sum(norm[ norm> 1. ]) * 0.5
        if np.isnan(lost).any(): print "Error@clac_loss: Exists NaN in lost..."
        return lost
    
    def target_value(inputs, reward, w_data, targets, state_next, order):
        # target value we want q-net to be, used in claculating loss
        global w, batch_size
        glimpse_max = 9
        gamma = 0.99
        if order == glimpse_max:
            return reward
        else:
            state_prev = state_next
            (next_act, next_value, next_cls_out, next_state
                ) = qnet_process(
                inputs, state_prev, w_data, targets, False, mode='test')
            targ_value = reward + gamma * next_value
            return targ_value

    def accuracy(inputs, w_data, targets):
        batch_size = targets.shape[0]
        lab_targ = np.argmax(targets, axis=len(targets.shape)-1)
        corrects = []
        first_correct = np.ones(lab_targ.shape)*np.Inf

        classify, action = process_one_batch(inputs, w_data, targets, mode='test')
        glimpse = len(classify)
        for order in range(glimpse):
            pred = classify[order]
            lab_pred = np.argmax(pred, axis=len(pred.shape)-1)
            correct = (lab_targ == lab_pred)
            corrects.append(correct)
            first_correct[correct] = np.minimum(first_correct[correct], np.ones(np.sum(correct))*order)
        wrong_later = (first_correct < glimpse) & (~correct)
        result = []
        final_correct = np.mean(correct)
        any_correct = first_correct[first_correct<10]
        result.append(final_correct)
        result.append(np.mean(any_correct)+1)
        result.append(np.min(any_correct)+1)
        result.append(np.max(any_correct)+1)
        result.append(np.std(any_correct))
        result.append(np.mean(wrong_later))
        return result
    
    def reward_for_cls(pred, targ, order):
        ax1, ax2 = len(pred.shape)-1, len(targ.shape)-1
        r = (np.argmax(pred, axis=ax1) == np.argmax(targ, axis=ax2))
        '''
        if order==9: return r*100
        else: return r*0
        '''
        return -log_loss(pred, targ)
        # return r*100

    return  (clac_loss, w, process_one_batch, accuracy)
    