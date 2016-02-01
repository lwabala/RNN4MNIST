from copy import deepcopy
import winsound
from math_fun_store import *


def make_batches(train_num, batch_size):
    # set batches index
    return [ slice(ii, ii+batch_size) 
        for ii in range(0, train_num-batch_size, batch_size) ]

def x_add_bias(x,b):
    '''Add the bias to the input X'''
    try: batch_size = x.shape[0]
    except: print "Error@ x_add_bias(x,b): x.shape=%s" %x.shape
    if isinstance(b, (int,float)): x_with_b = x + b        
    else: x_with_b = x + b.repeat(batch_size, axis=0)
    return x_with_b

def log_loss(pred, targ, select_correct=False):
    EPS=1e-2
    if select_correct:
        lab_pred = np.argmax(pred, axis=len(pred.shape)-1)
        lab_targ = np.argmax(targ, axis=len(targ.shape)-1)
        correct = lab_pred == lab_targ
        pred, targ = pred[correct], targ[correct]
    ax = len(targ.shape)
    batch_size = targ.shape[0]
    likelihood = np.multiply(targ, pred) + np.multiply(1.-targ, 1.-pred)
    likelihood_norm = likelihood + EPS
    log_likelihood = np.log(likelihood_norm)
    lost = -np.sum(log_likelihood, axis=ax-1).reshape([batch_size])
    return lost

def action_to_slice_img(imgs, slice_size, acts):
    batch_size = len(imgs)
    img_size = np.array(imgs[0].shape)
    act_size = np.array(acts[0].shape)
    scale = (img_size - slice_size) / act_size
    actions = matrix_argmax_batch(acts)
    act_2_pos = lambda act: act * scale
    outputs = np.zeros([batch_size,] + slice_size)
    for it in range(batch_size):
        action = actions[it]
        xx,yy = act_2_pos(action)
        slice_x = slice(xx, xx + slice_size[0])
        slice_y = slice(yy, yy + slice_size[1])
        outputs[it] = imgs[it, slice_x, slice_y]
    return outputs

def init_acts_in_order(actnum=[5,5], actmat=[5,5]):
    for it in range(len(actnum)):
        if actnum[it] > actmat[it]: actmat[it] = actnum[it]
    actset=[]
    for ii in range(actnum[0]):
        for jj in range(actnum[1]):
            action = np.zeros(actnum)
            actmatrix = np.zeros([1,] + list(actmat))
            action[ii,jj] = 1
            xx , yy = np.floor(np.array(actmat) - np.array(actnum))
            slice_x = slice(xx, xx + actnum[0])
            slice_y = slice(yy, yy + actnum[1])
            actmatrix[0][slice_x,slice_y] = action
            actset.append(actmatrix)
    return actset

def learning_rate_adjust(learning_rate, epoch, accuracy):
    learning_rate_down = 0
    if epoch > 20:
        ac_down_num = np.sum(np.diff(accuracy[-5:],2) < 0)
        if learning_rate_down >= 10:
            learning_rate *= 2.0
            learning_rate_down = 0
        if ac_down_num > 2:
            learning_rate *= 0.90
            learning_rate_down += 1
    return learning_rate

def act_cls_value(actions):
    batch_size = actions.shape[0]
    cls_ax = len(actions.shape)-1
    action = np.max(actions, axis=cls_ax)
    argmax_x = np.argmax(np.max(action, axis=cls_ax-1), axis=cls_ax-2)
    argmax_y = np.argmax(np.max(action, axis=cls_ax-2), axis=cls_ax-2)
    value = np.max(np.max(action, axis=2), axis=1)
    #cls_mat = np.argmax(actions, axis=cls_ax)
    classify = actions[range(batch_size),argmax_x,argmax_y,:]
    return classify.reshape([batch_size,1,-1]), action, value

def result_analysis(correct_list):
    first_correct = np.ones(correct_list[0].shape) * np.inf
    for order in range(len(correct_list)):
        correct = correct_list[order]
        if np.any(correct):
            first_correct[correct & np.isinf(first_correct)] = order + 1
    final_correct = np.mean(correct)
    first_correct_once = first_correct[~ np.isinf(first_correct)]
    first_correct_once = np.array(first_correct_once, dtype=int)
    wrong_later = (first_correct < (len(correct_list) - 1)) & (~ correct)
    if not first_correct_once.size: 
        first_correct_once, wrong_later = np.nan, np.nan
    return first_correct_once, wrong_later

def not_grad_node(node, copy=True):
    obj = deepcopy(node) if copy else node
    if hasattr(obj, 'tapes'): obj.tapes = {}
    if hasattr(obj, 'value'): obj = obj.value
    return obj