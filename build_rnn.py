from rnn_class import *
#from memory_profiler import profile

class Rnn4Mnist(object):
    def __init__(self,
                 glimpse=9,
                 layer_config=([8,8], [12,12], [16,16], [20,20], [24,24], [5,5,10]),
                 w_init_scale=0.10):
        self.glimpse = glimpse
        print "Glimpse = %s" %self.glimpse
        #x_size, h0_size, h1_size, h2_size, h3_size, act_size = layer_config
        x_size, h1_size, h2_size, act_size = [8,8], [256,], [576,], [5,5,10]
        gate_size, cls_size = [1,], [1,10]
        w_config = [#(weight_name ,(layers' shape) ,has_bias_weights)
                    ('x_h1',       (x_size, h1_size),    True),
                    ('h2_re_h2',   (h2_size, h2_size),   False),
                    ('gate_h2',    (h2_size, gate_size), True),
                    ('h1_h2',      (h1_size, h2_size),   True),
                    ('h2_act',     (h2_size, act_size),  True)]
        self.w = Weights(w_config, w_init_scale)
        layer_config = [#(layer_name ,layer_size ,active_fun)
                        ('x',      x_size,    False),
                        ('h1',     h1_size,   relu),
                        ('h2',     h2_size,   relu),
                        ('gate_h2',   gate_size, sigmoid(scale=0.1)),
                        ('action', act_size,  sigmoid(scale=0.3))]
        self.lyr = RnnLayers(layer_config)
        for lyr_config in self.lyr.config: print lyr_config
        self.act_in_ord = init_acts_in_order(actnum=act_size[0:2])

    def qnet_process(self, w_prev=False):
        '''Get input_img & recurrent_signal -> action(with cls) & value'''
        # set weight (self.w.data) as a attribute to self.w
        self.w.set_weights_from_data(self.w.data, self.w.config)
        # input_x(t) -> h0(t) ->h1(t)
        self.lyr.h1(self.lyr.x * self.w.x_h1)
        # h1(t) & h2(t-1) -> h2(t) -> action(t)
        self.lyr.h2 *= self.lyr.gate_h2(self.lyr.h2 * self.w.gate_h2)
        self.lyr.h2(self.lyr.h1 * self.w.h1_h2 + 
                    self.lyr.h2 * self.w.h2_re_h2)
        self.lyr.action(self.lyr.h2 * self.w.h2_act)
        action_batch = self.lyr.action.out
        classify, action, value = act_cls_value(action_batch)
        return value, action, classify

    def process_one_batch(self, imgs, targets):
        '''Whole Process of the q-rnn to classify a MNIST img by several glimpse'''
        batch_size = imgs.shape[0]
        classifies, actions, values, rewards = [], [], [], []
        act_mat = self.act_in_ord[np.int((len(self.act_in_ord) - 1) / 2)]
        action = act_mat.repeat(batch_size, axis=0)
        self.lyr.x.out = action_to_slice_img(imgs, self.lyr.x.size, action)
        self.lyr.h2.out = np.zeros([batch_size,] + self.lyr.h2.size)
        for order in range(self.glimpse):
            value, action, classify = self.qnet_process()
            reward = self.reward_for_cls(classify, targets, order)
            # pick up local img from origin img
            self.lyr.x(action_to_slice_img(imgs, self.lyr.x.size, action))
            classifies.append(classify)
            actions.append(action)
            values.append(value)
            rewards.append(reward)
        return classifies, actions, values, rewards

    def qnet_loss(self, w_data, imgs, targets):
        self.w.data = w_data
        classifies, actions, values, rewards = self.process_one_batch(imgs, targets)
        gamma = 0.9
        batch_size = values[0].shape[0]
        lost = np.float64(0.)
        for order in range(self.glimpse):
            if order == self.glimpse - 1:     targ_value = rewards[order]
            else: targ_value = gamma * values[order + 1] + rewards[order]
            value = values[order]
            # reinforcement learning loss:
            qnet_loss = np.linalg.norm(targ_value - value)
            lost += qnet_loss / (batch_size * self.glimpse)
            if np.isnan(lost).any(): 
                print "Error@qnet_loss: Exists NaN in reinforccce_loss..."
            # hybrid supervised classify loss:
            cls_loss = np.sum(log_loss(classifies[order], targets))
            cls_loss_sclae = (order+1) / self.glimpse
            lost += cls_loss / (batch_size * self.glimpse) * cls_loss_sclae
            if np.isnan(lost).any(): 
                print "Error@qnet_loss: Exists NaN in cls_loss..."
        return lost + self.regularization_loss()

    def regularization_loss(self):
        lost = 0.
        w_data = self.w.data
        w_data_abs = np.abs(w_data)
        lost += np.linalg.norm(w_data) * 0.001
        lost += np.sum(w_data_abs[w_data_abs > 0.1]) * 0.01
        lost += np.sum(w_data_abs[w_data_abs > 1.]) * 0.1
        lost += np.sum(w_data_abs[w_data_abs > 10.]) * 1.
        return lost

    def cls_accuracy(self, w_data, imgs, targets):
        batch_size = targets.shape[0]
        lab_targ = np.argmax(targets, axis=len(targets.shape)-1)
        self.w.data = w_data
        classify, action, _, _ = self.process_one_batch(imgs, targets)
        correct_list = []
        for order in range(len(classify)):
            pred = classify[order]
            lab_pred = np.argmax(pred, axis=len(pred.shape)-1)
            correct = (lab_targ == lab_pred)
            correct_list.append(correct)
        return correct_list, action

    def reward_for_cls(self, pred, targ, order):
        # several alternative reward function for test
        ax1, ax2 = len(pred.shape)-1, len(targ.shape)-1
        r = (np.argmax(pred, axis=ax1) == np.argmax(targ, axis=ax2))
        glimpse = self.glimpse - 1
        #if order == self.glimpse - 1:
        if order >= 0:
            return (r - ~r) * 1.0
            #return r * 1.0
            #return r * (order * 2 - glimpse) + (~r) * (-order/np.float(glimpse))
        else:
            return r * 0.
            #return (-r) * 0.1