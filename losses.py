from keras import backend as K
import tensorflow as tf
from tensorflow.python.ops import array_ops

'''
Compatible with tensorflow backend
'''


def focal_loss(gamma=2., alpha=.25):
    def focal_loss_fixed(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) - K.sum(
            (1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))
    return focal_loss_fixed


# focal loss with multi label
def focal_loss_multiclass(classes_num, gamma=2., alpha=.25, e=0.1):
    # classes_num contains sample number of each classes
    def focal_loss_fixed(y_true, y_pred):
        '''
        prediction_tensor is the output tensor with shape [None, 100], where 100 is the number of classes
        target_tensor is the label tensor, same shape as predcition_tensor
        '''
        # 1# get focal loss with no balanced weight which presented in paper function (4)
        zeros = array_ops.zeros_like(y_pred, dtype=y_pred.dtype)
        one_minus_p = array_ops.where(tf.greater(y_true, zeros), y_true - y_pred, zeros)
        FT = -1 * (one_minus_p ** gamma) * tf.log(tf.clip_by_value(y_pred, 1e-8, 1.0))

        # 2# get balanced weight alpha
        classes_weight = array_ops.zeros_like(y_pred, dtype=y_pred.dtype)

        total_num = float(sum(classes_num))
        classes_w_t1 = [total_num / ff for ff in classes_num]
        sum_ = sum(classes_w_t1)
        classes_w_t2 = [ff / sum_ for ff in classes_w_t1]  # scale
        classes_w_tensor = tf.convert_to_tensor(classes_w_t2, dtype=y_pred.dtype)
        classes_weight += classes_w_tensor

        alpha = array_ops.where(tf.greater(y_true, zeros), classes_weight, zeros)

        # 3# get balanced focal loss
        balanced_fl = alpha * FT
        balanced_fl = tf.reduce_sum(balanced_fl)

        # 4# add other op to prevent overfit
        # reference : https://spaces.ac.cn/archives/4493
        nb_classes = len(classes_num)
        fianal_loss = (1 - e) * balanced_fl + e * K.categorical_crossentropy(K.ones_like(y_pred) / nb_classes, y_pred)

        return fianal_loss

    return focal_loss_fixed
