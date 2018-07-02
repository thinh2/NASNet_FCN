import keras.backend as K
import tensorflow as tf
import sys

def loss_fn(y_true, y_pred):
    #y_true = tf.squeeze(y_true)
    #y_true = tf.cast(y_true, dtype=tf.int32)
    #return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
    #                                        labels=y_true, logits=y_pred))
    y_pred = K.reshape(y_pred, (-1, K.int_shape(y_pred)[-1]))
    log_softmax = tf.nn.log_softmax(y_pred)

    y_true = K.one_hot(tf.to_int32(K.flatten(y_true)), K.int_shape(y_pred)[-1])
    #y_true = tf.stack(unpacked[:-1], axis=-1)

    cross_entropy = -K.sum(y_true * log_softmax, axis=1)
    cross_entropy_mean = K.mean(cross_entropy)

    return cross_entropy_mean

def ensemble_loss_fn(y_true, y_pred):
    
    """
    default image size = 224, default last_dim = 6
    """
    batch_size = 10
    total_loss = tf.Variable(0, dtype=tf.float32)
    for idx in range(0, 3):
        tmp = tf.slice(y_pred, [idx, 0, 0, 0], [1, 224, 224, 6])
        tmp = tf.squeeze(tmp)
        if idx == 0:
            total_loss = loss_fn(y_true, tmp)
        else:
            total_loss = tf.add(total_loss, loss_fn(y_true, tmp))
    return total_loss
    

def fcn_xent_nobg(y_true, y_pred):
	y_true = y_true[:,:,:,1:]
	y_pred = y_pred[:,:,:,1:]

	y_true_reshaped = K.flatten(y_true)
	y_pred_reshaped = K.flatten(y_pred)

	return K.binary_crossentropy(y_pred_reshaped, y_true_reshaped)

def pixel_acc(y_true, y_pred):
	s = K.shape(y_true)

	# reshape such that w and h dim are multiplied together
	y_true_reshaped = K.reshape( y_true, tf.stack( [-1, s[1]*s[2], s[-1]] ) )
	y_pred_reshaped = K.reshape( y_pred, tf.stack( [-1, s[1]*s[2], s[-1]] ) )

	# correctly classified
	clf_pred = K.one_hot( K.argmax(y_pred_reshaped), nb_classes = s[-1])
	correct_pixels_per_class = K.cast( K.equal(clf_pred,y_true_reshaped), dtype='float32')

	return K.sum(correct_pixels_per_class) / K.cast(K.prod(s), dtype='float32')

def mean_acc(y_true, y_pred):
	s = K.shape(y_true)

	# reshape such that w and h dim are multiplied together
	y_true_reshaped = K.reshape( y_true, tf.stack( [-1, s[1]*s[2], s[-1]] ) )
	y_pred_reshaped = K.reshape( y_pred, tf.stack( [-1, s[1]*s[2], s[-1]] ) )

	# correctly classified
	clf_pred = K.one_hot( K.argmax(y_pred_reshaped), nb_classes = s[-1])
	equal_entries = K.cast(K.equal(clf_pred,y_true_reshaped), dtype='float32') * y_true_reshaped

	correct_pixels_per_class = K.sum(equal_entries, axis=1)
	n_pixels_per_class = K.sum(y_true_reshaped,axis=1)

	acc = correct_pixels_per_class / n_pixels_per_class
	acc_mask = tf.is_finite(acc)
	acc_masked = tf.boolean_mask(acc,acc_mask)

	return K.mean(acc_masked)

def mean_IoU(y_true, y_pred):
	s = K.shape(y_true)

	# reshape such that w and h dim are multiplied together
	y_true_reshaped = K.reshape( y_true, tf.stack( [-1, s[1]*s[2], s[-1]] ) )
	y_pred_reshaped = K.reshape( y_pred, tf.stack( [-1, s[1]*s[2], s[-1]] ) )

	# correctly classified
	clf_pred = K.one_hot( K.argmax(y_pred_reshaped), nb_classes = s[-1])
	equal_entries = K.cast(K.equal(clf_pred,y_true_reshaped), dtype='float32') * y_true_reshaped

	intersection = K.sum(equal_entries, axis=1)
	union_per_class = K.sum(y_true_reshaped,axis=1) + K.sum(y_pred_reshaped,axis=1)

	iou = intersection / (union_per_class - intersection)
	iou_mask = tf.is_finite(iou)
	iou_masked = tf.boolean_mask(iou,iou_mask)

	return K.mean( iou_masked )