
import numpy as np
import tensorflow as tf
from layers import *
from load import *

####################################

gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)

####################################

(x_train, y_train), (x_test, y_test) = load()
x_train = x_train.astype(np.float32)
y_train = y_train.astype(np.int64)
x_test = x_test.astype(np.float32)
y_test = y_test.astype(np.int64)

####################################

tmp, counts = np.unique(y_train, return_counts=True)
assert (np.all(tmp == [0,1,2,3]))

'''
weight = counts / np.sum(counts)
'''

'''
alpha_0 = counts[0] / np.sum(counts)
alpha_1 = counts[1] / np.sum(counts)
alpha_2 = counts[2] / np.sum(counts)
alpha_3 = counts[3] / np.sum(counts)
'''

# A value pos_weight > 1 decreases the false negative count, hence increasing the recall. 
# Conversely setting pos_weight < 1 decreases the false positive count and increases the precision. 
# This can be seen from the fact that pos_weight is introduced as a multiplicative coefficient for the positive labels term in the loss expression:

weight = counts / np.sum(counts)
weight = 1. / weight
weight = weight / np.max(weight)
weight[1] *= 100
weight[2] *= 300
weight[3] *= 100
weight_tf = tf.constant(weight, dtype=tf.float32)

####################################

model = model(layers=[

conv_block((3,3,1,32)),
max_pool(2),

conv_block((3,3,32,32)),
max_pool(2),

conv_block((3,3,32,64)),
max_pool(2),

conv_block((3,3,64,64)),
up_pool(2),

conv_block((3,3,64,32)),
up_pool(2),

conv_block((3,3,32,32)),
up_pool(2),

conv_block((3,3,32,32)),
conv_block((3,3,32,4), relu=False)
])

####################################

params = model.get_params()
optimizer = tf.keras.optimizers.Adam(lr=0.001)

####################################

@tf.function(experimental_relax_shapes=False)
def predict(model, x):
    out = model.train(x)
    out = tf.argmax(out, axis=-1)
    return out

@tf.function(experimental_relax_shapes=False)
def gradients(model, x, y):
    with tf.GradientTape() as tape:
        pred_logits = model.train(x)
        pred_label = tf.argmax(pred_logits, axis=-1)
        ################################################################################################################
        # loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=pred_logits)
        ################################################################################################################
        y_one_hot = tf.one_hot(y, depth=4)
        loss = tf.nn.weighted_cross_entropy_with_logits(labels=y_one_hot, logits=pred_logits, pos_weight=weight_tf)
        ################################################################################################################
        ################################################################################################################
    grad = tape.gradient(loss, params)
    return loss, grad, pred_label

####################################

def eval(labels, preds):
    assert (np.shape(labels) == np.shape(preds))
    assert (len(np.shape(labels)) == 4)
    tp = (labels == 1) * (preds == 1)
    fn = (labels == 1) * (preds == 0)
    tn = (labels == 0) * (preds == 0)
    fp = (labels == 0) * (preds == 1)
    correct = np.sum(tp, axis=(0,1,2))
    total = np.sum(labels, axis=(0,1,2))
    assert (np.all(correct <= total))
    return correct, total

####################################

batch_size = 10
for _ in range(20):
    _correct, _total = 0, 0
    for batch in range(0, len(x_train), batch_size):
        xs = x_train[batch:batch+batch_size]
        ys = y_train[batch:batch+batch_size]
        
        loss, grad, pred = gradients(model, xs, ys)
        optimizer.apply_gradients(zip(grad, params))

        label_np = tf.one_hot(ys, depth=4, axis=-1).numpy()
        pred_np = tf.one_hot(pred, depth=4, axis=-1).numpy()
        correct, total = eval(labels=label_np, preds=pred_np)

        _correct += correct
        _total += total
        
    accuracy = _correct / _total * 100
    print(accuracy)

####################################

colors = np.array([
[0, 0, 0],
[1, 0, 0],
[0, 1, 0],
[0, 0, 1]
])

batch_size = 10
for batch in range(0, len(x_test), batch_size):
    xs = x_test[batch:batch+batch_size]
    ys = y_test[batch:batch+batch_size]
    
    out = predict(model, xs)
    out_np = out.numpy()
    for i in range(batch_size):
        p = np.zeros(shape=(512, 512, 3))
        g = np.zeros(shape=(512, 512, 3))
        for c in range(4):
            p += (out_np[i, :, :] == c).reshape(512, 512, 1) * colors[c]
            g +=     (ys[i, :, :] == c).reshape(512, 512, 1) * colors[c]
        im = np.concatenate((p, g), axis=1)
        plt.imsave('%d.jpg' % (batch + i), im, dpi=300)

####################################








