
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

####################################

model = model(layers=[

conv_block((3,3,1,64)),
max_pool(2),

conv_block((3,3,64,128)),
max_pool(2),

conv_block((3,3,128,128)),
max_pool(2),

conv_block((3,3,128,128)),
up_pool(2),

conv_block((3,3,128,128)),
up_pool(2),

conv_block((3,3,128,64)),
up_pool(2),

conv_block((3,3,64,4))
])

####################################

params = model.get_params()
optimizer = tf.keras.optimizers.Adam(lr=0.01)

####################################

def pred(model, x):
    return model.train(x)

@tf.function(experimental_relax_shapes=False)
def gradients(model, x, y):
    with tf.GradientTape() as tape:
        pred_logits = model.train(x)
        pred_label = tf.argmax(pred_logits, axis=-1)
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=pred_logits)
    grad = tape.gradient(loss, params)
    return loss, grad, pred_label

####################################

def eval(labels, preds):
    assert (np.shape(labels) == np.shape(preds))
    assert (len(np.shape(labels)) == 4)
    correct = (labels == preds) * labels
    correct = np.sum(correct, axis=(0,1,2))
    total = np.sum(labels, axis=(0,1,2))
    assert (np.all(correct <= total))
    return correct, total

####################################

batch_size = 5
for _ in range(10):
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











