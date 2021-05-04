
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
        correct = tf.reduce_sum(tf.cast(tf.equal(pred_label, y), tf.float32))
        # total = tf.math.reduce_prod(tf.shape(y))
        # acc = correct / total
    
    grad = tape.gradient(loss, params)
    return loss, correct, grad

####################################

batch_size = 5
for _ in range(10):
    total_correct = 0
    for batch in range(0, len(x_train), batch_size):
        xs = x_train[batch:batch+batch_size].astype(np.float32)
        ys = y_train[batch:batch+batch_size].astype(np.int64)
        loss, correct, grad = gradients(model, xs, ys)
        optimizer.apply_gradients(zip(grad, params))
        total_correct += correct

        # out = pred(model, xs)
        # tf.print(tf.shape(out))
        # tf.print(tf.shape(xs))
        # tf.print(tf.shape(ys))

    print (total_correct / len(x_train) / (512 * 512) * 100)

####################################











