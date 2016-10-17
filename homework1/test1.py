import tensorflow as tf

# step(x) = { 1 if x > 0; -1 otherwise }
def step(x):
    is_greater = tf.greater(x, 0)
    as_float = tf.to_float(is_greater)
    doubled = tf.mul(as_float, 2)
    return tf.sub(doubled, 1)

T, F = 1., -1.
bias = 1.
train_in = [
    [T, T, bias],
    [T, F, bias],
    [F, T, bias],
    [F, F, bias],
]
train_out = [
    [T],
    [F],
    [F],
    [F],
]

w = tf.Variable(tf.random_normal([3, 1]))

output = step(tf.matmul(train_in, w))
error = tf.sub(train_out, output)
mse = tf.reduce_mean(tf.square(error))

delta = tf.matmul(train_in, error, transpose_a=True)
train = tf.assign(w, tf.add(w, delta))

sess = tf.Session()
sess.run(tf.initialize_all_variables())


err, target = 1, 0
epoch, max_epochs = 0, 10
while err > target and epoch < max_epochs:
    epoch += 1
    err, _ = sess.run([mse, train])
    print('epoch:', epoch, 'mse:', err)