import tensorflow as tf

x = tf.ones((2, 2))

with tf.GradientTape() as t:
    t.watch(x)
    y = tf.reduce_sum(x)
    z = tf.square(y)

dz_dx = t.gradient(z, x)
print(dz_dx)

x = tf.constant(3.0)
# Notice that persistent is False by default
with tf.GradientTape() as t:
    t.watch(x)
    # y = x^2
    y = x * x
    # z = y^2
    z = y * y
# Compute dz/dx. 4 * x^3 at x = 3 --> 108.0
dz_dx = t.gradient(z, x)
print(dz_dx)

# If you try to compute dy/dx after the gradient tape has expired:
try:
    dy_dx = t.gradient(y, x)  # 6.0
    print(dy_dx)
except RuntimeError as e:
    print("The error message you get is:")
    print(e)

x = tf.constant(3.0)
# Set persistent=True so that you can reuse the tape
with tf.GradientTape(persistent=True) as t:
    t.watch(x)
    # y = x^2
    y = x * x
    # z = y^2
    z = y * y
# Compute dz/dx. 4 * x^3 at x = 3 --> 108.0
dz_dx = t.gradient(z, x)
print(dz_dx)
dy_dx = t.gradient(y, x)  # 6.0
print(dy_dx)
del t

x = tf.Variable(1.0)
with tf.GradientTape() as tape_2:
    with tf.GradientTape() as tape_1:
        y = x * x * x
    # The first gradient calculation should occur at least
    # within the outer with block
    dy_dx = tape_1.gradient(y, x)
d2y_dx2 = tape_2.gradient(dy_dx, x)

print(dy_dx)
print(d2y_dx2)

x = tf.Variable(1.0)
with tf.GradientTape() as tape_2:
    with tf.GradientTape() as tape_1:
        y = x * x * x
# The first gradient call is outside the outer with block
# so the tape will expire after this
dy_dx = tape_1.gradient(y, x)
# The tape is now expired and the gradient output will be `None`
d2y_dx2 = tape_2.gradient(dy_dx, x)
print(dy_dx)
print(d2y_dx2)

x = tf.Variable(1.0)
# Setting persistent=True still won't work
with tf.GradientTape(persistent=True) as tape_2:
    # Setting persistent=True still won't work
    with tf.GradientTape(persistent=True) as tape_1:
        y = x * x * x
# The first gradient call is outside the outer with block
# so the tape will expire after this
dy_dx = tape_1.gradient(y, x)
# the output will be `None`
d2y_dx2 = tape_2.gradient(dy_dx, x)
print(dy_dx)
print(d2y_dx2)

x = tf.Variable(1.0)
with tf.GradientTape() as tape_2:
    with tf.GradientTape() as tape_1:
        y = x * x * x
        dy_dx = tape_1.gradient(y, x)
        # this is acceptable
        d2y_dx2 = tape_2.gradient(dy_dx, x)

print(dy_dx)
print(d2y_dx2)