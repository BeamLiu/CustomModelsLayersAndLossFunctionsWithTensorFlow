import keras
import tensorflow as tf
import tensorflow_datasets as tfds


class IdentityBlock(keras.Model):
    def __init__(self, filters, kernel_size):
        super(IdentityBlock, self).__init__(name='')
        self.conv1 = keras.layers.Conv2D(filters, kernel_size, padding='same')
        self.bn1 = keras.layers.BatchNormalization()

        self.conv2 = keras.layers.Conv2D(filters, kernel_size, padding='same')
        self.bn2 = keras.layers.BatchNormalization()

        self.act = keras.layers.Activation('relu')
        self.add = keras.layers.Add()

    def call(self, input_tensor):
        x = self.conv1(input_tensor)
        x = self.bn1(x)
        x = self.act(x)

        x = self.conv2(x)
        x = self.bn2(x)

        x = self.add([x, input_tensor])
        x = self.act(x)
        return x


class ResNet(keras.Model):
    def __init__(self, num_classes):
        super(ResNet, self).__init__()
        self.conv = keras.layers.Conv2D(64, 7, padding='same')
        self.bn = keras.layers.BatchNormalization()
        self.act = keras.layers.Activation('relu')
        self.max_pool = keras.layers.MaxPool2D((3, 3))

        # Use the Identity blocks that you just defined
        self.id1a = IdentityBlock(64, 3)
        self.id1b = IdentityBlock(64, 3)

        self.global_pool = keras.layers.GlobalAveragePooling2D()
        self.classifier = keras.layers.Dense(num_classes, activation='softmax')

    def call(self, inputs):
        x = self.conv(inputs)
        x = self.bn(x)
        x = self.act(x)
        x = self.max_pool(x)

        # insert the identity blocks in the middle of the network
        x = self.id1a(x)
        x = self.id1b(x)

        x = self.global_pool(x)
        return self.classifier(x)

# utility function to normalize the images and return (image, label) pairs.
def preprocess(features):
    return tf.cast(features['image'], tf.float32) / 255., features['label']

# create a ResNet instance with 10 output units for MNIST
resnet = ResNet(10)
print(resnet.summary())
# plot_model(resnet, show_shapes=True, show_layer_names=True, expand_nested=True, to_file='c1-6-resnet.png')
resnet.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# load and preprocess the dataset
dataset = tfds.load('mnist', split=tfds.Split.TRAIN, data_dir='./dataset')
dataset = dataset.map(preprocess).batch(32)

# train the model.
resnet.fit(dataset, epochs=1)

