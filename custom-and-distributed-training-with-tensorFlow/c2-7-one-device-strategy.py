import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds

tfds.disable_progress_bar()

# choose a device type such as CPU or GPU
devices = tf.config.list_physical_devices('GPU')
print(devices[0])
# You'll see that the name will look something like "/physical_device:GPU:0"
# Just take the GPU:0 part and use that as the name
gpu_name = "GPU:0"
# define the strategy and pass in the device name
one_strategy = tf.distribute.OneDeviceStrategy(device=gpu_name)

pixels = 224
MODULE_HANDLE = 'https://tfhub.dev/tensorflow/resnet_50/feature_vector/1'
IMAGE_SIZE = (pixels, pixels)
BATCH_SIZE = 32
print("Using {} with input size {}".format(MODULE_HANDLE, IMAGE_SIZE))

splits = ['train[:80%]', 'train[80%:90%]', 'train[90%:]']
(train_examples, validation_examples, test_examples), info = tfds.load('cats_vs_dogs', with_info=True,
                                                                       as_supervised=True, split=splits)
num_examples = info.splits['train'].num_examples
num_classes = info.features['label'].num_classes


def format_image(image, label):
    image = tf.image.resize(image, IMAGE_SIZE) / 255.0
    return image, label


train_batches = train_examples.shuffle(num_examples // 4).map(format_image).batch(BATCH_SIZE).prefetch(1)
validation_batches = validation_examples.map(format_image).batch(BATCH_SIZE).prefetch(1)
test_batches = test_examples.map(format_image).batch(1)

for image_batch, label_batch in train_batches.take(1):
    pass

print(image_batch.shape)

do_fine_tuning = False


def build_and_compile_model():
    print("Building model with", MODULE_HANDLE)
    # configures the feature extractor fetched from TF Hub
    feature_extractor = hub.KerasLayer(MODULE_HANDLE,
                                       input_shape=IMAGE_SIZE + (3,),
                                       trainable=do_fine_tuning)
    # define the model
    model = tf.keras.Sequential([
        feature_extractor,
        # append a dense with softmax for the number of classes
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    # display summary
    print(model.summary())
    # configure the optimizer, loss and metrics
    optimizer = tf.keras.optimizers.SGD(lr=0.002, momentum=0.9) if do_fine_tuning else 'adam'
    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

with one_strategy.scope():
    model = build_and_compile_model()

EPOCHS = 5
hist = model.fit(train_batches,
                 epochs=EPOCHS,
                 validation_data=validation_batches)
