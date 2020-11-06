# AICamp Deep Learning for Computer Vision (Cohort #5) - Capstone Assignment
# Yossarian King / October 2020

# JumpNet test program.


import tensorflow as tf
import tensorflow.keras.layers as layers

from jumpnet import JumpNet

jumpnet = (
	JumpNet(shape=(32, 32, 3))
	.stem()
	.group(filters=32, blocks=3, blockfilters=16)
	.group(filters=64, blocks=4, blockfilters=32)
	.group(filters=128, blocks=3, blockfilters=64)
	.classifier(classes=10)
)

jumpnet.model.summary()
jumpnet.model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()

history = jumpnet.model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))
test_loss, test_acc = jumpnet.model.evaluate(test_images,  test_labels, verbose=2)

print("test accuracy", test_acc)
print("test loss", test_loss)
