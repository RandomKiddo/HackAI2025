"""
TEST FILE

THIS FILE HOLDS MODEL VERSION USED AS MODELS IN optimize_lr.py.
"""

import tensorflow as tf

img_height = img_width = 512

class HackAIModel(tf.keras.Model):
    def __init__(self):
        super(HackAIModel, self).__init__()

        self.net = tf.keras.applications.MobileNetV3Small(
            input_shape=(img_height, img_width, 3),
            include_top=False,
            weights='imagenet'
        )
        self.net.trainable = False

        self.flatten = tf.keras.layers.Flatten()
        self.regression1 = tf.keras.layers.Dense(32, activation='relu')
        self.regression2 = tf.keras.layers.Dense(16, activation='relu')

        self.left_head1 = tf.keras.layers.Dense(8, activation='relu')
        self.left_head2 = tf.keras.layers.Dense(1, activation='relu', name='distance')

        self.middle_head1 = tf.keras.layers.Dense(8, activation='relu')
        self.middle_head2 = tf.keras.layers.Dense(1, activation='relu', name='x')

        self.right_head = tf.keras.layers.Dense(1, activation='relu', name='y')
    
    def call(self, inputs):
        x = self.net(inputs)
        x = self.flatten(x)
        x = self.regression1(x)
        x = self.regression2(x)

        left_head = self.left_head1(x)
        left_head = self.left_head2(left_head)

        middle_head = self.middle_head1(x)
        middle_head = self.middle_head2(middle_head)

        right_head = self.right_head(x)

        return left_head, middle_head, right_head
    

class HeatmapModel(tf.keras.Model):
    def __init__(self):
        super(HeatmapModel, self).__init__()

        self.net = tf.keras.applications.MobileNetV3Small(
            input_shape=(img_height, img_width, 3),
            include_top=False,
            weights='imagenet'
        )
        self.net.trainable = False

        self.conv1 = tf.keras.layers.Conv2D(64, (5, 5), activation='swish', padding='same')
        self.conv2 = tf.keras.layers.Conv2D(64, (3, 3), activation='swish', padding='same')
        self.conv3 = tf.keras.layers.Conv2D(32, (3, 3), activation='swish', padding='same')
        self.conv4 = tf.keras.layers.Conv2D(128, (1, 1), activation='swish')
        self.pool = tf.keras.layers.GlobalMaxPooling2D()

        self.flatten = tf.keras.layers.Flatten()
        self.regression = tf.keras.layers.Dense(16, activation='swish')

        self.left_head1 = tf.keras.layers.Dropout(0.3)
        self.left_head2 = tf.keras.layers.Dense(8, activation='swish')
        self.left_head3 = tf.keras.layers.Dense(4, activation='swish')
        self.left_head4 = tf.keras.layers.Dense(1, activation='swish', name='distance')

        self.middle_head1 = tf.keras.layers.Dense(8, activation='swish')
        self.middle_head2 = tf.keras.layers.Dense(512, activation='swish')
        self.middle_head3 = tf.keras.layers.Dense(512, activation='linear')
        self.middle_head4 = tf.keras.layers.Softmax(name='x')

        self.right_head1 = tf.keras.layers.Dropout(0.3)
        self.right_head2 = tf.keras.layers.Dense(8, activation='swish')
        self.right_head3 = tf.keras.layers.Dense(512, activation='swish')
        self.right_head4 = tf.keras.layers.Dense(512, activation='linear')
        self.right_head5 = tf.keras.layers.Softmax(name='y')
    
    def call(self, inputs):
        x = self.net(inputs)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.regression(x)

        left_head = self.left_head1(x)
        left_head = self.left_head2(left_head)
        left_head = self.left_head3(left_head)
        left_head = self.left_head4(left_head)

        middle_head = self.middle_head1(x)
        middle_head = self.middle_head2(middle_head)
        middle_head = self.middle_head3(middle_head)
        middle_head = self.middle_head4(middle_head)

        right_head = self.right_head1(x)
        right_head = self.right_head2(right_head)
        right_head = self.right_head3(right_head)
        right_head = self.right_head4(right_head)
        right_head = self.right_head5(right_head)

        return left_head, middle_head, right_head