import tensorflow as tf
from tensorflow.keras import layers, Model

class MobileNet(Model):
    def __init__(self):
        super(MobileNet, self).__init__()
        
        # Create the base model
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=(224, 224, 3),
            include_top=False,
            weights='imagenet'
        )
        base_model.trainable = False

        # Create the full model
        inputs = layers.Input(shape=(224, 224, 3))
        x = base_model(inputs, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.2)(x)
        outputs = layers.Dense(2, activation='softmax')(x)
        
        # Build the model in a way that matches the saved weights
        self.model = Model(inputs, outputs)

    def call(self, inputs):
        return self.model(inputs)

    def load_weights(self, checkpoint):
        self.model.load_weights(checkpoint)

    def predict(self, inputs):
        return self.model.predict(inputs)
