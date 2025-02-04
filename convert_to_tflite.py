import tensorflow as tf
from net import MobileNet

def convert_to_tflite():
    try:
        # Create model with explicit input shape
        base_model = MobileNet()
        # Load the weights first
        base_model.load_weights('models/model_split.h5')
        
        # Create a new model with explicit input shape
        inputs = tf.keras.Input(shape=(224, 224, 3))
        outputs = base_model(inputs)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)

        # Convert the model to TFLite format
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        
        # Optimize for latency
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        # Quantize the model to reduce size and improve speed
        converter.target_spec.supported_types = [tf.float16]
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS
        ]

        # Convert the model
        tflite_model = converter.convert()

        # Save the TFLite model
        with open('models/model_split.tflite', 'wb') as f:
            f.write(tflite_model)
        
        print("Model converted and saved as model_split.tflite")
    
    except Exception as e:
        print(f"Error during conversion: {str(e)}")

if __name__ == '__main__':
    convert_to_tflite() 