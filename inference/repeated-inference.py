import numpy as np
import cv2
import tensorflow as tf
from keras.preprocessing.image import load_img, img_to_array
from keras.utils import custom_object_scope


def combined_loss(y_true, y_pred):
    ssim_loss = 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, 1.0))
    return ssim_loss


# Load the model within a custom object scope
with custom_object_scope({'combined_loss': combined_loss}):
    correction_model = tf.keras.models.load_model('correction-finalv7.h5')
detection_model = tf.keras.models.load_model('detectionv2.h5')


def load_and_preprocess_image(file_path):
    img = load_img(file_path)
    img = img_to_array(img) / 255.0  # Normalize pixel values to [0, 1]
    return img


def detect_pixelation(image):
    prediction = detection_model.predict(image)
    return prediction[0][0] > 0.5


def correct_image(image):
    corrected_image = correction_model.predict(image)
    corrected_image = np.clip(corrected_image, 0, 1)  # Clip values to [0, 1]
    return corrected_image[0]


def pad_image(image, target_shape=(256, 256, 3)):
    # Calculate padding
    padding = [(0, max(0, target_shape[i] - image.shape[i])) for i in range(3)]
    return np.pad(image, padding, mode='constant', constant_values=0)


def process_image_blocks(input_image_path):
    # Load input image
    input_image = load_and_preprocess_image(input_image_path)
    h, w, _ = input_image.shape

    # Prepare an output image of the same size
    output_image = np.zeros((h, w, 3))

    # Loop over the input image in 256x256 blocks
    for y in range(0, h, 256):
        for x in range(0, w, 256):
            block = input_image[y:y+256, x:x+256]

            # Pad block to ensure it is (256, 256, 3)
            padded_block = pad_image(block)
            padded_block = np.expand_dims(padded_block, axis=0)  # Shape: (1, 256, 256, 3)

            # Detect and correct pixelation in a loop
            while detect_pixelation(padded_block):
                print(f"Pixelated image detected at block ({y},{x})")
                # Correct pixelated block
                corrected_block = correct_image(padded_block)
                padded_block = np.expand_dims(corrected_block, axis=0)

            block_height = min(256, h - y)
            block_width = min(256, w - x)
            output_image[y:y+block_height, x:x+block_width] = padded_block[0][:block_height, :block_width]

    return output_image


# Example usage
output_image = process_image_blocks(input("Enter image path: "))

# Convert output image to proper format and save
output_image = (output_image * 255).astype(np.uint8)
cv2.imwrite('output.jpg', cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR))
