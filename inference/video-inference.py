import numpy as np
import cv2
import tensorflow as tf
from keras.preprocessing.image import img_to_array
from keras.utils import custom_object_scope


def combined_loss(y_true, y_pred):
    ssim_loss = 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, 1.0))
    return ssim_loss


# Load the model within a custom object scope
with custom_object_scope({'combined_loss': combined_loss}):
    correction_model = tf.keras.models.load_model('correction-finalv7.h5')
detection_model = tf.keras.models.load_model('detectionv2.h5')


def preprocess_frame(frame):
    img = cv2.resize(frame, (256, 256))
    img = img_to_array(img) / 255.0  # Normalize pixel values to [0, 1]
    return img


def detect_pixelation(image):
    prediction = detection_model.predict(image)
    return prediction[0][0] > 0.5


def correct_image(image):
    corrected_image = correction_model.predict(image)
    corrected_image = np.clip(corrected_image[0], 0, 1)  # Clip values to [0, 1]
    return corrected_image


def pad_image(image, target_shape=(256, 256, 3)):
    padding = [(0, max(0, target_shape[i] - image.shape[i])) for i in range(3)]
    return np.pad(image, padding, mode='constant', constant_values=0)


def process_frame_blocks(frame):
    input_image = preprocess_frame(frame)
    h, w, _ = input_image.shape

    output_image = np.zeros((h, w, 3))

    for y in range(0, h, 256):
        for x in range(0, w, 256):
            block = input_image[y:y+256, x:x+256]

            padded_block = pad_image(block)
            padded_block = np.expand_dims(padded_block, axis=0)

            if detect_pixelation(padded_block):
                corrected_block = correct_image(padded_block)
            else:
                corrected_block = block

            block_height = min(256, h - y)
            block_width = min(256, w - x)

            output_image[y:y+block_height, x:x+block_width] = corrected_block[:block_height, :block_width]

    return output_image


def main():
    cap = cv2.VideoCapture(0)  # Open the camera stream

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        corrected_frame = process_frame_blocks(frame)
        corrected_frame = (corrected_frame * 255).astype(np.uint8)

        combined_frame = np.hstack((cv2.resize(frame, (512, 512)), cv2.resize(corrected_frame, (512, 512))))

        cv2.imshow('Original and Corrected Frames', combined_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
