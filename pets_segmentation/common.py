import os
import settings
from tensorflow.keras import layers
from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img
import numpy as np
import random
import PIL
from PIL import ImageOps

def load_paths():
    """load list of paths"""
    settings.input_img_paths = sorted(
        [
            os.path.join(settings.input_dir, fname)
            for fname in os.listdir(settings.input_dir)
            if fname.endswith(".jpg")
        ]
    )
    settings.target_img_paths = sorted(
        [
            os.path.join(settings.target_dir, fname)
            for fname in os.listdir(settings.target_dir)
            if fname.endswith(".png") and not fname.startswith(".")
        ]
    )
    print("Number of samples:", len(settings.input_img_paths))
    for input_path, target_path in zip(settings.input_img_paths[:10], settings.target_img_paths[:10]):
        print(input_path, "|", target_path)


class OxfordPets(keras.utils.Sequence):
    """Helper to iterate over the data (as Numpy arrays)."""

    def __init__(self, batch_size, img_size, input_img_paths, target_img_paths):
        self.batch_size = batch_size
        self.img_size = img_size
        self.input_img_paths = input_img_paths
        self.target_img_paths = target_img_paths

    def __len__(self):
        return len(self.target_img_paths) // self.batch_size

    def __getitem__(self, idx):
        """Returns tuple (input, target) correspond to batch #idx."""
        i = idx * self.batch_size
        batch_input_img_paths = self.input_img_paths[i : i + self.batch_size]
        batch_target_img_paths = self.target_img_paths[i : i + self.batch_size]
        x = np.zeros((self.batch_size,) + self.img_size + (3,), dtype="float32")
        for j, path in enumerate(batch_input_img_paths):
            img = load_img(path, target_size=self.img_size)
            x[j] = img
        y = np.zeros((self.batch_size,) + self.img_size + (1,), dtype="uint8")
        for j, path in enumerate(batch_target_img_paths):
            img = load_img(path, target_size=self.img_size, color_mode="grayscale")
            y[j] = np.expand_dims(img, 2)
        return x, y


def get_model(img_size, num_classes):
    """Perpare U-Net Xception-style model"""
    inputs = keras.Input(shape=img_size + (3,))

    ### [First half of the network: downsampling inputs] ###

    # Entry block
    x = layers.Conv2D(32, 3, strides=2, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    # Blocks 1, 2, 3 are identical apart from the feature depth.
    for filters in [64, 128, 256]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(filters, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    ### [Second half of the network: upsampling inputs] ###

    for filters in [256, 128, 64, 32]:
        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.UpSampling2D(2)(x)

        # Project residual
        residual = layers.UpSampling2D(2)(previous_block_activation)
        residual = layers.Conv2D(filters, 1, padding="same")(residual)
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    # Add a per-pixel classification layer
    outputs = layers.Conv2D(num_classes, 3, activation="softmax", padding="same")(x)

    # Define the model
    model = keras.Model(inputs, outputs)
    return model

def get_dataset(val_samples=1000, seed=1337):
    random.Random(seed).shuffle(settings.input_img_paths)
    random.Random(seed).shuffle(settings.target_img_paths)
    train_input_img_paths = settings.input_img_paths[:-val_samples]
    train_target_img_paths = settings.target_img_paths[:-val_samples]
    val_input_img_paths = settings.input_img_paths[-val_samples:]
    val_target_img_paths = settings.target_img_paths[-val_samples:]

    # Instantiate data Sequences for each split
    return (
        OxfordPets(settings.batch_size, settings.img_size, train_input_img_paths, train_target_img_paths), 
        OxfordPets(settings.batch_size, settings.img_size, val_input_img_paths, val_target_img_paths))

def mask_to_img(pred_res):
    """Quick utility to display a model's prediction."""
    mask = np.argmax(pred_res, axis=-1)
    mask = np.expand_dims(mask, axis=-1)
    return ImageOps.autocontrast(keras.preprocessing.image.array_to_img(mask))

def to_rgb(im):
    # I think this will be slow
    im = im.reshape(im.shape[0], im.shape[1])
    w,h = im.shape
    ret = np.empty((w,h, 3), dtype=np.uint8)
    ret[:, :, 0] = im
    ret[:, :, 1] = im
    ret[:, :, 2] = im
    return ret