import common
import settings
from tensorflow.keras.preprocessing.image import load_img, img_to_array, array_to_img
import PIL
from PIL import ImageOps
from tensorflow import keras
from tensorflow.keras.models import load_model
import numpy as np

# common.load_paths()

# Display input image #9
# Display auto-contrast version of corresponding target (per-pixel categories)
# img = ImageOps.autocontrast(PIL.Image.open(settings.target_img_paths[9]))
# img.show()


# Free up RAM in case the model definition cells were run multiple times
# keras.backend.clear_session()

# Build model
# model = common.get_model(settings.img_size, settings.num_classes)
# model.summary()
# model.compile(optimizer="rmsprop", loss="sparse_categorical_crossentropy")

# train_gen, val_gen = common.get_dataset()
# callbacks = [
# #    keras.callbacks.ModelCheckpoint("oxford_segmentation.h5", save_best_only=True)
# ]

# Train the model, doing validation at the end of each epoch.
# epochs = 15
# model.fit(train_gen, epochs=epochs, validation_data=val_gen, callbacks=callbacks)

# load model
model = load_model('oxford_segmentation.h5')
frame = img_to_array(load_img('./images/Abyssinian_109.jpg', target_size=settings.img_size))
print(frame.shape)
val_preds = model.predict(frame[np.newaxis, :])

mask = img_to_array(common.mask_to_img(val_preds[0]))
mask = common.to_rgb(mask)
print(frame.shape, mask.shape)
array_to_img(np.concatenate((frame, mask), axis=1)).show()


# array_to_img(frame).show()
# np.concatenate((frame, b.T), axis=1)
# i = 10

# # Display input image
# PIL.Image.open(filename=val_input_img_paths[i]).show()

# # Display ground-truth target mask
# PIL.ImageOps.autocontrast(load_img(val_target_img_paths[i])).show()

# # Display mask predicted by our model
# common.mask_to_img(val_preds[0]).show()
