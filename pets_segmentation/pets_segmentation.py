import common
import settings
from tensorflow.keras.preprocessing.image import load_img, img_to_array, array_to_img
from tensorflow.keras.models import load_model
import numpy as np
import cv2

# load model
model = load_model('oxford_segmentation.h5')
cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    frame = cv2.resize(frame, settings.img_size)
    val_preds = model.predict(frame[np.newaxis, :])
    mask = common.to_rgb(img_to_array(common.mask_to_img(val_preds[0])))
    #join frame and mask
    frame_mask = np.concatenate((frame, mask), axis=1)

    cv2.imshow('frame_mask', frame_mask)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()