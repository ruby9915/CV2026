import sys
import cv2
import numpy as np
import mediapipe as mp

mp_selfie_segmentation = mp.solutions.selfie_segmentation

BLUR_KSIZE = 55
THRESHOLD = 0.5
OUTPUT = "selfie.jpg"


def blur_background(frame, mask):
    condition = np.stack((mask,) * 3, axis=-1) > THRESHOLD
    background = cv2.GaussianBlur(frame, (BLUR_KSIZE, BLUR_KSIZE), 0)
    return np.where(condition, frame, background)


def run_image(path):
    image = cv2.imread(path)
    if image is None:
        raise IOError("cannot read " + path)
    with mp_selfie_segmentation.SelfieSegmentation(model_selection=0) as segmenter:
        results = segmenter.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    output = blur_background(image, results.segmentation_mask)
    cv2.imwrite(OUTPUT, output)
    print("saved", OUTPUT)
    cv2.imshow("Selfie Segmentation", np.hstack((image, output)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def run_camera(index=0):
    cap = cv2.VideoCapture(index)
    with mp_selfie_segmentation.SelfieSegmentation(model_selection=0) as segmenter:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                continue
            frame = cv2.flip(frame, 1)
            results = segmenter.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            output = blur_background(frame, results.segmentation_mask)
            cv2.imshow("Selfie Segmentation  (s: save, q: quit)", output)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("s"):
                cv2.imwrite(OUTPUT, output)
                print("saved", OUTPUT)
            if key in (27, ord("q")):
                break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        run_image(sys.argv[1])
    else:
        run_camera(0)
