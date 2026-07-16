"""
Shared OpenCV helpers: decode base64 frames from the browser, find the largest
face in a frame, and normalize it into the fixed-size grayscale square the
CNN expects.
"""
import base64
import cv2
import numpy as np

IMG_SIZE = 100  # CNN input is IMG_SIZE x IMG_SIZE x 1

_face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def decode_base64_image(data_url):
    """Turns a 'data:image/jpeg;base64,...' string from <canvas>.toDataURL() into a BGR numpy array."""
    header, encoded = data_url.split(",", 1)
    binary = base64.b64decode(encoded)
    arr = np.frombuffer(binary, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img


def detect_largest_face(bgr_image):
    """Returns (x, y, w, h) of the largest detected face, or None."""
    gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    faces = _face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=6, minSize=(80, 80)
    )
    if len(faces) == 0:
        return None
    # pick the largest bounding box (closest to camera)
    faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
    return faces[0]


def crop_and_normalize_face(bgr_image, box):
    x, y, w, h = box
    face = bgr_image[y:y + h, x:x + w]
    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    return resized  # uint8, shape (IMG_SIZE, IMG_SIZE)


def extract_face_from_dataurl(data_url):
    """Full pipeline: dataURL -> normalized face array, or None if no face found."""
    img = decode_base64_image(data_url)
    box = detect_largest_face(img)
    if box is None:
        return None, None
    face = crop_and_normalize_face(img, box)
    return face, box
