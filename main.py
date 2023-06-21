import cv2
import tensorflow as tf
import utils

def main():
    face_detection_model = utils.FaceDetector()
    mask_detection_model = utils.MaskDetector()
    vid = cv2.VideoCapture(0)

    while True:
        ret, frame = vid.read()
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = face_detection_model.detect(gray)

            if len(faces):
                x, y, w, h = faces[0]
                color = utils.Colors.RED
                has_mask = mask_detection_model.detect(frame)

                if has_mask:
                    color = utils.Colors.GREEN

                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

            cv2.imshow("Mask Detection", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            break

    vid.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()