import cv2
import mediapipe as mp
mp_obj = mp.solutions.objectron
mp_drawing = mp.solutions.drawing_utils

#setup video capture
cap = cv2.VideoCapture(0)

#Use objectron to detect object in the video stream

with mp_obj.Objectron(static_image_mode = False,
                      max_num_objects = 5,
                      min_detection_confidence = 0.5,
                      model_name = 'Cup') as objecttron:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # Convert the image to RGB
        image = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        res = objecttron.process(image)

        #Draw the image color to RGB
        if res.detected_objects:
            for detected_object in res.detected_objects:
                mp_drawing.draw_landmarks(frame,
                                          detected_object.landmarks_2d,
                                          mp_obj.BOX_CONNECTIONS)
                mp_drawing.draw_axis(frame , detected_object.rotation , detected_object.translation)
        # Display the output
        cv2.imshow("MediaPipe Objectron" ,frame)

        if cv2.waitKey(5) & 0xFF == 27:
            break
cap.release()
cv2.destroyAllWindows()