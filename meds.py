import face_recognition as fp
import cv2
import numpy as np
from cvzone import FPS
#may 2021
#scope-detection box change

cap = cv2.VideoCapture(0)
fpsReader = FPS()

person1_image = fp.load_image_file("musk.jpg")
person1_face_encoding = fp.face_encodings(person1_image)[0]
person2_image = fp.load_image_file("dev.jpg")
person2_face_encoding = fp.face_encodings(person2_image)[0]

kn_encod = [person1_face_encoding,person2_face_encoding]
kn_names = ["Musk","Devesh"]

f_loc = []
f_en = []
names = []
process_bool = True

while True:

    ret, img = cap.read()
    fps, img = fpsReader.update(img)

    if process_bool:
        small_img = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)

        rgb_small_img = np.ascontiguousarray(small_img[:, :, ::-1])

        f_loc = fp.face_locations(rgb_small_img)#[(44, 118, 95, 66)]
        f_en = fp.face_encodings(rgb_small_img, f_loc)#[array([-0.02903933,  0.13830945,  0.03686822, -0.09154546, -0.06150129,])],,,,,list
        print(type(f_en))

        face_names = []
        #recognition
        for encoding in f_en:
            matches = fp.compare_faces(kn_encod, encoding)
            name = "new_person"
            dist = fp.face_distance(kn_encod, encoding)
            best = np.argmin(dist)
            if matches[best]:
                name = kn_names[best]
            names.append(name)

    process_bool = not process_bool

    for (up, s1, down, s2), name in zip(f_loc, names):
        up *= 4
        s1 *= 4
        down *= 4
        s2 *= 4
        cv2.rectangle(img, (s2, up), (s1, down), (255, 255, 255), 2)
        cv2.rectangle(img, (s2, up - 35), (s1,up), (255, 255, 255), cv2.FILLED)
        cv2.putText(img, name, (s2 + 6, up - 6), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0,0), 2)

    cv2.imshow('output', img)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
