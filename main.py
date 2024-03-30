import cv2

face1 = "opencv_face_detector.pbtxt"
face2 = "opencv_face_detector_uint8.pb"

age1 = "age_deploy.prototxt"
age2 = "age_net.caffemodel"

gender1 = "gender_deploy.prototxt"
gender2 = "gender_net.caffemodel"

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

face = cv2.dnn.readNet(face2, face1)
ages = cv2.dnn.readNet(age2, age1)
gender = cv2.dnn.readNet(gender2, gender1)

la = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', 
      '(25-32)', '(38-43)', '(48-53)', '(60-100)'] 
lg = ['Male', 'Female']

cap = cv2.VideoCapture(0)


while True:

    ret, img = cap.read()
    img = cv2.resize(img, (720, 640))

    print(img.shape)

    fr_cv = img.copy()

    fr_h = fr_cv.shape[0] 
    fr_w = fr_cv.shape[1] 

    blob = cv2.dnn.blobFromImage(fr_cv, 1.0, (300, 300), [104, 117, 123], True, False)
  
    face.setInput(blob) 
    detections = face.forward()
    faceBoxes = []

    for i in range(detections.shape[2]): 
      
        #Bounding box creation if confidence > 0.7

        confidence = detections[0, 0, i, 2]

        if confidence > 0.6:
            x1 = int(detections[0, 0, i, 3]*fr_w)
            y1 = int(detections[0, 0, i, 4]*fr_h)
            x2 = int(detections[0, 0, i, 5]*fr_w)
            y2 = int(detections[0, 0, i, 6]*fr_h)
          
            faceBoxes.append([x1, y1, x2, y2])
          
            cv2.rectangle(fr_cv, (x1, y1), (x2, y2), (0, 255, 255), int(round(fr_h/150)), 8)

    if not faceBoxes:
        print("No Face Detected")
    else:
        for faceBox in faceBoxes:

            extracted_face = fr_cv[max(0, faceBox[1] - 15): min(faceBox[3] + 15, fr_cv.shape[0] - 1),
                             max(0, faceBox[0] - 15):min(faceBox[2] + 15, fr_cv.shape[1] - 1)]

            blob = cv2.dnn.blobFromImage(extracted_face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

            ages.setInput(blob)
            age = ages.forward()
            age_pred = la[age[0].argmax()]

            gender.setInput(blob)
            g = gender.forward()
            gender_pred = lg[g[0].argmax()]
            print(gender_pred)

            cv2.putText(fr_cv, f'{gender_pred}{age_pred}', (faceBox[0] - 50, faceBox[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (217, 0, 0), 2, cv2.LINE_AA)





    cv2.imshow("img", fr_cv)
    cv2.waitKey(10)