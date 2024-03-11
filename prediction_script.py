from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt



# Load a model
model = YOLO("./runs/detect/train2/weights/best.pt")  

input_img= 'image.jpeg'

results =  model.predict(input_img) # return a list of Results objects
result = results[0]

boxes = result.boxes  # Boxes object for bbox outputs

output_img = cv2.imread(input_img)

for boxe in boxes:
    boxe_coordinates = boxe.xyxy[0].tolist()
    boxe_coordinates = [round(x) for x in boxe_coordinates]

    boxe_class_number = boxe.cls[0].item()
    boxe_class_name = result.names[boxe_class_number]
    boxe_probability = boxe.conf[0].item()

    print("Object Class:",boxe_class_name )
    print("Object Class Number:",boxe_class_number )
    print("Coordinates:", boxe_coordinates)
    print("Probability:", boxe_probability)

    (startX, startY, endX, endY) = boxe_coordinates
    cv2.rectangle(output_img, (startX, startY),   (endX, endY), (0, 255, 0), 2)
    y = startY - 10 if startY - 10 > 10 else startY + 10
    cv2.putText(output_img,boxe_class_name, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0), 2)

    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Class probabilities for classification outputs




plt.imshow(output_img)
plt.axis('off')
plt.show()