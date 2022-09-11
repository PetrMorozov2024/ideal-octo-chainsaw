from imageai.Detection import ObjectDetection
import os

path = os.getcwd()

detector = ObjectDetection()

detector.setModelTypeAsYOLOv3()
detector.setModelPath(os.path.join(path, "yolo.h5"))
detector.loadModel()
detections = detector.detectObjectsFromImage(input_image=(os.path.join(path, "jiraf.jpg")),
output_image_path =os.path.join(path, "photo3.jpg"), minimum_percentage_probability=30)

for eachObject in detections:
    print(eachObject['name'], ":", eachObject['percentage_probability'], ":", eachObject["box_points"])
    print('-----------------------------------')