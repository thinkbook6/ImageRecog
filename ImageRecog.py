import io
import os
from datetime import datetime

import numpy

import cv2
from google.cloud import vision_v1p3beta1 as vision

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'G:/Docs/Python/ImageRecok/key/FruitRecogProject-key.json'

SOURCE_PATH = "G:/Docs/Python/ImageRecok/"

CATEGORY = 'Fruit'

def load_food_category(CATEGORY):
    names = [line.rstrip('\n').lower() for line in open('food_dictionary/' + CATEGORY + '.food_dictionary')]
    return names


def recognize_food(img_path, list_foods):
    
    start_time = datetime.now()

    # Read image with opencv
    img = cv2.imread(img_path)

    # Get image size
    height, width = img.shape[:2]

    #scale Image

    img = cv2.resize(img, (800, int((height * 800) / width)))

    #Save image to temp file

    cv2.imwrite(SOURCE_PATH + 'output.jpg', img)

    img_path = SOURCE_PATH + 'output.jpg'

    #create google vision client

    client = vision.ImageAnnotatorClient()

    #Read image file

    with io.open(img_path, 'rb') as image_file:
    	content = image_file.read()

    image = vision.types.Image(content=content)

    #recognize Content

    response = client.label_detection(image=image)

    labels = response.label_annotations

    for label in labels:
        # if len(text.description) == 10:
        desc = label.description.lower()
        score = round(label.score, 2)
        print("label: ", desc, "  score: ", score)
        if (desc in list_foods):
            # score = round(label.score, 3)
            # print(desc, 'score: ', score)

            # Put text license plate number to image
            cv2.putText(img, desc.upper() + " ???", (300, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (50, 50, 200), 2)
            cv2.imshow('Recognize & Draw', img)
            cv2.waitKey(0)

            # Get first fruit only
            break

    print('Total time: {}'.format(datetime.now() - start_time))


print('---------- Start FOOD Recognition --------')
list_foods = load_food_category(CATEGORY)
print(list_foods)
path = SOURCE_PATH + 'pic1/.jpg'
recognize_food(path, list_foods)
print('---------- End ----------')