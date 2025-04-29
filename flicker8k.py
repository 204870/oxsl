import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas
import PIL
import torch
from torchvision import transforms

# get list of images
images = []
f = open("images.txt")
for i in f:
    images.append(i[:-1])
f.close()

# load alexnet
model = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet', pretrained=True)
model.eval()

# load YOLOv5
yolo = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

for im in images:
    img = PIL.Image.open(im)

    # image preprocessing
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(img)
    input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

    # predict results
    with torch.no_grad():
        output = model(input_batch)

    # run softmax to get probabilities
    probabilities = torch.nn.functional.softmax(output[0], dim=0)

    with open("imagenet_classes.txt", "r") as f:
        categories = [s.strip() for s in f.readlines()]
    # Show top categories per image
    top5_prob, top5_catid = torch.topk(probabilities, 5)
    f = open("out/" + im[:-4] + "_imagenetcats.txt", "a")
    for i in range(top5_prob.size(0)):
        f.write(str(categories[top5_catid[i]]) + "\t" + str(top5_prob[i].item()))
        f.write("\n")
    f.close()

    # predict results
    results = yolo(img)
    pd_result = results.pandas().xyxy[0]

    img2 = np.asarray(cv2.imread(im))

    COLORS = np.random.uniform(124, 200, size=(pd_result.shape[0], 3))
    f = open("out/" + im[:-4] + "_yolocats.txt", "a")
    for idx in range(pd_result.shape[0]):
        # boxes
        box = pd_result.iloc[idx,:4].to_numpy(dtype=np.int32)
        startX, startY, endX, endY = box[0], box[1], box[2], box[3]
        # confidence
        confidence = pd_result.iloc[idx,4]
        confidence= format(confidence, '.4f')
        # name
        class_name = pd_result.name[idx]
        f.write(str(class_name) + "\t" + str(confidence))
        f.write("\n")
        # plot
        cv2.rectangle(img2, (startX, startY), (endX, endY), COLORS[idx], 5)
        y = startY - 15 if startY - 15 > 15 else startY + 15
        cv2.putText(img2, class_name, (startX, y-25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.75, COLORS[idx], 2)
        cv2.putText(img2, confidence, (startX, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.75, COLORS[idx], 2)
    f.close()

    #img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    img2path = "out/" + im[:-4] + "_labeled.jpg"
    cv2.imwrite(img2path, img2)
    #print("saved " + im[:-4] + "_labeled.jpg")
