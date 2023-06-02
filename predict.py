import json

from ultralytics import YOLO

# Load a model
model = YOLO('best_300.pt')  # load an official model
# model = YOLO('path/to/best_300.pt')  # load a custom model

# Predict with the model
# results = model.predict(source='pCard3', save=True, save_txt=True,project="playing_card",name="predict")
_boxes = []
results = model.predict(source='pCard3/1.jpg', save=True, save_txt=True, project="playing_card", name="predict")
# results = model('https://cdn.britannica.com/23/194523-050-E6C02DBE/selection-American-playing-cards-jack-queen-ace.jpg')
for result in results:
    r = result.numpy()
    names = r.names
    boxes = r.boxes
    for box in boxes:
            b = box.xywh[0].tolist()  # get box coordinates in (top, left, bottom, right) format
            c = int(box.cls[0])
            cf = float(box.conf[0])
            n = names[c]
            _boxes.append({
                "label": c,
                'name': n,
                'probability': cf,
                'bounding': b

            })
j = json.dumps(_boxes)
print(_boxes)
