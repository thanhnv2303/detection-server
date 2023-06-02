import io
from PIL import Image
from flask import Flask, request
from ultralytics import YOLO

app = Flask(__name__)
model = YOLO('best_300.pt')


@app.route("/", methods=["POST", "GET"])
def hello():
    return {
        "hello": "2"
    }


@app.route("/detect", methods=["POST"])
def predict():
    if not request.method == "POST":
        return

    if request.files.get("image"):
        image_file = request.files["image"]
        image_bytes = image_file.read()

        conf = float(request.form.get("conf") or 0.45)
        if conf > 1 or conf < 0:
            conf = 0.5
        img = Image.open(io.BytesIO(image_bytes))
        results = model.predict(source=img, conf=conf)
        _boxes = []
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
        results_json = {
            "boxes": _boxes,
            "total": len(_boxes)
        }
        return results_json


# ngrok_tunnel = ngrok.connect(8000)
# print('Public URL:', ngrok_tunnel.public_url)
# nest_asyncio.apply()
app.run(host="0.0.0.0", port=8000)
