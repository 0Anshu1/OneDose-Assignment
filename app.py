from flask import Flask, request, jsonify
from flask_cors import CORS #for handling cross-origin resources(like frontend)
from ultralytics import YOLO
from PIL import Image
import io

app = Flask(__name__)
CORS(app)

model = YOLO("../yolov8n.pt")
@app.route('/detect', methods=['POST'])
def detect_objects():
    print("Request received!")
    print("Form keys:", request.form.keys())
    print("File keys:", request.files.keys()) #it will help us to debug if some errors regarding recieving files are recieved
    if 'image' not in request.files:
        print("Image not found in request.files")
        return jsonify({'error': 'No image file provided'}), 400

    image_file = request.files['image']
    print(f"Received image file: {image_file.filename}")

    try:
        image = Image.open(io.BytesIO(image_file.read()))
    except Exception as e:
        print("Error reading image:", e)
        return jsonify({'error': 'Invalid image'}), 400

    results = model(image)

    detections = []
    for result in results:
        for box in result.boxes:
            detections.append({
                "class": model.names[int(box.cls)],
                "confidence": float(box.conf),
                "bbox": [float(coord) for coord in box.xyxy[0]]
            })

    return jsonify({"detected_objects": detections})
if __name__ == '__main__':
    app.run(debug=True)
