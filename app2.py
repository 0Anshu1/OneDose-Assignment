from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from PIL import Image
import io

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"], #allowing the sources like websites or frontend running on port 3000
    #it is similar to the line CORS(app) that I used in flask code.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = YOLO("../yolov8n.pt")

@app.post("/detect")
async def detect_objects(image: UploadFile = File(...)):
    if not image:
        raise HTTPException(status_code=400, detail="No image file provided")

    try:
        contents = await image.read()
        pil_image = Image.open(io.BytesIO(contents))
        #detecting the objects
        results = model(pil_image)

        detections = []
        for result in results:
            for box in result.boxes:
                detections.append({
                    "class": model.names[int(box.cls)],
                    "confidence": float(box.conf),
                    "bbox": [float(coord) for coord in box.xyxy[0]]
                })

        return {"detected_objects": detections}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")
