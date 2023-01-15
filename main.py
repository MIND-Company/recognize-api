from fastapi import FastAPI, UploadFile, File
from recognition_model import recognize_plate
import uvicorn
import cv2

app = FastAPI()


@app.post("/api/camera/get-carplate")
def register_entry(img: UploadFile = File(...)):
    # img = cv2.imread(img.filename)
    return {"plate": recognize_plate(img.file)}


if __name__ == "__main__":
    uvicorn.run("main:app", port=8000, log_level="info", reload=True)
