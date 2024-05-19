from fastapi import FastAPI, File, UploadFile
from typing import List
import io
import numpy as np
from PIL import Image
import tensorflow as tf

app = FastAPI()

# Cargar el modelo h5 preentrenado
model = tf.keras.models.load_model('modelo.h5')

# Clases de ejemplo para el modelo (ajustar según el modelo utilizado)
CLASSES = [
    "perro",
    "caballo",
    "elefante",
    "polilla",
    "gallina",
    "gato",
    "vaca",
    "oveja",
    "araña",
    "ardilla"
]

# Función para procesar la imagen y obtener las predicciones
def read_imagefile(file) -> Image.Image:
    image = Image.open(io.BytesIO(file))
    image = image.convert("RGB")  # Convertir imagen a RGB (por si es RGBA u otro formato)
    image = image.resize((100, 100))  # Redimensionar a 224x224 (ajustar según el modelo)
    return image

def preprocess_image(image: Image.Image) -> np.ndarray:
    image = np.array(image) / 255.0  # Normalizar los valores de píxeles
    image = np.expand_dims(image, axis=0)
    return image

def predict(image: np.ndarray) -> List[float]:
    # Preprocesar la imagen antes de la predicción
    image = preprocess_image(image)
    prediction = model.predict(image)
    return prediction[0].tolist()

@app.post("/predict/")
async def predict_api(file: UploadFile = File(...)):
    try:
        image = read_imagefile(await file.read())
        predictions = predict(image)
        results = dict(zip(CLASSES, predictions))
        return results
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
