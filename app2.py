from flask import Flask, request, jsonify, render_template
import numpy as np
import cv2
import matplotlib.pyplot as plt
import io
import base64
from PIL import Image


app = Flask(__name__)

# Cargar el modelo Haarcascades para la detección de rostros
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Función para detectar el rostro y dispersar los puntos clave
def process_image(image_data):
   # Leer la imagen desde los datos recibidos usando PIL y convertir a array numpy
    image = Image.open(io.BytesIO(image_data))
   image = np.array(image)

   # Asegurarse de que la imagen tiene 3 canales (RGB) antes de redimensionar
    if len(image.shape) == 2:
       image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

   # Redimensionar la imagen a (96, 96) con interpolación bicúbica para mejorar calidad
    resized_image = cv2.resize(image, (96, 96), interpolation=cv2.INTER_CUBIC)
