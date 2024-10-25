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
