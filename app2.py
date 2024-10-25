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

   # Convertir la imagen redimensionada a escala de grises
    gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)

   # Detectar el rostro en la imagen
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) == 0:
        return jsonify({'error': 'No face detected'}), 400

   # Tomar la primera cara detectada (si hay varias)
    (x, y, w, h) = faces[0]
    face_region = gray_image[y:y+h, x:x+w]

    # Inicializar el detector ORB con límite de 15 puntos clave
    orb = cv2.ORB_create(nfeatures=15)

   # Detectar puntos clave y descriptores en toda la región del rostro
    keypoints, descriptors = orb.detectAndCompute(face_region, None)

   # Dibujar "X" rojas en las posiciones de los puntos clave en la región del rostro
   plt.figure(figsize=(3, 3), dpi=96)
   plt.imshow(face_region, cmap='gray')

   for kp in keypoints:
      x_kp, y_kp = kp.pt
      plt.plot(x_kp, y_kp, 'rx', markersize=8)  # Marcadores más grandes para asegurar que sean visibles

   plt.axis('off')  # No mostrar ejes
   # Convertir el gráfico de matplotlib en imagen y luego a base64

      buf = io.BytesIO()
      plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
      buf.seek(0)
      img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
   return img_base64

   # Ruta para la página HTML
   @app.route('/')
   def index():
      return render_template('index.html')

# Ruta para procesar la imagen

