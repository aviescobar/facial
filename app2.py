from flask import Flask, request, jsonify, render_template
import numpy as np
import cv2
import matplotlib.pyplot as plt
import io
import base64
from PIL import Image


app = Flask(__name__)

# Cargar el modelo Haarcascades para la detección de rostros
