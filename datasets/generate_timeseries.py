
import random

def generate():
    return [random.gauss(0,1) for _ in range(200)]
