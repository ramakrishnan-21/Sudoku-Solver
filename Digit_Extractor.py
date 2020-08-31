import numpy as np
import cv2
from keras.models import load_model

def load_model_mnist():
    model_mnist = load_model('model/CNN.h5')
    return model_mnist

def preprocess_for_CNN(digit: np.ndarray):
    digit = 255 - digit
    digit = digit/255
    digit = digit.reshape((1, 28, 28, 1))
    return digit

def take_decision(probabilities, cost_r=10, cost_w=30):
    assert cost_w != 0
    pred_class = np.argmax(probabilities)
    if(probabilities[pred_class] > (1 - (cost_r/cost_w))):
        return pred_class
    return 0

def predictDigit(digit,model_mnist):
    processed_digit = preprocess_for_CNN(digit)
    prob = model_mnist.predict(processed_digit)
    prob = prob[0]
    return take_decision(prob, 8, 10)

def checkEmpty(digits):
    solution = [-1] * 81
    for i in range(0,81):
        if np.sum(digits[i])==0:
            solution[i]=0
    for i in range(0,81):
        if np.sum(digits[i])!=0:
            solution[i] = predictDigit(digits[i])
    solution = np.array(solution)
    return solution

def digit_extractor(digits):
    solution = [-1] * 81
    model_mnist = load_model_mnist()
    for i in range(0,81):
        if np.sum(digits[i])==0:
            solution[i]=0
    for i in range(0,81):
        if np.sum(digits[i])!=0:
            solution[i] = predictDigit(digits[i],model_mnist)
    solution = np.array(solution)
    solution = solution.reshape(9, 9)
    return solution
