#!/usr/bin/env python
# coding: utf-8


from Sudoku_Preprocessor import preprocess_output
from Digit_Extractor import digit_extractor
from Sudoku_Solver import final_sudoku
import cv2
import time
import matplotlib.pyplot as plt



def see_input_image(img):
    plt.imshow(img)



def sud_sol(orginal):
    digits = preprocess_output(orginal)
    solution = digit_extractor(digits)
    final_sudoku(solution)


def main(): 
    img_path = input("Enter path of the image: ")
    print("\n")
    original = cv2.imread(img_path)
    #see_input_image(original)
    t1 = cv2.getTickCount()
    sud_sol(original)
    t2 = cv2.getTickCount()
    t = (t2 - t1) / cv2.getTickFrequency()



if __name__ == "__main__":
    main()

