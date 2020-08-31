import copy
import numpy as np
from keras.models import load_model
import sys

def load_model_sudoku():
    model_sudoku = load_model('model/sudoku.model')
    return model_sudoku

def norm(a):
     return (a / 9) - .5


def denorm(a):
    return (a + .5) * 9


def predict_sudoku(sample, model_sudoku):
    feat = copy.copy(sample)
    while(1):
        out = model_sudoku.predict(feat.reshape((1,9,9,1)))  
        out = out.squeeze()
        pred = np.argmax(out, axis=1).reshape((9,9))+1 
        prob = np.around(np.max(out, axis=1).reshape((9,9)), 2) 
        feat = denorm(feat).reshape((9,9))
        mask = (feat==0)
        if(mask.sum()==0):
            break
        prob_new = prob * mask
        ind = np.argmax(prob_new)
        x, y = (ind//9), (ind%9)
        val = pred[x][y]
        feat[x][y] = val
        feat = norm(feat)
    return pred

def solve_sudoku(game,model_sudoku):
    game = np.array(game).reshape(9,9,1)
    game = norm(game)
    game = predict_sudoku(game,model_sudoku)
    return game


def output(a):
    sys.stdout.write(str(a))

def display_sudoku(sudoku):
    for i in range(9):
        for j in range(9):
            cell = sudoku[i][j]
            if cell == 0 or isinstance(cell, set):
                output('.')
            else:
                output(cell)
            if (j + 1) % 3 == 0 and j < 8:
                output(' |')

            if j != 8:
                output('  ')
        output('\n')
        if (i + 1) % 3 == 0 and i < 8:
            output("--------+----------+---------\n")
   
def print_col_sum(grid):
    print(np.sum(grid,axis = 1))

def final_sudoku(solution):
    model_sudoku = load_model_sudoku()
    game = solve_sudoku(solution,model_sudoku)
    print("Solution")
    print("\n")
    display_sudoku(game)
    print("\n")
    print("\n")
    print("Columns sum (if everyting is 45 then we have solved it correctly)")
    print_col_sum(game)