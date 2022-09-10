import cv2
import imutils
import numpy as np
from tensorflow.keras.models import load_model
import sudoku_utilities as su

#loading the model, image and image shape
model = load_model('model/model_mnist/')
img_path = '10.png'
img_shape = [28,28]

# reading the image and resizing it to 600
img = cv2.imread(img_path)
img = imutils.resize(img,width=600)

# getting only the sudoku image from the whole image 
board_image, gray_board_image = su.finding_board_image(img)

# creating empty board and location of cells whose value are already filled
board = np.zeros(shape=(9,9),dtype='int')
cell_locs = []

# converting board from image data 
su.sudoku_from_image(board, gray_board_image, img_shape, model, cell_locs)
# display the predicted digits on the sudoku image 
x = su.display_numbers_on_board(board, cell_locs, board_image)

# ask the user if all the integers predicted right or wrong and ask for correct one if they are wrong
su.user_confimation(board, board_image, cell_locs)

#solving the board and saving its result to board
board = su.solve(board)

# displaying the values of solved board on the sudoku image 
x = su.display_numbers_on_board(board, cell_locs, board_image)            

# Showing the final result 
cv2.imshow('solved',x)
cv2.waitKey(0)
cv2.destroyAllWindows()