Whole Process :

    1. image of sudoku :
        find sudoku board on image extract that imae and its grayscale image
    
    2. Extrating the digits : 
        extract digit from the grayscale image(all the 81 cells, with none if none present)
        Basically save the endpoints of each digit cell.

    3. convert those image to array (converting all those cells into array)

    4. Predict from that array using mnist model 

    5. Add this predicted value to a board and null if not present

    6. Print these predicted numbers on board using cv2 putText

    7. Ask user if the predictions are wrong. If yes, ask him the wrong prediction and update them in board

    8. Solve the board finally

    9. Show this solved sudoku on the same image using putText

    10. You are done with it!!! Hurray :)


Detailing of each function : 

    1. Finding Puzzle : 

        Make grayscale image from BGR(default color for cv2)
        Use gaussian blur on that grayscale image
        Use adaptive filtering for that blur image
        Bitwise and with this filter
        Now find contours on the image
        Grab all the contours found
        Now sort them(on the basis of contour area, in descending order)
        Apply approxPolyDP for each contour using perimeter of that contour(used for edge removal)
        Finally create perspective view for sudoku using contour.

    2. Extract digits : 
        
        Use threshold on each subimage of digit and clear_border(remove extra white border)
        Find contours
        Grab contours
        If no contour : return none (since the image is empty)
        Find the largest contour in the cell and create a mask for the contour
        Find filled percentage of all and if it is less than 0.03 ignore it since it will be none.
        All the mask to the thresholded image and then dilate it.
        Return this dilated image.

    3. Display all numbers on board :

        For each cell(whose endpoint coordinates were saved in the array while extracted each cell) extract those endpoints
        putText in that cell using cv2
        That's it 
        
    4. Main function :

        Read sudoku image
        Resize it
        Call find_puzzle and get sudoku and it's grayscale image
        Find the step size of each cell from the grayscale image so as to do slicing to extract each cell and save the endpoints of the cell 
        Call extract_digit for each cell and get the dilated image 
        Convert this dilated image to array after some manipulation
        Predict from the model 
        Save the prediction in the board
        And Display this board on the same grayscale sudoku image

        Now ask that user that if all the predictions are correct and if some are wrong then ask him the right value and confirm that whether sudoku to be solved is correct or not.

        Now solve the board using sudoku module and display this solved board on the same extracted sudoku image.

        Display this solved image to the user. 

        It's done...
        Hurray!!!