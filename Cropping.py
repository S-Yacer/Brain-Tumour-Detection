import cv2
import numpy as np
import os

def image_cropping_directory():

   path = 'C:\\Users\\44749\\Desktop\\Main Folders\\Final Year Project\\Dataset folders\\Augmented Dataset\\Training\\Non-Cropped\\Mal'
   count = -1

   for filename in os.listdir(path):
        image_crop = cv2.imread(os.path.join(path, filename), cv2.IMREAD_GRAYSCALE)


# Trying multiple thresholds, currently 40 is ok for most dataset
        _, Threshold = cv2.threshold(image_crop, 40, 255, cv2.THRESH_BINARY)


        img2, contours, hierarchy = cv2.findContours(Threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


# Putting all contours together and reshaping to (_,2) (x,y) respectively
        cnts = np.vstack(contours).reshape(-1, 2)

# Extract the most left, most right, uppermost and lowermost point
        x_min = np.min(cnts[:, 0])
        y_min = np.min(cnts[:, 1])
        x_max = np.max(cnts[:, 0])
        y_max = np.max(cnts[:, 1])


        cropped = image_crop[y_min:y_max, x_min:x_max]

# Resizing to 128 x 128 for unity
        image = cv2.resize(cropped, (128, 128))

# To save image titles differently -> avoid overwrite
        count = count + 1

# Saving the cropped image
        path2 = 'C:\\Users\\44749\\Desktop\\Main Folders\\Final Year Project\\Dataset folders\\Augmented Dataset\\Training\\Cropped\\Cropped_Mal'
        cv2.imwrite(os.path.join(path2, 'image_cropped' + str(count) + '.jpg'), image)


if __name__ == "__main__":
    image_cropping_directory()

