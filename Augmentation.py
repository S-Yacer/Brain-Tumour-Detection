from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import os

def Dataset_Augmentation():

    path = 'C:\\Users\\44749\\Desktop\\Main Folders\\Final Year Project\\Dataset folders\\Original Dataset\\Training\\MAL'
    count_title = -1

    Data_generator = ImageDataGenerator(

        rotation_range = 40,
        width_shift_range = 0.2,
        height_shift_range = 0.2,
        rescale = 1./255,
        shear_range = 0.2,
        zoom_range = 0.2,
        horizontal_flip = True,
        vertical_flip= True,
        fill_mode = 'nearest')

    for filename in os.listdir(path):
        image_source = load_img(os.path.join(path, filename))
        image_array = img_to_array(image_source)

        # converting to a 4D image in order to process it through keras datagen
        image_array = image_array.reshape((1,) + image_array.shape)

        count = 0
        count_title = count_title + 1
        for batch in Data_generator.flow(image_array, batch_size = 1, save_to_dir = 'C:\\Users\\44749\\Desktop\\Main Folders\\Final Year Project\\Dataset folders\\Augmented ORIGINAL\\TRAIN\\MAL',
                                        save_prefix = 'Aug' + str(count_title), save_format = 'jpeg'):
            count = count + 1
            if count > 10:
                break


if __name__ == "__main__":
    Dataset_Augmentation()

