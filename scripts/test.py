import os
import numpy as np
from tensorflow import keras
from keras import utils

COLOR_MODE = "rgb"
ROOT = os.path.dirname(os.getcwd())
IMG_SIZE = (224, 224)
BATCH_SIZE = 64
sports = ["Basketball", "Football", "Rowing", "Swimming", "Tennis", "Yoga"]
CLASS_COUNT = len(sports)

output_dir = ROOT + "/output/"

class Test:
    data = None
    model = None

    def __init__(self, data_path: str):
        """

        :param data_path: path to directory where the data is located,
                          the test images should be in a subdirectory
        """
        self.__load_data(data_path)
        self.__load_model()

    def __load_data(self, data_path: str):
        self.data = utils.image_dataset_from_directory(data_path,
                                                       image_size=IMG_SIZE,
                                                       batch_size=BATCH_SIZE,
                                                       color_mode=COLOR_MODE,
                                                       shuffle=False)

    def __load_model(self):
        xception_model_path = ROOT + '/saved_models/pretrained-xcep-e05-acc0.97.hdf5'
        self.model = keras.models.load_model(xception_model_path)

    def predict(self, save: bool = True):
        pred = self.model.predict(self.data)
        pred_list = [sports[np.argmax(sample_pred)] for sample_pred in pred]
        print(pred_list)
        if save:
            self.__save_to_csv(pred_list)

    def __save_to_csv(self, pred_list):
        np.savetxt(f'{output_dir}\\predictions{len(os.listdir(output_dir)) + 1}.csv', pred_list, fmt='%s')
