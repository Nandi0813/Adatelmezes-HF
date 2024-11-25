import os
import cv2
import pandas as pd
from keras.src.preprocessing.image import ImageDataGenerator
import numpy as np


def process_image_data(base_path, image_size, targets=None):
    """
    Univerzális függvény train és teszt adatok feldolgozására.

    Paraméterek:
    - base_path: A mappa, amely a képeket vagy mappákat tartalmazza.
    - image_size: A képek újraméretezésének mérete (szélesség, magasság).
    - targets: Az osztálycímkék listája (ha train adatokat dolgozol fel). Ha None, tesztadatként kezeli.

    Visszatérési érték:
    - DataFrame, amely tartalmazza a képadatokat laposított formában és címkékkel.
    """
    data = []
    is_training = targets is not None

    if is_training:
        folders = [folder for folder in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, folder))]
        for folder, target in zip(sorted(folders), targets):
            folder_path = os.path.join(base_path, folder)
            for filename in os.listdir(folder_path):
                if filename.endswith(".png"):
                    img_path = os.path.join(folder_path, filename)
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        img_resized = cv2.resize(img, image_size)
                        img_flattened = img_resized.flatten() / 255.0
                        data.append({'target': target, 'pixels': img_flattened})
    else:
        for filename in os.listdir(base_path):
            img_path = os.path.join(base_path, filename)
            if os.path.isfile(img_path) and filename.endswith(".png"):
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    img_resized = cv2.resize(img, image_size)
                    img_flattened = img_resized.flatten() / 255.0
                    data.append({'label': filename, 'pixels': img_flattened})

    df = pd.DataFrame(data)
    pixel_columns = [f'pixel_{i}' for i in range(image_size[0] * image_size[1])]
    pixels_expanded = pd.DataFrame(df['pixels'].to_list(), columns=pixel_columns)
    df = pd.concat([df.drop(columns='pixels'), pixels_expanded], axis=1)

    return df, None

def process_image_data_with_augmentation(base_path, image_size, targets=None, batch_size=128):
    """
    Univerzális függvény train és teszt adatok feldolgozására éldetektálással.

    Paraméterek:
    - base_path: A mappa, amely a képeket vagy mappákat tartalmazza.
    - image_size: A képek újraméretezésének mérete (szélesség, magasság).
    - targets: Az osztálycímkék listája (ha train adatokat dolgozol fel). Ha None, tesztadatként kezeli.

    Visszatérési érték:
    - DataFrame, amely tartalmazza a képadatokat laposított formában és címkékkel.
    """
    data = []

    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=5,
        width_shift_range=0.05,
        height_shift_range=0.05,
        shear_range=0.05,
        zoom_range=0.05,
        validation_split=0.15,
        preprocessing_function=laplace_preprocessing
    )

    test_datagen = ImageDataGenerator(
        rescale=1. / 255,
        preprocessing_function=laplace_preprocessing
    )



    is_training = targets is not None

    if is_training:
        train_generator = train_datagen.flow_from_directory(
            directory=base_path,
            target_size=image_size,
            batch_size=batch_size,
            class_mode='sparse',  # Adjust depending on your task
            subset='training',
            shuffle=True,
            color_mode = 'grayscale'
        )
        val_generator = train_datagen.flow_from_directory(
            directory=base_path,
            target_size=image_size,
            batch_size=batch_size,
            class_mode='sparse',
            subset='validation',
            shuffle=False,
            color_mode = 'grayscale'
        )
        return train_generator, val_generator
    else:
        test_generator = test_datagen.flow_from_directory(
            directory=base_path,
            target_size=image_size,
            classes=['test'],
            shuffle=False,
            color_mode = 'grayscale'
        )
        return test_generator, None


def laplace_preprocessing(image):
    """
    Laplace szűrés alkalmazása egy képre.

    Paraméter:
    - image: Input kép numpy tömbként (lebegőpontos vagy más formátum).

    Visszatérés:
    - Szűrt kép.
    """

    # Ha a kép RGB, konvertáljuk Grayscale-re
    if image.ndim == 3 and image.shape[-1] == 3:  # RGB kép
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # OpenCV Laplace szűrő csak 8-bites vagy 32-bites bemenetet támogat
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)  # Skálázás és konvertálás 8-bitre

    # Laplace szűrő alkalmazása
    edges = cv2.Laplacian(image, cv2.CV_64F)
    edges = cv2.convertScaleAbs(edges)  # Konvertálás 8-bites formátumba

    # Normalizálás [0, 1] tartományra
    edges = edges / 255.0

    # Bővítsük a dimenziókat a TensorFlow kompatibilitás miatt
    return np.expand_dims(edges, axis=-1)

def canny_preprocessing(image, threshold1=100, threshold2=200):
    """
    Canny éldetektálás alkalmazása egy képre.

    Paraméterek:
    - image: Input kép numpy tömbként (lebegőpontos vagy más formátum).
    - threshold1: Az alsó határ a hiszterézis küszöböléshez.
    - threshold2: A felső határ a hiszterézis küszöböléshez.

    Visszatérés:
    - Szűrt kép, Canny éldetektálással.
    """

    # Ha a kép RGB, konvertáljuk Grayscale-re
    if image.ndim == 3 and image.shape[-1] == 3:  # RGB kép
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # OpenCV Canny szűrő csak 8-bites bemenetet támogat
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)  # Skálázás és konvertálás 8-bitre

    # Canny éldetektálás alkalmazása
    edges = cv2.Canny(image, threshold1, threshold2)

    # Normalizálás [0, 1] tartományra
    edges = edges / 255.0

    # Bővítsük a dimenziókat a TensorFlow kompatibilitás miatt
    return np.expand_dims(edges, axis=-1)

def sobel_preprocessing(image):
    """
    Sobel éldetektálás alkalmazása egy képre.

    Paraméter:
    - image: Input kép numpy tömbként (lebegőpontos vagy más formátum).

    Visszatérés:
    - Szűrt kép a Sobel éldetektálással.
    """

    # Ha a kép RGB, konvertáljuk Grayscale-re
    if image.ndim == 3 and image.shape[-1] == 3:  # RGB kép
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # OpenCV Sobel szűrő csak 8-bites vagy 32-bites bemenetet támogat
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)  # Skálázás és konvertálás 8-bitre

    # Sobel szűrő alkalmazása
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)  # Gradiens X irányban
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)  # Gradiens Y irányban

    # Gradiens nagyságának számítása
    sobel_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    sobel_magnitude = cv2.convertScaleAbs(sobel_magnitude)  # 8-bites konvertálás

    # Normalizálás [0, 1] tartományra
    sobel_normalized = sobel_magnitude / 255.0

    # Bővítsük a dimenziókat a TensorFlow kompatibilitás miatt
    return np.expand_dims(sobel_normalized, axis=-1)

