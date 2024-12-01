import os
import cv2
import pandas as pd

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

    return df

def process_image_data_with_edges(base_path, image_size, targets=None):
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
                        img_blurred = cv2.GaussianBlur(img_resized, (3, 3), 0)
                        img_enhanced = cv2.equalizeHist(img_blurred)

                        img_flattened = img_enhanced.flatten() / 255.0
                        data.append({'target': target, 'pixels': img_flattened})
    else:
        for filename in os.listdir(base_path):
            img_path = os.path.join(base_path, filename)
            if os.path.isfile(img_path) and filename.endswith(".png"):
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    img_resized = cv2.resize(img, image_size)
                    img_blurred = cv2.GaussianBlur(img_resized, (3, 3), 0)
                    img_enhanced = cv2.equalizeHist(img_blurred)

                    img_flattened = img_enhanced.flatten() / 255.0
                    data.append({'target': filename, 'pixels': img_flattened})

    df = pd.DataFrame(data)
    pixel_columns = [f'pixel_{i}' for i in range(image_size[0] * image_size[1])]
    pixels_expanded = pd.DataFrame(df['pixels'].to_list(), columns=pixel_columns)
    df = pd.concat([df.drop(columns='pixels'), pixels_expanded], axis=1)

    return df