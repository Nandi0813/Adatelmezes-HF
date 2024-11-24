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
                        # Resize image
                        img_resized = cv2.resize(img, image_size)
                        # Edge detection
                        edges = cv2.Laplacian(img_resized, cv2.CV_64F)
                        edges = cv2.convertScaleAbs(edges)  # Convert to 8-bit
                        img_flattened = edges.flatten() / 255.0
                        data.append({'target': target, 'pixels': img_flattened})
    else:
        for filename in os.listdir(base_path):
            img_path = os.path.join(base_path, filename)
            if os.path.isfile(img_path) and filename.endswith(".png"):
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    # Resize image
                    img_resized = cv2.resize(img, image_size)
                    # Edge detection
                    edges = cv2.Laplacian(img_resized, cv2.CV_64F)
                    edges = cv2.convertScaleAbs(edges)  # Convert to 8-bit
                    img_flattened = edges.flatten() / 255.0
                    data.append({'label': filename, 'pixels': img_flattened})

    df = pd.DataFrame(data)
    pixel_columns = [f'pixel_{i}' for i in range(image_size[0] * image_size[1])]
    pixels_expanded = pd.DataFrame(df['pixels'].to_list(), columns=pixel_columns)
    df = pd.concat([df.drop(columns='pixels'), pixels_expanded], axis=1)

    return df

def process_image_data_with_edges_kenny(base_path, image_size, targets=None):
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
                        # Resize image
                        img_resized = cv2.resize(img, image_size)
                        # Edge detection using Canny
                        edges = cv2.Canny(img_resized, 100, 200)
                        img_flattened = edges.flatten() / 255.0
                        data.append({'target': target, 'pixels': img_flattened})
    else:
        for filename in os.listdir(base_path):
            img_path = os.path.join(base_path, filename)
            if os.path.isfile(img_path) and filename.endswith(".png"):
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    # Resize image
                    img_resized = cv2.resize(img, image_size)
                    # Edge detection using Canny
                    edges = cv2.Canny(img_resized, 100, 200)
                    img_flattened = edges.flatten() / 255.0
                    data.append({'label': filename, 'pixels': img_flattened})

    df = pd.DataFrame(data)
    pixel_columns = [f'pixel_{i}' for i in range(image_size[0] * image_size[1])]
    pixels_expanded = pd.DataFrame(df['pixels'].to_list(), columns=pixel_columns)
    df = pd.concat([df.drop(columns='pixels'), pixels_expanded], axis=1)

    return df

def process_image_data_with_edges_sobel(base_path, image_size, targets=None):
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
                        # Resize image
                        img_resized = cv2.resize(img, image_size)
                        # Edge detection using Sobel
                        sobelx = cv2.Sobel(img_resized, cv2.CV_64F, 1, 0, ksize=3)
                        sobely = cv2.Sobel(img_resized, cv2.CV_64F, 0, 1, ksize=3)
                        edges = cv2.magnitude(sobelx, sobely)
                        edges = cv2.convertScaleAbs(edges)  # Convert to 8-bit
                        img_flattened = edges.flatten() / 255.0
                        data.append({'target': target, 'pixels': img_flattened})
    else:
        for filename in os.listdir(base_path):
            img_path = os.path.join(base_path, filename)
            if os.path.isfile(img_path) and filename.endswith(".png"):
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    # Resize image
                    img_resized = cv2.resize(img, image_size)
                    # Edge detection using Sobel
                    sobelx = cv2.Sobel(img_resized, cv2.CV_64F, 1, 0, ksize=3)
                    sobely = cv2.Sobel(img_resized, cv2.CV_64F, 0, 1, ksize=3)
                    edges = cv2.magnitude(sobelx, sobely)
                    edges = cv2.convertScaleAbs(edges)  # Convert to 8-bit
                    img_flattened = edges.flatten() / 255.0
                    data.append({'label': filename, 'pixels': img_flattened})

    df = pd.DataFrame(data)
    pixel_columns = [f'pixel_{i}' for i in range(image_size[0] * image_size[1])]
    pixels_expanded = pd.DataFrame(df['pixels'].to_list(), columns=pixel_columns)
    df = pd.concat([df.drop(columns='pixels'), pixels_expanded], axis=1)

    return df
