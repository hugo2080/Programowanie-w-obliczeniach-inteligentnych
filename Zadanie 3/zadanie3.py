import os
import cv2
from skimage.feature import graycomatrix, graycoprops
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

def process_textures(input_directory:str, output_directory:str, image_size, sample_size:tuple=(128, 128)):
    # utworzenie katalogu wyjściowego, jeśli nie istnieje
    for category_folder in os.listdir(input_directory):
        category_path = os.path.join(input_directory, category_folder)
        if os.path.isdir(category_path):
            output_category_path = os.path.join(output_directory, category_folder)
            os.makedirs(output_category_path, exist_ok=True)

            # obróka obrazów w każdej kategorii
            for image_file in os.listdir(category_path):
                if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        image_path = os.path.join(category_path, image_file)
                        image = cv2.imread(image_path) # wczytanie obrazu
                        resized_image = cv2.resize(image, image_size) # zmiana rozmiaru obrazu do określonego rozmiaru

                        counter = 0
                        for y in range(0, resized_image.shape[0] - sample_size[1] + 1, sample_size[1]):
                            for x in range(0, resized_image.shape[1] - sample_size[0] + 1, sample_size[0]):
                                sample = resized_image[y:y + sample_size[1], x:x + sample_size[0]] # wycięcie próbki z obrazu
                                output_sample_path = os.path.join(output_category_path, f"{os.path.splitext(image_file)[0]}_sample_{counter}.png")
                                cv2.imwrite(output_sample_path, sample) # zapisanie próbki do katalogu wyjściowego
                                counter += 1


def extract_features(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE) # wczytanie w skali szarości

    target_levels = 64 # ilość poziomów jasności po kwantyzacji
    interval = 256 // target_levels # dzielnik do kwantyzacji
    reduced_image = (image // interval) # kwantyzacja

    # wyznaczanie cech GLCM
    distances = [1, 3, 5] # odległości pikseli
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4] # kąty w radianach
    glcm = graycomatrix(reduced_image, distances=distances, angles=angles, levels=target_levels, symmetric=True, normed=True) # obliczanie macierzy zdarzeń (ang. Grey-Level Co-occurrence Matrix)
    
    properties = ['dissimilarity', 'correlation', 'contrast', 'energy', 'homogeneity', 'ASM'] # cechy do wyznaczenia
    vector = [] # wektor cech
    for prop in properties:
        vector.extend(graycoprops(glcm, prop).ravel())

    return vector


def create_dataset(samples_directory, output_csv):
    dataset = []  # przygotowanie pustej listy na dane

    for category_folder in os.listdir(samples_directory):
        category_path = os.path.join(samples_directory, category_folder)
        if os.path.isdir(category_path):
            for image_file in os.listdir(category_path):
                if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_path = os.path.join(category_path, image_file)
                    vec = extract_features(image_path)  # wyodrębnianie cech z próbki
                    vec.append(category_folder)  # dodanie etykiety kategorii
                    dataset.append(vec)  # dodanie całego wiersza do dataset

    # zapisanie danych do pliku CSV
    columns = [f"feature_{i}" for i in range(72)] + ['label']
    df = pd.DataFrame(dataset, columns=columns)
    df.to_csv(output_csv, index=False)


def classifier(X, Y):
    model = svm.SVC(gamma='auto') # tworzenie klasyfikatora SVM z automatycznym doborem parametru gamma
    
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.33) # podział danych na zbiór treningowy i testowy
    
    model.fit(x_train, y_train) # trenowanie klasyfikatora na zbiorze treningowym
    y_pred = model.predict(x_test) # przewidywanie etykiet dla zbioru testowego
    acc = accuracy_score(y_test, y_pred)
    print(f"Dokładność modelu: {acc}")


if __name__ == '__main__':
    input_directory = 'Tekstury' # katalog wejściowy z obrazami
    output_directory = 'Tekstury_samples' # katalog wyjściowy na próbki
    image_size = (512, 512) # rozmiar obrazu po zmianie rozmiaru
    sample_size = (128, 128) # rozmiar próbki (szerokość, wysokość)

    process_textures(input_directory, output_directory, image_size, sample_size)

    output_csv = 'dataset.csv' # nazwa pliku CSV z danymi
    create_dataset(output_directory, output_csv)

    df = pd.read_csv(output_csv)
    X = df.drop(columns=['label']) # pobranie cech z kolumn, bez kolumny 'label'
    Y = df['label'] # pobranie etykiet z kolumny 'label'
    classifier(X, Y)
