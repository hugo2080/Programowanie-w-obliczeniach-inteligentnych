import os
import cv2

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


if __name__ == '__main__':
    input_directory = 'Tekstury'  # katalog wejściowy z obrazami
    output_directory = 'Tekstury_samples'  # katalog wyjściowy na próbki
    image_size = (512, 512)  # rozmiar obrazu po zmianie rozmiaru
    sample_size = (128, 128)  # rozmiar próbki (szerokość, wysokość)

    process_textures(input_directory, output_directory, image_size, sample_size)