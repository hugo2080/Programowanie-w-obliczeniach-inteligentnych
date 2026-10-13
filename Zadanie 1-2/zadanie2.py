import numpy as np

def RANSAC(W, threshold:float=0.1, max_iterations:int=1000):
    # Implementacja algorytmu RANSAC
    best_inliers = []
    best_model = None

    for iteration in range(max_iterations):
        # 3 losowe punkty z chmury punktów W
        sample_indices = np.random.choice(len(W), 3, replace=False)
        A, B, C = W[sample_indices]

        # wektory łączące punkty
        vec_Va = B - A
        vec_Vb = C - A

        # obliczenie wektora normalnego do płaszczyzny
        vec_Vc = np.cross(vec_Va, vec_Vb) # iloczyn wektorowy dwóch wektorów tworzących płaszczyznę
        
        # sprawdzenie, czy punkty nie leżą na jednej prostej
        norm_Vc = np.linalg.norm(vec_Vc)
        if norm_Vc == 0:
            continue  # pomijamy iterację, punkty są współliniowe

        # normalizacja
        vec_Uc = vec_Vc / norm_Vc

        # obliczenie wyrazu wolnego równania płaszczyzny
        D = -np.sum(np.multiply(vec_Uc, C)) # mnożenie element po elemencie wektora normalnego i punktu C, a następnie sumowanie

        # sprawdzenie, jak dobrze punkty należą do płaszczyzny
        # określ warunek dobrego dopasowania, np. odległość punktu od płaszczyzny mniejsza niż pewien próg
        # wyznacz grupę inliers (punktów należących do płaszczyzny) i oblicz ile ich jest oraz outliers (punktów nie należących do płaszczyzny)
        # jeśli liczba inliers jest większa niż dotychczasowa najlepsza liczba inliers, zapisz nowy model płaszczyzny i zaktualizuj najlepszą liczbę inliers

        distance_all_points = np.abs(np.dot(W, vec_Uc) + D) / np.linalg.norm(vec_Uc) # obliczenie odległości wszystkich punktów od płaszczyzny
        inliers = np.where(np.abs(distance_all_points) <= threshold)[0] # indeksy punktów należących do płaszczyzny
        model_size = len(inliers) # liczba punktów należących do płaszczyzny
        best_model_size = len(best_inliers)

        if model_size > best_model_size:
            best_inliers = inliers
            best_model = (vec_Uc, D) # zapisanie najlepszego modelu płaszczyzny
        # jeśli osiągnięto maksymalną liczbę iteracji lub znaleziono wystarczającą liczbę inliers, przejdź dalej
        # dla inlierów z najlepszej grupy, oblicz nowy model płaszczyzny i zaktualizuj najlepszą liczbę inliers (np. metodą najmniejszych kwadratów)

    return best_model, best_inliers


if __name__ == '__main__':
    points = np.loadtxt('horizontal_surface_points.xyz', delimiter=',', skiprows=1)
    model, inliers = RANSAC(points, threshold=0.1, max_iterations=1000)
    print(f"Znaleziono {len(inliers)}/{len(points)} inlierów")
    print(f"Model płaszczyzny: wektor normalny = {model[0]}, D = {model[1]}")