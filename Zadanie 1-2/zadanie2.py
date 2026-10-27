import numpy as np
from csv import reader
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
import pyransac3d as pyrsc

def RANSAC(W, threshold:float=0.1, max_iterations:int=1000):
    # Implementacja algorytmu RANSAC
    best_inliers = []
    best_model = None

    for _ in range(max_iterations):
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


def points_reader(namefile:str):
    with open(namefile, 'r', newline='\n') as file:
        csv_reader = reader(file)
        next(csv_reader)  # pomijanie nagłówka
        points = np.array([list(map(float, row)) for row in csv_reader]) # konwersja danych z pliku do tablicy numpy
    return points


if __name__ == '__main__':
    cloud_points = points_reader('cloud_points.xyz')
    # 3 & 4
    clusterer = KMeans(n_clusters=3)
    X = np.array(cloud_points)
    y_pred = clusterer.fit_predict(X)

    red = y_pred == 0
    green = y_pred == 1
    blue = y_pred == 2

    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111, projection='3d')
    plt.tight_layout()

    ax1.scatter(X[red, 0], X[red, 1], X[red, 2], c='r', s=1, label='Cluster 1')
    ax1.scatter(X[green, 0], X[green, 1], X[green, 2], c='g', s=1, label='Cluster 2')
    ax1.scatter(X[blue, 0], X[blue, 1], X[blue, 2], c='b', s=1, label='Cluster 3')

    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('KMeans 3D Clustering')
    ax1.legend()

    clusters = {
        'Klaster 1 (czerwony)': X[red],
        'Klaster 2 (zielony)': X[green],
        'Klaster 3 (niebieski)': X[blue],
    }

    for name, cluster in clusters.items():
        model, inliers = RANSAC(cluster)
        normal_vec, D = model
        
        # obliczenie odległości punktów od znalezionej płaszczyzny
        distances = np.abs(np.dot(cluster, normal_vec) + D)
        inlier_ratio = len(inliers) / len(cluster)
        
        print(f"\n--- {name} (RANSAC) ---")
        print(f"Wektor normalny: [{normal_vec[0]:.4f}, {normal_vec[1]:.4f}, {normal_vec[2]:.4f}]")
        print(f"Odsetek inlierów: {inlier_ratio:.1%}, Średnia odległość: {np.mean(distances):.2f}")
        
        if inlier_ratio > 0.5:
            if abs(normal_vec[2]) > 0.8:
                typ = "płaszczyzna pozioma"
            else:
                typ = "płaszczyzna pionowa"
            print(f"Decyzja: Chmura jest płaszczyzną -> {typ}")
        else:
            print("Decyzja: Chmura nie jest płaszczyzną")

    # 6
    clusterer = DBSCAN(eps=20.0)
    X = np.array(cloud_points)
    y_pred = clusterer.fit_predict(X)
    
    red = y_pred == 0
    green = y_pred == 1
    blue = y_pred == 2
    
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111, projection='3d')
    plt.tight_layout()
    
    ax2.scatter(X[red, 0], X[red, 1], X[red, 2], c='r', s=1, label='Cluster 1')
    ax2.scatter(X[green, 0], X[green, 1], X[green, 2], c='g', s=1, label='Cluster 2')
    ax2.scatter(X[blue, 0], X[blue, 1], X[blue, 2], c='b', s=1, label='Cluster 3')
    
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.set_title('DBSCAN 3D Clustering')
    ax2.legend()

    plane = pyrsc.Plane()

    clusters = {
        'Klaster 1 (czerwony)': X[red],
        'Klaster 2 (zielony)': X[green],
        'Klaster 3 (niebieski)': X[blue],
    }

    for name, cluster in clusters.items():
        equation, inliers = plane.fit(cluster, thresh=0.1, maxIteration=1000)
        normal_vec = equation[:3]
        D = equation[3]
        
        inlier_ratio = len(inliers) / len(cluster)

        print(f"\n--- {name} (pyransac3d) ---")
        print(f"Wektor normalny: [{normal_vec[0]:.4f}, {normal_vec[1]:.4f}, {normal_vec[2]:.4f}]")
        print(f"Odsetek inlierów: {inlier_ratio:.1%}")
            
        if inlier_ratio > 0.5:
            if abs(normal_vec[2]) > 0.8:
                typ = "płaszczyzna pozioma"
            else:
                typ = "płaszczyzna pionowa"
            print(f"Decyzja: Chmura jest płaszczyzną -> {typ}")
        else:
            print("Decyzja: Chmura nie jest płaszczyzną")

    plt.show()