from scipy.stats import norm
from csv import writer

def generate_points(num_points:int = 2000):
    # obiekty reprezentujące rozkłady prawdopodobieństwa dla każdej zmiennej
    distribution_x = norm(loc=0, scale=20) # loc to średnia, scale to odchylenie standardowe
    distribution_y = norm(loc=0, scale=200) # 0 to początek układu współrzędnych
    distribution_z = norm(loc=0.2, scale=0.05) # 0,2 to podniesienie układu współrzędnych w osi Z, 0,05 oznacza płaską poziomą powierzchnię

    x = distribution_x.rvs(size=num_points) # generowanie losowych próbek z rozkładu normalnego dla zmiennej x
    y = distribution_y.rvs(size=num_points)
    z = distribution_z.rvs(size=num_points)

    points = zip(x, y, z) # łączenie próbek w krotki (x, y, z)
    return points


if __name__ == '__main__':
    cloud_points = generate_points(2000) # generowanie punktów

    with open('cloud_points.csv', 'w', newline='\n') as csvfile: # otwieranie pliku CSV do zapisu
        csv_writer = writer(csvfile) # tworzenie obiektu writer do zapisu danych w formacie CSV
        csv_writer.writerow(['x', 'y', 'z']) # zapisanie nagłówków kolumn
        for point in cloud_points:
            csv_writer.writerow(point) # zapisanie każdego punktu do pliku CSV
