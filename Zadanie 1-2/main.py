from scipy.stats import norm, uniform
from csv import writer
from numpy import cos, pi, sin

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


def generate_points_flat_horizontal_surface(num_points:int = 2000, length = 200, width = 200):
    # obiekty reprezentujące rozkłady prawdopodobieństwa dla każdej zmiennej
    distribution_x = uniform(loc=0, scale=length) # loc to począrtek, scale to koniec układu współrzędnych
    distribution_y = uniform(loc=0, scale=width) # długość to lenght - start (w tym przypadku start = 0)
    distribution_z = norm(loc=0.2, scale=0.05)

    x = distribution_x.rvs(size=num_points) # generowanie losowych próbek z rozkładu normalnego dla zmiennej x
    y = distribution_y.rvs(size=num_points)
    z = distribution_z.rvs(size=num_points)

    points = zip(x, y, z) # łączenie próbek w krotki (x, y, z)
    return points


def generate_points_flat_vertical_surface(num_points:int = 2000, length = 200, height = 200):
    # obiekty reprezentujące rozkłady prawdopodobieństwa dla każdej zmiennej
    distribution_x = uniform(loc=0, scale=length) 
    distribution_y = norm(loc=0.2, scale=0.05)
    distribution_z = uniform(loc=0, scale=height)

    x = distribution_x.rvs(size=num_points) # generowanie losowych próbek z rozkładu normalnego dla zmiennej x
    y = distribution_y.rvs(size=num_points)
    z = distribution_z.rvs(size=num_points)

    points = zip(x, y, z) # łączenie próbek w krotki (x, y, z)
    return points


def generate_points_cylindrical_surface(num_points:int = 2000, radius = 100, height = 200):
    theta = uniform(loc=0, scale=2 * pi).rvs(size=num_points) # kąt obrotu wokół osi Z, równomiernie rozłożony w zakresie od 0 do 2π
    x = radius * cos(theta)
    y = radius * sin(theta)

    distribution_z = uniform(loc=0, scale=height) # wysokość cylindra
    z = distribution_z.rvs(size=num_points)

    points = zip(x, y, z) # łączenie próbek w krotki (x, y, z)
    return points


if __name__ == '__main__':
    cloud_points = generate_points(2000) # generowanie punktów
    horizontal_surface_points = generate_points_flat_horizontal_surface(2000, 200, 200)
    vertical_surface_points = generate_points_flat_vertical_surface(2000, 200, 200)
    cylindrical_surface_points = generate_points_cylindrical_surface(2000, 100, 200)

    with open('cloud_points.xyz', 'w', newline='\n') as file: # otwieranie pliku do zapisu
        csv_writer = writer(file) # tworzenie obiektu writer do zapisu danych
        csv_writer.writerow(['x', 'y', 'z']) # zapisanie nagłówków kolumn
        for point in cloud_points:
            csv_writer.writerow(point) # zapisanie każdego punktu do pliku

    with open('horizontal_surface_points.xyz', 'w', newline='\n') as file:
        csv_writer = writer(file)
        csv_writer.writerow(['x', 'y', 'z'])
        for point in horizontal_surface_points:
            csv_writer.writerow(point)

    with open('vertical_surface_points.xyz', 'w', newline='\n') as file:
        csv_writer = writer(file)
        csv_writer.writerow(['x', 'y', 'z'])
        for point in vertical_surface_points:
            csv_writer.writerow(point)

    with open('cylindrical_surface_points.xyz', 'w', newline='\n') as file:
        csv_writer = writer(file)
        csv_writer.writerow(['x', 'y', 'z'])
        for point in cylindrical_surface_points:
            csv_writer.writerow(point)