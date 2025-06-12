import random

num_filas = 2000
archivo = "data.csv"

with open(archivo, "w") as f:
    f.write("x,y\n")

    for _ in range(num_filas):
        x = random.uniform(0, 100)
        y = random.uniform(0, 100)
        f.write(f"{x},{y}\n")

print(f"Archivo creado: {archivo}")
