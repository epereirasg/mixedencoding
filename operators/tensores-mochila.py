import numpy as np

# Tamaño de la mochila y valores de los artículos
capacidad_mochila = 10
valores = [8, 10, 6, 3, 7]
pesos = [4, 5, 3, 2, 5]

# Crear un tensor tridimensional para representar las codificaciones
# Dimensión 0: Binaria, Dimensión 1: Permutación, Dimensión 2: Números reales
tensor = np.zeros((3, len(valores), len(valores)))

# Generar soluciones iniciales en cada dimensión del tensor
for i in range(len(valores)):
    tensor[0, :, i] = np.random.randint(2, size=len(valores))  # Codificación binaria
    tensor[1, :, i] = np.random.permutation(len(valores))  # Codificación de permutación
    tensor[2, :, i] = np.random.rand(len(valores))  # Codificación de números reales


# Operador de mutación: Cambiar un valor aleatorio en cada solución
tensor_mutado = np.copy(tensor)
for dim in range(3):
    indice_mutacion = np.random.randint(len(valores))
    tensor_mutado[dim, indice_mutacion, np.random.randint(len(valores))] = np.random.rand()

# Evaluar las soluciones y encontrar la mejor en cada dimensión
for dim in range(3):
    solucion_actual = tensor[dim, :, :]
    valores_totales = np.sum(solucion_actual * valores)
    pesos_totales = np.sum(solucion_actual * pesos)
    print(f"Dimensión {dim}: Valor total = {valores_totales}, Peso total = {pesos_totales}")

# NOTA: Este es solo un ejemplo simplificado para ilustrar el concepto. La implementación y aplicación en problemas reales pueden requerir ajustes y consideraciones adicionales.

