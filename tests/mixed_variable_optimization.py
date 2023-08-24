import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam



"""
Problema de Asignación de Tareas con Codificación Híbrida

Este problema trata sobre la asignación de tareas a trabajadores con el objetivo de minimizar el costo total de asignación. Se busca resolver el problema utilizando un algoritmo genético y una codificación híbrida que combina varios tipos de variables para representar las decisiones de asignación. Estas decisiones incluyen la asignación de tareas obligatorias y opcionales, así como el tiempo estimado para completar las tareas asignadas.

Definición Matemática:

Dado un conjunto de trabajadores W y un conjunto de tareas T, el objetivo es encontrar una asignación A que minimice el costo total de asignación. La asignación A se representa como un vector x de longitud n, donde n es el número total de variables de decisión en la codificación híbrida. La asignación A consta de:

- Las primeras 5 variables de x representan la asignación de tareas obligatorias y son binarias: x_i ∈ {0, 1}, donde i = 1, 2, ..., 5.
- Las siguientes 5 variables de x representan el tiempo estimado para completar las tareas asignadas: x_i ∈ ℝ, donde i = 6, 7, ..., 10.
- Las siguientes 5 variables de x representan una permutación de las tareas restantes: x_i ∈ ℕ, donde i = 11, 12, ..., 15.
- Las últimas 5 variables de x representan la cantidad de tareas opcionales asignadas: x_i ∈ ℕ, donde i = 16, 17, ..., 20.

El costo total de asignación C se define como:

C = ∑_{i=1}^{5} x_i * 100 + ∑_{i=6}^{10} x_i * 10 + ∑_{i=16}^{20} x_i * 5

Donde x_i es el valor de la variable de decisión i en el vector x.

El algoritmo genético se utiliza para evolucionar las asignaciones a lo largo de generaciones, buscando una solución que minimice el costo total de asignación. Se utiliza una función de aptitud basada en el costo para evaluar la calidad de cada asignación en la población. A través de selección, cruzamiento y mutación, el algoritmo genético busca encontrar la mejor asignación que cumpla con las restricciones del problema y minimice el costo total.
"""

# Función objetivo de prueba
def objective_function(solution):
    binary_section = solution[:5]
    real_section = solution[5:10]
    permutation_section = solution[10:15]
    integer_section = solution[15:]

    cost = np.sum(binary_section) * 100
    cost += np.sum(real_section) * 10
    cost += np.sum(integer_section) * 5
    return cost

# Función para generar permutaciones únicas
def generate_unique_permutation(length):
    perm = np.arange(length)
    np.random.shuffle(perm)
    return perm

# Parámetros del algoritmo genético
population_size = 50
solution_length = 20
mutation_rate = 0.1
generations = 100

# Crear población inicial con variables de decisión mixtas
population = np.random.random(size=(population_size, solution_length)).astype(np.float32)
population[:, :5] = np.round(population[:, :5])  # Inicializar variables binarias
population[:, 15:20] = np.random.randint(0, 10, size=(population_size, 5))  # Inicializar variables enteras
population[:, 10:15] = np.array([generate_unique_permutation(5) for _ in range(population_size)])  # Generar permutaciones únicas

# Crear modelo para optimizar las soluciones
model = Sequential([
    Dense(128, activation='relu', input_shape=(solution_length,)),
    Dense(1)
])

model.compile(optimizer=Adam(learning_rate=0.01), loss='mean_squared_error')

for generation in range(generations):
    # Evaluar la población
    fitness = np.array([objective_function(solution) for solution in population])

    # Entrenar el modelo para aprender la relación entre soluciones y puntuaciones de aptitud
    model.fit(population, fitness, epochs=1, verbose=0)

    # Seleccionar padres usando la probabilidad de aptitud
    fitness_probs = fitness / np.sum(fitness)
    parents_indices = np.random.choice(population_size, size=population_size, p=fitness_probs)

    # Cruzamiento y mutación
    children = []
    for i in range(population_size):
        parent1 = population[parents_indices[i]]
        parent2 = population[np.random.choice(population_size)]

        crossover_point = np.random.randint(1, solution_length)
        child = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))

        for j in range(solution_length):
            if np.random.rand() < mutation_rate:
                if j < 5:  # Variables binarias
                    child[j] = 1 - child[j]
                elif 5 <= j < 10:  # Variables reales
                    child[j] = np.random.uniform(0, 1)
                elif 15 <= j < 20:  # Variables enteras
                    child[j] = np.random.randint(0, 10)

        # Corregir la asignación de permutaciones
        if 10 <= j < 15:  # Permutaciones
            child[j] = generate_unique_permutation(5)

        children.append(child)

    population = np.array(children)

# Obtener la mejor solución encontrada
best_solution = population[np.argmin(fitness)]
best_fitness = fitness[np.argmin(fitness)]

print("Mejor solución:", best_solution)
print("Mejor aptitud:", best_fitness)