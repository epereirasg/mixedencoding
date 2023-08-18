import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# Función objetivo: Minimizar el costo total de asignación
def objective_function(solution):
    binary_section = solution[:, :5]
    real_section = solution[:, 5:10]
    permutation_section = solution[:, 10:15]
    integer_section = solution[:, 15:]

    cost = np.sum(binary_section, axis=1) * 100  # Costo de tareas obligatorias
    cost += np.sum(real_section, axis=1) * 10  # Costo estimado de tiempo
    cost += np.sum(integer_section, axis=1) * 5  # Costo de tareas opcionales asignadas
    return cost

# Función para validar y corregir las soluciones generadas
def validate_solution(solution):
    # Asegurar que las variables binarias estén en {0, 1}
    solution[:, :5] = np.round(solution[:, :5])
    # Asegurar que las variables reales estén en el rango [0, 1]
    solution[:, 5:10] = np.clip(solution[:, 5:10], 0, 1)
    # Asegurar que las variables de permutación sean permutaciones válidas
    for i in range(solution.shape[0]):
        perm_section = solution[i, 10:15]
        unique_elements = np.unique(perm_section)
        if len(unique_elements) != perm_section.shape[0]:
            perm_section[:] = np.random.permutation(perm_section)
    # Asegurar que las variables enteras sean enteras no negativas
    solution[:, 15:] = np.round(np.clip(solution[:, 15:], 0, None))

# Parámetros del algoritmo genético
population_size = 50
solution_length = 20
mutation_rate = 0.1
generations = 100

# Crear la población inicial
population = np.random.rand(population_size, solution_length)

# Crear el modelo para optimizar las soluciones
model = Sequential([
    Dense(128, activation='relu', input_shape=(solution_length,)),
    Dense(1)
])

model.compile(optimizer=Adam(learning_rate=0.01), loss='mean_squared_error')

for generation in range(generations):
    fitness = np.array([objective_function(solution.reshape(1, solution_length)) for solution in population])
    model.fit(population, fitness, epochs=1, verbose=0)

    fitness_probs = fitness / np.sum(fitness)
    parents_indices = np.random.choice(population_size, size=population_size)

    children = []
    for i in range(population_size):
        parent1 = population[parents_indices[i]]
        parent2 = population[np.random.choice(population_size)]

        crossover_point = np.random.randint(1, solution_length)
        child = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))

        for j in range(solution_length):
            if np.random.rand() < mutation_rate:
                child[j] = np.random.rand()  # Mutación aleatoria en todas las secciones

        children.append(child)

    population = np.array(children)
    validate_solution(population)

best_solution = population[np.argmin(fitness)]
best_fitness = fitness[np.argmin(fitness)]

print("Mejor solución:", best_solution)
print("Mejor aptitud:", best_fitness)