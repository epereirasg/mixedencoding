import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# Función objetivo de prueba (ajustar según tus necesidades)
def objective_function(solution):
    binary_section = solution[:5]
    real_section = solution[5:10]
    permutation_section = solution[10:15]
    integer_section = solution[15:]

    cost = np.sum(binary_section) * 100
    cost += np.sum(real_section) * 10
    cost += np.sum(integer_section) * 5
    return cost

# Parámetros del algoritmo genético
population_size = 50
solution_length = 20
mutation_rate = 0.1
generations = 100

# Crear población inicial con variables de decisión mixtas
population = np.random.random(size=(population_size, solution_length)).astype(np.float32)
population[:, :5] = np.round(population[:, :5])  # Inicializar variables binarias

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
                elif 10 <= j < 15:  # Variables enteras
                    child[j] = np.random.randint(0, 10)
                elif 15 <= j < 20:  # Variables enteras
                    child[j] = np.random.randint(0, 10)

        children.append(child)

    population = np.array(children)

# Obtener la mejor solución encontrada
best_solution = population[np.argmin(fitness)]
best_fitness = fitness[np.argmin(fitness)]

print("Mejor solución:", best_solution)
print("Mejor aptitud:", best_fitness)