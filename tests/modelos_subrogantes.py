import numpy as np
from sklearn.ensemble import RandomForestRegressor

# Generar datos de entrenamiento
x_train = np.random.uniform(-10, 10, size=20)
y_train = x_train ** 2 - 4 * x_train + 4

# Ajustar un modelo subrogante (por ejemplo, Random Forest)
model = RandomForestRegressor()
model.fit(x_train.reshape(-1, 1), y_train)


def fitness_function(x, model):
    # Utilizar el modelo subrogante para predecir f(x)
    predicted_fx = model.predict(np.array([x]).reshape(-1, 1))
    return predicted_fx[0]


def genetic_algorithm(model, generations, population_size, mutation_rate):
    population = np.random.uniform(-10, 10, size=population_size)

    for gen in range(generations):
        # Evaluar la aptitud de cada individuo en la población utilizando el modelo subrogante
        fitness_values = [fitness_function(x, model) for x in population]

        # Seleccionar padres para la reproducción (por ejemplo, torneo)
        parents_indices = np.random.choice(population_size, size=population_size // 2, replace=False)

        # Generar descendencia mediante cruzamiento y mutación
        offspring = []
        for i in range(0, len(parents_indices), 2):
            parent1 = population[parents_indices[i]]
            parent2 = population[parents_indices[i + 1]]
            child = (parent1 + parent2) / 2  # Cruzamiento promedio
            if np.random.rand() < mutation_rate:
                child += np.random.uniform(-0.5, 0.5)  # Mutación
            offspring.append(child)

        # Reemplazar la población anterior con la nueva descendencia
        population[:len(offspring)] = offspring

    # Encontrar la mejor solución y su valor objetivo utilizando el modelo subrogante
    best_solution_index = np.argmin(fitness_values)
    best_solution = population[best_solution_index]
    best_fitness = fitness_values[best_solution_index]

    return best_solution, best_fitness


# Parámetros del algoritmo genético
generations = 50
population_size = 50
mutation_rate = 0.1

# Ejecutar el algoritmo genético utilizando el modelo subrogante
best_solution, best_fitness = genetic_algorithm(model, generations, population_size, mutation_rate)

print("Mejor solución encontrada:", best_solution)
print("Valor objetivo correspondiente (según el modelo subrogante):", best_fitness)