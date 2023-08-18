import random

# Función de evaluación (objetivo a optimizar)
def fitness_function(x):
    return -x * x  # Minimizamos el negativo del cuadrado para obtener el máximo

# Parámetros del algoritmo genético
population_size = 10
num_generations = 50
mutation_rate = 0.1

# Codificación binaria
def binary_encoding(length):
    return [random.choice([0, 1]) for _ in range(length)]

# Algoritmo genético
def genetic_algorithm(encoding_function, encoding_args):
    population = [encoding_function(*encoding_args) for _ in range(population_size)]

    for generation in range(num_generations):
        # Evaluación de la población
        fitness_scores = [fitness_function(individual) for individual in population]

        # Selección de padres (seleccionamos los mejores individuos)
        parents = sorted(range(len(population)), key=lambda i: fitness_scores[i], reverse=True)[:population_size // 2]

        # Cruce y mutación para generar nueva generación
        new_population = []
        for _ in range(population_size // 2):
            parent1 = random.choice(parents)
            parent2 = random.choice(parents)
            crossover_point = random.randint(1, len(population[parent1]) - 1)
            child = population[parent1][:crossover_point] + population[parent2][crossover_point:]

            # Mutación
            if random.random() < mutation_rate:
                mutation_index = random.randint(0, len(child) - 1)
                child[mutation_index] = 1 - child[mutation_index]

            new_population.append(child)

        population = new_population

    fitness_scores = [fitness_function(individual) for individual in population]
    best_solution = population[fitness_scores.index(max(fitness_scores))]
    best_fitness = max(fitness_scores)

    return best_solution, best_fitness

# Ejemplo de uso
if __name__ == "__main__":
    print("Codificación Binaria:")
    best_binary, best_fitness_binary = genetic_algorithm(binary_encoding, [10])
    print("Mejor solución binaria:", best_binary)
    print("Mejor fitness:", best_fitness_binary)