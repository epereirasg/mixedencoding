import random

# Definimos el tamaño de la población y la longitud del cromosoma binario
pop_size = 10
chromosome_length = 20

# Definimos los límites de la variable real
lower_bound = 0
upper_bound = 1

# Definimos una función de fitness de ejemplo
def fitness_function(chromosome):
    integer_part = int(chromosome[:10], 2)  # Convertir los primeros 10 bits a un entero
    real_part = lower_bound + int(chromosome[10:], 2) * (upper_bound - lower_bound) / (2 ** 10 - 1)  # Convertir los últimos 10 bits a un número real
    return (integer_part + real_part) ** 2

# Construimos una población aleatoria de cromosomas binarios
population = []
for i in range(pop_size):
    chromosome = ''
    for j in range(chromosome_length):
        chromosome += str(random.randint(0, 1))
    population.append(chromosome)

# Evolución de la población
for generation in range(100):
    # Evaluamos la aptitud de cada cromosoma
    fitness_scores = [fitness_function(chromosome) for chromosome in population]
    # Selección
    parents = random.choices(population, weights=fitness_scores, k=2)
    # Cruce: mezcla de los primeros 10 bits del primer cromosoma y los últimos 10 bits del segundo cromosoma
    offspring = parents[0][:10] + parents[1][10:]
    # Mutación: cambiar aleatoriamente uno de los bits
    mutation_bit = random.randint(0, chromosome_length - 1)
    if offspring[mutation_bit] == '0':
        offspring = offspring[:mutation_bit] + '1' + offspring[mutation_bit+1:]
    else:
        offspring = offspring[:mutation_bit] + '0' + offspring[mutation_bit+1:]
    # Agregamos el nuevo hijo a la población
    population.append(offspring)
    # Selección de sobrevivientes: reemplazamos los cromosomas menos aptos
    fitness_scores = [fitness_function(chromosome) for chromosome in population]
    # Seleccionamos solo los cromosomas menos aptos
    indices_to_remove = random.choices(range(len(population)), weights=fitness_scores, k=pop_size - 1)
    # Utilizamos una lista temporal para almacenar los cromosomas generados en esta generación
    new_population = [population[i] for i in range(len(population)) if i not in indices_to_remove]
    # Utilizamos la nueva población para la siguiente generación
    population = new_population

# Obtenemos los cromosomas más aptos
fitness_scores = [fitness_function(chromosome) for chromosome in population]
best_chromosome = population[fitness_scores.index(max(fitness_scores))]

print('Mejor cromosoma:', best_chromosome)
integer_part = int(best_chromosome[:10], 2)
real_part = lower_bound + int(best_chromosome[10:], 2) * (upper_bound - lower_bound) / (2 ** 10 - 1)
print('Valor entero:', integer_part)
print('Valor real:', real_part)