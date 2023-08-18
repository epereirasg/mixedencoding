import numpy as np


def simulated_binary_crossover(P1, P2, eta_c, pc):
    C1 = np.copy(P1)
    C2 = np.copy(P2)

    if np.random.random() <= pc:
        for d in range(len(P1)):
            if np.random.random() <= 0.5:
                beta = (2 * np.random.random()) ** (1.0 / (eta_c + 1))
            else:
                beta = (0.5 / (1.0 - np.random.random())) ** (1.0 / (eta_c + 1))

            C1[d] = 0.5 * ((1 + beta) * P1[d] + (1 - beta) * P2[d])
            C2[d] = 0.5 * ((1 - beta) * P1[d] + (1 + beta) * P2[d])

            if C1[d] < 0:
                C1[d] = 0
            if C2[d] < 0:
                C2[d] = 0

            if np.random.random() <= (1 - beta):
                temp = C1[d]
                C1[d] = C2[d]
                C2[d] = temp

    return C1, C2


# Example usage
population = np.random.rand(25, 5)  # Example population with 25 individuals and 5 variables
parents_indices = np.random.randint(0, 25, size=24)  # Example parent indices

children = []
for i in range(0, len(parents_indices), 2):
    parent1 = population[parents_indices[i]]
    parent2 = population[parents_indices[i + 1]]
    eta_c = 2
    pc = 0.8
    child1, child2 = simulated_binary_crossover(parent1, parent2, eta_c, pc)
    children.append(child1)
    children.append(child2)

print("Children:")
for child in children:
    print(child)

