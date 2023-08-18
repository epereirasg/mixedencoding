from jmetal.core.problem import Problem
from jmetal.core.solution import Solution
from jmetal.operator.mutation import PolynomialMutation
from jmetal.operator.crossover import SBXCrossover
from jmetal.algorithm.multiobjective.nsgaii import NSGAII
from jmetal.lab.visualization import Plot
import random

# Definición de la clase de solución para variables mixtas
class MixedVariablesSolution(Solution):
    def __init__(self, number_of_variables, number_of_objectives, number_of_constraints):
        super(MixedVariablesSolution, self).__init__(number_of_variables, number_of_objectives, number_of_constraints)
        self.types = [0] * number_of_variables
        self.lower_bound = [0] * number_of_variables
        self.upper_bound = [0] * number_of_variables

# Definición del problema
class MixedVariablesProblem(Problem):
    BINARY = 0
    INTEGER = 1
    REAL = 2
    def __init__(self):
        super(MixedVariablesProblem, self).__init__()
        self.number_of_variables = 6
        self.number_of_objectives = 2
        self.number_of_constraints = 0
        self.obj_directions = [self.MAXIMIZE, self.MINIMIZE]
        self.lower_bound = [0, 0, 0, 0, 0, 0]
        self.upper_bound = [1, 1, 10, 10, 10, 10]
        self.types = [self.BINARY, self.BINARY, self.INTEGER, self.INTEGER, self.INTEGER, self.REAL]

    # Resto del código...

    def create_solution(self) -> MixedVariablesSolution:
        # Implementar la creación de una solución para el problema
        solution = MixedVariablesSolution(
            number_of_variables=self.number_of_variables,
            number_of_objectives=self.number_of_objectives,
            number_of_constraints=self.number_of_constraints
        )

        for i in range(self.number_of_variables):
            if self.types[i] == self.BINARY:
                solution.variables[i] = random.choice([0, 1])
            elif self.types[i] == self.INTEGER:
                solution.variables[i] = random.randint(self.lower_bound[i], self.upper_bound[i])
            elif self.types[i] == self.REAL:
                solution.variables[i] = random.uniform(self.lower_bound[i], self.upper_bound[i])

        return solution

    def evaluate(self, solution: MixedVariablesSolution) -> MixedVariablesSolution:
        # Implementar la evaluación de la solución para el problema
        # Utilizar la función random para generar números aleatorios
        for i in range(self.number_of_variables):
            if self.types[i] == self.BINARY:
                solution.objectives[0] += random.random() * solution.variables[i]
                solution.objectives[1] += (1 - random.random()) * solution.variables[i]
            elif self.types[i] == self.INTEGER:
                solution.objectives[0] += random.random() * solution.variables[i]
                solution.objectives[1] += random.random() * (self.upper_bound[i] - solution.variables[i])
            elif self.types[i] == self.REAL:
                solution.objectives[0] += random.random() * solution.variables[i]
                solution.objectives[1] += random.random() * (self.upper_bound[i] - solution.variables[i])

        return solution

    def get_name(self) -> str:
        # Definir el nombre del problema
        return "MixedVariablesProblem"

# Creación de una instancia del problema
problem = MixedVariablesProblem()

# Configuración de los operadores de cruce y mutación
crossover = SBXCrossover(probability=1.0, distribution_index=20)
mutation = PolynomialMutation(probability=1.0, distribution_index=20)

# Configuración del algoritmo NSGA-II
algorithm = NSGAII(
    problem=problem,
    population_size=100,
    offspring_population_size=100,  # Tamaño de la población descendiente
    mutation=mutation,
    crossover=crossover,
    termination_criterion=100,
)

# Ejecución del algoritmo
algorithm.run()

# Obtención de los resultados
front = algorithm.get_result()

# Visualización de la solución
plot_front = Plot(title='Pareto front approximation', axis_labels=['F1', 'F2'])
plot_front.plot(front, label='NSGA-II')
plot_front.show()

# Imprimir los resultados
for i, solution in enumerate(front):
    print(f"Solución {i+1}:")
    print(f"Variables binarias: {solution.variables[0]}, {solution.variables[1]}")
    print(f"Variables enteras: {solution.variables[2]}, {solution.variables[3]}, {solution.variables[4]}")
    print(f"Variable real: {solution.variables[5]}")
    print(f"Objetivos: {solution.objectives[0]}")
