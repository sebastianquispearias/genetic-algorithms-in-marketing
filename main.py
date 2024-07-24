import pandas as pd
import random
import multiprocessing

# Cargar el conjunto de datos
df = pd.read_csv('data/bank.csv', sep=';')

# Preprocesar el conjunto de datos
from sklearn.preprocessing import LabelEncoder

# Crear un diccionario para almacenar los codificadores
label_encoders = {}

# Convertir las variables categóricas en numéricas
categorical_columns = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome', 'y']
for column in categorical_columns:
    label_encoders[column] = LabelEncoder()
    df[column] = label_encoders[column].fit_transform(df[column])

# Convertir el DataFrame en una lista de cadenas binarias
def preprocess_data(df):
    binary_strings = df.apply(lambda row: ''.join(format(x, 'b').zfill(8) for x in row), axis=1)
    return binary_strings.tolist()

customer_list = preprocess_data(df)

# Definición de parámetros del algoritmo genético
POPULATION_SIZE = 100
MUTATION_RATE = 0.01
CROSSOVER_RATE = 0.7
MAX_GENERATIONS = 5000

total_string_length = len(customer_list[0])

# Función paralelizada para calcular la aptitud
def fitness_parallel(individual, customer_list):
    total_fitness = 0
    for customer in customer_list:
        genes_coincident = sum(1 for a, b in zip(individual, customer) if a == b)
        total_fitness += genes_coincident
    return total_fitness

# Evaluación de la aptitud para toda la población
def evaluate_population(population):
    with multiprocessing.Pool() as pool:
        fitness_values = pool.starmap(fitness_parallel, [(ind, customer_list) for ind in population])
    return fitness_values

# Función para aplicar la mutación
def mutate(individual):
    return ''.join(
        bit if random.random() > MUTATION_RATE else ('1' if bit == '0' else '0')
        for bit in individual
    )

# Función para realizar el cruce de dos individuos
def crossover(parent1, parent2):
    if random.random() > CROSSOVER_RATE:
        return parent1, parent2
    point = random.randint(0, total_string_length - 1)
    return parent1[:point] + parent2[point:], parent2[:point] + parent1[point:]

# Función principal del algoritmo genético
def main():
    # Inicialización de una población aleatoria
    population = [''.join(random.choice(['0', '1']) for _ in range(total_string_length)) for _ in range(POPULATION_SIZE)]

    generation = 0
    last_best_fitness = 0
    stagnant_count = 0  # Contador de generaciones sin mejora
    
    while generation < MAX_GENERATIONS:
        population.sort(key=lambda ind: fitness_parallel(ind, customer_list), reverse=True)
        
        # Comprobar si se encontró la solución
        if fitness_parallel(population[0], customer_list) == total_string_length:
            print(f"¡Solución encontrada en la generación {generation}!")
            print("Mejor individuo:", population[0])
            break
        
        # Comprobar el progreso estancado
        current_best_fitness = fitness_parallel(population[0], customer_list)
        if current_best_fitness == last_best_fitness:
            stagnant_count += 1
        else:
            stagnant_count = 0
        last_best_fitness = current_best_fitness
        
        # Si no hay mejora durante 100 generaciones, terminar
        if stagnant_count >= 100:
            print("Sin mejora durante 100 generaciones, terminando...")
            break

        # Seleccionar padres y generar nueva población
        new_population = population[:2]  # Elitismo: tomar los dos mejores directamente para la próxima generación
        while len(new_population) < POPULATION_SIZE:
            parent1 = random.choice(population[:50])  # Seleccionar de los 50 mejores individuos
            parent2 = random.choice(population[:50])
            offspring1, offspring2 = crossover(parent1, parent2)
            new_population.extend([mutate(offspring1), mutate(offspring2)])

        population = new_population
        generation += 1
        print(f"Generación {generation}: Mejor aptitud: {fitness_parallel(population[0], customer_list)}")

    if generation == MAX_GENERATIONS:
        print("Se alcanzó el número máximo de generaciones sin encontrar una solución.")
        print("Mejor individuo:", population[0])

    return population, [fitness_parallel(ind, customer_list) for ind in population]

# Ejecutar el algoritmo genético
if __name__ == "__main__":
    final_population, objective_values = main()
    print("\nPoblación Final:", final_population)
    print("Valores Objetivos:", objective_values)
