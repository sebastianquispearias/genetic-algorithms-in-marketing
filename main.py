import pandas as pd
import random
import multiprocessing
from sklearn.preprocessing import LabelEncoder

# Cargar datos desde un archivo CSV
df = pd.read_csv('data/bank.csv', sep=';')

# Convertir datos categóricos a numéricos
label_encoders = {}
categorical_columns = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome', 'y']
for column in categorical_columns:
    label_encoders[column] = LabelEncoder()
    df[column] = label_encoders[column].fit_transform(df[column])

# Convertir el DataFrame en una lista de cadenas binarias
def preprocess_data(df):
    binary_strings = df.apply(lambda row: ''.join(format(x, 'b').zfill(8) for x in row), axis=1)
    return binary_strings.tolist()

customer_list = preprocess_data(df)

# Parámetros del algoritmo genético
POPULATION_SIZE = 100
MUTATION_RATE = 0.01
CROSSOVER_RATE = 0.7
MAX_GENERATIONS = 5000

total_string_length = len(customer_list[0])

# Calcular la aptitud de un individuo comparando con todos los clientes
def fitness_parallel(individual, customer_list):
    total_fitness = 0
    for customer in customer_list:
        genes_coincident = sum(1 for a, b in zip(individual, customer) if a == b)
        total_fitness += genes_coincident
    return total_fitness

# Evaluar la aptitud de toda la población
def evaluate_population(population):
    with multiprocessing.Pool() as pool:
        fitness_values = pool.starmap(fitness_parallel, [(ind, customer_list) for ind in population])
    return fitness_values

# Aplicar mutación a un individuo
def mutate(individual):
    return ''.join(
        bit if random.random() > MUTATION_RATE else ('1' if bit == '0' else '0')
        for bit in individual
    )

# Realizar cruce entre dos padres para generar descendencia
def crossover(parent1, parent2):
    if random.random() > CROSSOVER_RATE:
        return parent1, parent2
    point = random.randint(0, total_string_length - 1)
    return parent1[:point] + parent2[point:], parent2[:point] + parent1[point:]

# Algoritmo genético principal
def main():
    # Generar población inicial aleatoria
    population = [''.join(random.choice(['0', '1']) for _ in range(total_string_length)) for _ in range(POPULATION_SIZE)]

    generation = 0
    last_best_fitness = 0
    stagnant_count = 0
    
    while generation < MAX_GENERATIONS:
        # Ordenar la población según la aptitud
        population.sort(key=lambda ind: fitness_parallel(ind, customer_list), reverse=True)
        
        # Obtener la mejor aptitud de la generación actual
        current_best_fitness = fitness_parallel(population[0], customer_list)
        
        # Verificar si se ha encontrado una solución óptima
        if current_best_fitness == total_string_length:
            print(f"¡Solución encontrada en la generación {generation}!")
            print("Mejor individuo:", population[0])
            break
        
        # Comprobar estancamiento en la mejora de la aptitud
        if current_best_fitness == last_best_fitness:
            stagnant_count += 1
        else:
            stagnant_count = 0
        last_best_fitness = current_best_fitness
        
        # Terminar si no hay mejoras después de 100 generaciones
        if stagnant_count >= 100:
            print("Sin mejora durante 100 generaciones, terminando...")
            break

        # Crear nueva población aplicando elitismo y generando nuevos individuos
        new_population = population[:2]
        while len(new_population) < POPULATION_SIZE:
            parent1 = random.choice(population[:50])
            parent2 = random.choice(population[:50])
            offspring1, offspring2 = crossover(parent1, parent2)
            new_population.extend([mutate(offspring1), mutate(offspring2)])

        population = new_population
        generation += 1
        print(f"Generación {generation}: Mejor aptitud: {current_best_fitness}")

    if generation == MAX_GENERATIONS:
        print("Se alcanzó el número máximo de generaciones sin encontrar una solución.")
        print("Mejor individuo:", population[0])

    return population, [fitness_parallel(ind, customer_list) for ind in population]

# Ejecutar el algoritmo genético
if __name__ == "__main__":
    final_population, objective_values = main()
    print("\nPoblación Final:", final_population)
    print("Valores Objetivos:", objective_values)
