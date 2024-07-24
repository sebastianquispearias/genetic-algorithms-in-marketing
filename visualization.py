import matplotlib.pyplot as plt
import random

def plot_fitness_evolution(generations, best_fitness, average_fitness):
    # Gráfico de la evolución de la aptitud
    plt.figure(figsize=(10, 6))
    plt.plot(generations, best_fitness, label='Mejor Aptitud')
    plt.plot(generations, average_fitness, label='Aptitud Promedio')
    plt.xlabel('Generación')
    plt.ylabel('Aptitud')
    plt.title('Evolución de la Aptitud')
    plt.legend()
    plt.savefig('images/fitness_evolution.png')
    plt.show()

# Datos ficticios para visualización
generations = list(range(1, 101))
best_fitness = [random.uniform(50, 100) for _ in generations]
average_fitness = [random.uniform(40, 90) for _ in generations]

# Llamada a la función para generar el gráfico
plot_fitness_evolution(generations, best_fitness, average_fitness)
