# Algoritmo Genético para la Gestión de Perfiles de Clientes

Este proyecto implementa un algoritmo genético para optimizar la selección de perfiles de clientes basados en múltiples atributos. El objetivo es identificar los perfiles de clientes más valiosos para una empresa y adaptar las estrategias de marketing en consecuencia.

## Descripción

El algoritmo utiliza cadenas binarias para representar diferentes atributos de los clientes, incluyendo datos personales, secciones visitadas, enlaces, relaciones y compras. La aptitud de cada perfil se evalúa en función de su potencial valor económico para la empresa.

## Estructura del Proyecto

- `main.py`: Contiene el código principal del algoritmo genético.
- `utils.py`: Funciones auxiliares para la generación de perfiles, evaluación de aptitud, cruce y mutación.
- `visualization.py`: Herramientas para la visualización de resultados.
- `data/`: Directorio para almacenar datos de entrada y salida.

## Cómo Ejecutar

1. Clona este repositorio.
2. Instala las dependencias:
   ```bash
   pip install -r requirements.txt
