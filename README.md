
# PREDICCIÓN DE FÓRMULAS INDUSTRIALES

## Nombre del proyecto
**Predicción de Fórmulas Industriales**

## Breve descripción
Sistema modular para el procesamiento, análisis y predicción de indicadores industriales a partir de archivos CSV, con cálculo de fórmulas, validación de modelos y generación de reportes automáticos.

## Objetivo principal
Desarrollar una herramienta flexible y robusta que permita a la industria analizar datos de producción, calcular indicadores clave y predecir comportamientos futuros, facilitando la toma de decisiones basada en datos reales.


## Metodología aplicada: Kaizen y las 5S
Durante el desarrollo se aplicó la metodología Kaizen, enfocada en la mejora continua, implementando las 5S para optimizar el proceso y la calidad del software. Así, cada S se aplicó de la siguiente manera:

- **Seiri (Clasificar):**
	- Se realizó una revisión exhaustiva de los requerimientos y del código existente.
	- Se eliminaron archivos, funciones y dependencias innecesarias.
	- Se priorizaron los módulos realmente útiles para el objetivo industrial.

- **Seiton (Ordenar):**
	- Se diseñó una estructura de carpetas clara y lógica (core, data, formulas, interface, models, utils, docs).
	- Se documentó la ubicación y propósito de cada módulo.
	- Se implementaron convenciones de nombres para archivos, clases y funciones, facilitando la búsqueda y el mantenimiento.

- **Seiso (Limpiar):**
	- Se depuró el código eliminando redundancias, comentarios obsoletos y advertencias.
	- Se automatizó la generación de logs y reportes para detectar errores y mantener la limpieza del entorno de trabajo.
	- Se revisó periódicamente la base de datos de pruebas y los archivos temporales.

- **Seiketsu (Estandarizar):**
	- Se establecieron estándares de codificación y documentación para todo el equipo.
	- Se definieron formatos de entrada/salida y plantillas para reportes y logs.
	- Se promovió el uso de scripts de instalación y ejecución uniformes.

- **Shitsuke (Disciplina):**
	- Se fomentó la disciplina en la revisión de código y la actualización de la documentación.
	- Se implementaron rutinas de control de calidad y validación de modelos.
	- Se promovió la mejora continua mediante reuniones de retroalimentación y la adopción de buenas prácticas de ingeniería de software.




## Fase destacada: Hacer (Do)
La fase "Hacer" fue el núcleo del desarrollo y abarcó actividades técnicas y de ingeniería de datos, incluyendo:

- **Procesamiento y limpieza de datos:**
	- Implementación de parsers avanzados para archivos CSV industriales, capaces de detectar formatos, validar calidad y extraer información relevante.
	- Limpieza y filtrado de datos para asegurar la integridad y utilidad de la información procesada.

- **Cálculo de fórmulas industriales:**
	- Desarrollo de algoritmos robustos para el cálculo de indicadores clave a partir de los datos reales.
	- Se calcularon y predijeron **tres fórmulas principales**:
		1. **IVM (Índice de Variabilidad de Microparadas):**
			 - Fórmula: (σ/μ) × 100%
			 - Mide la variabilidad de las microparadas en la producción.
			 - Se calcula usando la desviación estándar y la media de los tiempos de microparadas.
		2. **IET (Índice de Eficiencia Temporal):**
			 - Fórmula: ((T. Disponible - T. Microparadas) / T. Disponible) × 100%
			 - Evalúa la eficiencia temporal del proceso productivo.
			 - Se calcula a partir del tiempo disponible y el tiempo perdido por microparadas.
		3. **IPC (Índice de Productividad de Conteo):**
			 - Fórmula: (Promedio.Conteo / Máximo.Conteo) × 100%
			 - Mide la productividad relativa respecto al máximo alcanzado.
			 - Se calcula usando el promedio y el máximo de conteos de producción.

- **Predicción de indicadores:**
	- Para cada fórmula, se desarrolló un modelo predictivo específico usando **Random Forest Regressor** (scikit-learn), entrenado con datos históricos y características temporales.
	- Los modelos predicen el valor futuro de cada indicador (IVM, IET, IPC) a distintos horizontes temporales (por ejemplo, 2 horas).
	- Se aplican técnicas de validación cruzada y detección de anomalías para asegurar la confiabilidad de las predicciones.

- **Validación y monitoreo:**
	- Validación cruzada de los modelos para asegurar precisión y confianza.
	- Generación automática de reportes ejecutivos y logs detallados para monitoreo y trazabilidad.

- **Organización y documentación:**
	- Organización modular del código, limpieza continua y documentación detallada de cada componente y proceso.

## ¿Cómo se realiza la predicción en `main_predictivo.py`?
El archivo `main_predictivo.py` es el núcleo del sistema de predicción industrial. Su funcionamiento es el siguiente:

1. **Inicialización:** Se carga el archivo de datos históricos (`datos.csv`) usando un parser avanzado que detecta el formato y valida la calidad de los datos.
2. **Análisis histórico:** Se calculan los indicadores industriales principales a partir de los datos reales, estableciendo una línea base.
3. **Entrenamiento de modelos:** Se entrenan modelos predictivos para cada indicador clave, ajustando los horizontes temporales y la precisión requerida.
4. **Validación:** Se valida el desempeño de los modelos usando los últimos registros, asegurando que cumplen con los objetivos de precisión y confianza.
5. **Producción y predicción:** Una vez validados, los modelos se ponen en producción y se realizan predicciones automáticas sobre los datos más recientes, generando alertas y reportes para el usuario final.
6. **Monitoreo:** El sistema puede simular monitoreo en tiempo real, enviando notificaciones y generando reportes periódicos.

El flujo completo está automatizado y documentado en los logs y reportes generados por el sistema.

## Tecnologías utilizadas
- Python 3.10+
- pandas
- numpy
- scikit-learn
- Estructura modular y buenas prácticas de ingeniería de software

## Descripción
Este proyecto implementa un sistema modular para el procesamiento, análisis y predicción de indicadores industriales a partir de archivos CSV con formatos específicos del sector. Incluye cálculo de fórmulas industriales, validación de modelos predictivos y generación de reportes automáticos.

## Estructura del repositorio

```
├── main.py                      # Ejecución principal: procesamiento y cálculo de fórmulas
├── main_predictivo.py           # Ejecución de sistema predictivo y generación de reportes
├── datos.csv                    # Archivo de datos de ejemplo
├── core/                        # Configuración y constantes
├── data/                        # Parsers y utilidades para archivos CSV/Excel
├── docs/                        # Documentación técnica y guías
├── formulas/                    # Cálculo de fórmulas industriales
├── interface/                   # Interfaz de línea de comandos (CLI)
├── models/                      # Modelos de predicción y validación
├── utils/                       # Utilidades generales
```

## Dependencias principales
- Python 3.10+
- pandas
- numpy
- scikit-learn

Instala las dependencias con:
```bash
pip install -r requirements.txt
```

## Uso básico

### Procesamiento y cálculo de fórmulas
Ejecuta:
```bash
python main.py
```
Esto analizará el archivo `datos.csv`, mostrará un resumen y calculará los principales indicadores industriales.

### Sistema predictivo y reportes
Ejecuta:
```bash
python main_predictivo.py
```
Genera reportes automáticos y logs en archivos de texto y `sistema_predictivo.log`.

## Módulos principales

- **core/config.py**: Configuración basada en requerimientos industriales.
- **data/csv_parser_avanzado.py**: Parser avanzado para archivos CSV industriales.
- **formulas/calculador_industrial.py**: Cálculo de fórmulas industriales con mapeo inteligente de columnas.
- **interface/cli.py**: Interfaz de línea de comandos para análisis y reportes.
- **models/prediction_models.py**: Modelos de predicción y resultados estructurados.
- **models/validation.py**: Validación y métricas de modelos predictivos.

## Ejemplo de ejecución

Al ejecutar `main.py`, el sistema:
1. Analiza el formato del archivo CSV.
2. Muestra un resumen de los datos cargados.
3. Calcula indicadores como IVM e IET, mostrando interpretaciones y registros válidos.
4. Presenta el mapeo de columnas detectado automáticamente.

## Contacto y soporte
Para dudas o mejoras, contacta a los desarrolladores o revisa la carpeta `docs/` para futuras guías y referencias.