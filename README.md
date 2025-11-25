Descripción del Proyecto: Predictor de Churn de Clientes Telco
Este proyecto consiste en el desarrollo de una Aplicación Web Interactiva para la predicción de la tasa de abandono de clientes (Churn) en la industria de las telecomunicaciones, utilizando modelos de Machine Learning.

La solución está diseñada para ser funcional, fácilmente desplegable y cumple con los requisitos técnicos de un proyecto de ciencia de datos operacionalizado.

Objetivo del Proyecto
El objetivo principal es proporcionar una herramienta predictiva en tiempo real que, basada en las características de un cliente (servicios contratados, antigüedad, cargos mensuales, etc.), determine la probabilidad de que este abandone el servicio de la compañía. Esto permite a la empresa tomar acciones proactivas de retención.

Modelos de Predicción e Implementación
La aplicación incorpora dos modelos de clasificación, ambos accesibles a través de una única interfaz de entrada de datos:

Modelo de Regresión Logística (Logistic Regression):

Salida: Proporciona la Probabilidad de Churn (valor continuo entre 0 y 1) y la Clasificación final (Yes/No).

Modelo K-Nearest Neighbors (KNN):

Salida: Proporciona una Clasificación discreta (Yes/No) basada en la predicción del vecino más cercano.

Estrategia de Carga de Modelos:
Los modelos están estructurados para ser pre-entrenados y cargados en la web, cumpliendo con los requisitos técnicos:

Al iniciar la aplicación (localmente o en el despliegue), si los archivos de modelo no existen, el script entrena automáticamente los pipelines de scikit-learn y guarda los objetos (.joblib equivalentes a .pkl) en la carpeta dedicada /modelos.

En ejecuciones posteriores, los modelos se cargan directamente desde la ruta /modelos, garantizando una respuesta inmediata.

Interfaz de Usuario y Funcionalidad
La interfaz de usuario es clara, funcional y sigue un diseño organizado en columnas.

Formulario de Entrada: Se presenta un formulario organizado en tres secciones (Demografía/Contrato, Servicios y Cargos) que incluye campos interactivos como select boxes, sliders y number inputs para capturar las variables relevantes del dataset Telco.

Resultados: La aplicación muestra los resultados de ambos modelos de manera separada y distintiva, utilizando indicadores visuales:

Probabilidad de Churn (Regresión Logística).

Clasificación Yes/No (para ambos modelos, con mensajes de éxito/error).

Acceso Online: El proyecto está optimizado para el despliegue continuo en Streamlit Cloud, siendo accesible directamente en el enlace proporcionado: https://telco-customer-churn-95u48i4qhlb8x9hzx6gla2.streamlit.app.
