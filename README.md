# Dashboard Streamlit

## Prerrequisitos

- Python 3.8 o superior

## Instalación y Configuración


### Descargar el repositorio de GitHub
```bash
# Debes darle click al boton verde que dice CODE y ahi descargas el archivo .ZIP
```


### Crear entorno virtual (Desde la terminal)
```bash
# Crear el entorno virtual
python -m venv dashboard_env

# Activar el entorno virtual
# En Windows:
dashboard_env\Scripts\activate
```

### Instalar dependencias (Desde la terminal)
### Con este archivo (requirements.txt) se instalan todas las librerias necesarias (Se puede hacer solo en el entorno para que no ocupe espacio en el pc)
```bash
pip install -r requirements.txt
```

### Ejecutar el dashboard (Desde la terminal)
```bash
streamlit run app.py
```

El dashboard se abrirá automáticamente en tu navegador en `http://localhost:8501`

## Estructura del proyecto

```
tu-repositorio/
├── app.py                 # Archivo principal del dashboard (SIEMPRE LLAMARLO APP.PY)
├── requirements.txt       # Dependencias del proyecto (ARCHIVO CON LOS REQUISITOS PARA TENER EL AMBIENTE VIRTUAL IGUAL)
├── data/                  # Carpeta con archivos de datos (AQUI SE GUARDARAN LOS DATAFRAMES)
├── components/            # Componentes reutilizables (IMAGENES Y LOGOS ESTATICOS QUE QUERAMOS PONER EN EL DASHBOARD)
└── README.md             # Este archivo
```
