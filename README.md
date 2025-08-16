# Dashboard Streamlit

## Prerrequisitos

- Python 3.8 o superior

## Instalación y Configuración


### Crear entorno virtual (Desde la terminal)
```bash
# Crear el entorno virtual
python -m venv dashboard_env

# Activar el entorno virtual
# En Windows:
dashboard_env\Scripts\activate
```

### 3. Instalar dependencias (Desde la terminal)
```bash
pip install -r requirements.txt
```

### 4. Ejecutar el dashboard (Desde la terminal)
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


### Para desactivar el entorno virtual:
```bash
deactivate
```
