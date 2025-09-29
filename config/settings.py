import logging
import os
from datetime import datetime
class ConfiguracionSistema:
    """Configuraciones centralizadas del sistema"""
    
    # Configuración de datos
    RUTA_DATOS_DEFAULT = "datos.csv"  # Sin "./"
    RUTA_MODELOS = "modelos_entrenados"
    RUTA_LOGS = "logs"
    
    # Configuración de modelos
    HORIZONTE_HORAS_DEFAULT = 2
    PRECISION_MINIMA = 70.0
    CONFIANZA_OBJETIVO = 80.0
    
    # Configuración de usuarios
    INDICADORES_PRIORITARIOS = ['IVM', 'IET', 'IPC']
    HORIZONTE_MIN_MINUTOS = 15
    HORIZONTE_MAX_HORAS = 24
    
    # Configuración de calidad de datos
    CALIDAD_MINIMA_OK = 75.0
    CALIDAD_MINIMA_WARNING = 60.0
    
    # Configuración de monitoreo
    INTERVALO_MONITOREO_MINUTOS = 60
    
    @classmethod
    def get_config(cls) -> dict:
        """Obtener configuración como diccionario"""
        return {
            'ruta_datos': cls.RUTA_DATOS_DEFAULT,
            'ruta_modelos': cls.RUTA_MODELOS,
            'horizonte_horas': cls.HORIZONTE_HORAS_DEFAULT,
            'precision_minima': cls.PRECISION_MINIMA,
            'confianza_objetivo': cls.CONFIANZA_OBJETIVO,
            'indicadores_prioritarios': cls.INDICADORES_PRIORITARIOS
        }

def configurar_logging(nivel=logging.INFO):
    """Configurar sistema de logging"""
    formato = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    logging.basicConfig(
        level=nivel,
        format=formato,
        handlers=[
            logging.FileHandler('sistema_predictivo.log', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    # Reducir verbosidad de módulos problemáticos
    logging.getLogger('ModeloPrediccionIET').setLevel(logging.WARNING)
    logging.getLogger('ModeloPrediccionIPC').setLevel(logging.WARNING)
    
    return logging.getLogger("SistemaPredictivoIndustrial")