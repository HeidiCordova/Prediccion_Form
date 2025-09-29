
import logging
import os
from datetime import datetime

def configurar_logging(nivel=logging.INFO, ruta_logs="logs"):
    """Configurar sistema de logging centralizado"""
    
    # Crear directorio de logs si no existe
    os.makedirs(ruta_logs, exist_ok=True)
    
    # Formato de log
    formato = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Archivo de log con timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    archivo_log = os.path.join(ruta_logs, f'sistema_predictivo_{timestamp}.log')
    
    # Configurar logging
    logging.basicConfig(
        level=nivel,
        format=formato,
        handlers=[
            logging.FileHandler(archivo_log, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    # Reducir verbosidad de módulos específicos
    logging.getLogger('ModeloPrediccionIET').setLevel(logging.WARNING)
    logging.getLogger('ModeloPrediccionIPC').setLevel(logging.WARNING)
    
    return logging.getLogger("SistemaPredictivoIndustrial")
