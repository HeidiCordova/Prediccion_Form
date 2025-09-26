# ============================================================================
# SISTEMA DE PREDICCIÓN INDUSTRIAL MODULAR - SUPERUSUARIO
# Basado en requerimientos del cuestionario de usuarios industriales
# ============================================================================

import os
import sys
import time
import datetime
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')
@dataclass
class ConfiguracionSuperusuario:
    """Configuración basada en resultados del cuestionario industrial"""
    
    # Horizontes temporales según menciones del cuestionario
    horizontes_temporales = {
        'inmediato_15_30min': {
            'periodos': 6, 
            'minutos': 30, 
            'menciones': 2, 
            'prioridad': 'baja',
            'porcentaje_usuarios': 15
        },
        'corto_1_2h': {
            'periodos': 24, 
            'minutos': 120, 
            'menciones': 5, 
            'prioridad': 'alta',
            'porcentaje_usuarios': 38
        },
        'planificacion_24h': {
            'periodos': 288, 
            'minutos': 1440, 
            'menciones': 6, 
            'prioridad': 'muy_alta',
            'porcentaje_usuarios': 46
        }
    }
    
    # Umbrales basados en requerimientos del 92% de confianza
    umbrales = {
        'verde_max': 25,
        'amarilla_max': 50,
        'roja_min': 50,
        'confianza_minima': 80
    }
    
    # Pesos de indicadores según prioridad mencionada
    pesos_indicadores = {
        'IVM': 30,  # 7 menciones
        'IET': 30,  # 6 menciones  
        'IIPNP': 25,  # 6 menciones
        'IPC': 10,  # 4 menciones
        'FDT': 5   # 3 menciones
    }
    
    # Precisión objetivo (>90% según requerimientos)
    precision_objetivo = 90