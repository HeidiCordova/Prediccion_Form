from dataclasses import dataclass
from typing import Dict, List, Optional
from datetime import datetime

@dataclass
class EstadoSistema:
    """Estado del sistema predictivo"""
    inicializado: bool = False
    modelos_entrenados: bool = False
    validacion_ok: bool = False
    en_produccion: bool = False
    
    def esta_operativo(self) -> bool:
        return self.en_produccion

@dataclass
class ResultadoInicializacion:
    """Resultado del proceso de inicialización"""
    carga_datos: str = "PENDIENTE"
    calidad_datos: str = "PENDIENTE"
    analisis_historico: str = "PENDIENTE"
    entrenamiento: str = "PENDIENTE"
    validacion: str = "PENDIENTE"
    produccion: str = "PENDIENTE"
    error_critico: Optional[str] = None
    
    def esta_ok(self) -> bool:
        return all(estado in ['OK', 'WARNING'] for estado in [
            self.carga_datos, self.calidad_datos, self.analisis_historico,
            self.entrenamiento, self.validacion, self.produccion
        ]) and self.error_critico is None

# AÑADIR ESTA CLASE:
@dataclass
class ResultadoPrediccion:
    """Resultado de una predicción completa"""
    timestamp: datetime
    horizonte_horas: int
    predicciones: Dict
    total_predicciones: int
    reporte_usuario: str
    error: Optional[str] = None
    
    def es_exitosa(self) -> bool:
        """Verificar si la predicción fue exitosa"""
        return self.error is None and self.total_predicciones > 0