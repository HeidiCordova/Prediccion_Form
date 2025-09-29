
import time
from datetime import datetime, timedelta

class MonitorTiempoReal:
    """Monitor de tiempo real para el sistema predictivo"""
    
    def __init__(self, sistema_predictivo, intervalo_minutos: int = 60):
        self.sistema = sistema_predictivo
        self.intervalo_minutos = intervalo_minutos
        self.logger = sistema_predictivo.logger
        self.activo = False
    
    def iniciar_monitoreo(self):
        """Iniciar monitoreo en tiempo real"""
        self.logger.info(f"Iniciando monitoreo cada {self.intervalo_minutos} minutos...")
        self.activo = True
        
        try:
            resultado = self.sistema.realizar_prediccion_completa(horizonte_horas=2)
            
            if resultado.es_exitosa():
                self._mostrar_reporte_automatico(resultado)
                self._verificar_anomalias(resultado)
            else:
                self.logger.error(f"Error en monitoreo: {resultado.error}")
            
            return resultado
            
        except Exception as e:
            self.logger.error(f"Error en monitoreo tiempo real: {e}")
            return {'error': str(e)}
    
    def _mostrar_reporte_automatico(self, resultado):
        """Mostrar reporte automático"""
        print("\n" + "="*60)
        print("REPORTE AUTOMÁTICO DEL SISTEMA")
        print("="*60)
        print(resultado.reporte_usuario)
        print("="*60)
    
    def _verificar_anomalias(self, resultado):
        """Verificar y reportar anomalías"""
        for nombre, pred in resultado.predicciones.items():
            if pred is not None and pred.es_anomalia:
                self.logger.warning(f"ANOMALÍA detectada en {nombre}: {pred.valor_predicho}")