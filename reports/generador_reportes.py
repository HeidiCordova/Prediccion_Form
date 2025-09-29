
from datetime import datetime
from typing import Dict, List
class GeneradorReportes:
    """Generador de reportes del sistema"""
    
    @staticmethod
    def generar_reporte_usuario_simple(predicciones: Dict, horizonte_horas: int) -> str:
        """Generar reporte simple para usuarios finales"""
        output = []
        
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
        output.append(f"PREDICCIÓN INDUSTRIAL - {timestamp}")
        output.append(f"Horizonte: {horizonte_horas} horas")
        output.append("")
        
        output.append("VALORES PREDICHOS:")
        output.append("-" * 30)
        
        for indicador in ['IVM', 'IET', 'IPC']:
            if indicador in predicciones and predicciones[indicador] is not None:
                pred = predicciones[indicador]
                anomalia_tag = " (ANOMALÍA)" if pred.es_anomalia else ""
                output.append(f"{indicador}: {pred.valor_predicho:.2f} (Confianza: {pred.confianza:.1f}%){anomalia_tag}")
            else:
                output.append(f"{indicador}: No disponible")
        
        proxima_actualizacion = datetime.datetime.now() + datetime.timedelta(hours=1)
        output.append("")
        output.append(f"Próxima actualización: {proxima_actualizacion.strftime('%H:%M')}")
        
        return '\n'.join(output)