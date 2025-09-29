import logging
import os
from datetime import datetime
import os
import sys
from config.settings import ConfiguracionSistema
from config.logging_config import configurar_logging
from core.sistema_base import SistemaPredictivoBase
from core.tipos import EstadoSistema, ResultadoInicializacion, ResultadoPrediccion
from data.csv_parser_avanzado import CSVParserAvanzado
from formulas.calculador_industrial import CalculadorFormulasIndustriales
from models.prediction_models import GestorPredicciones
from models.validation import ValidadorModelos, ValidadorCalidadDatos
from reports.generador_reportes import GeneradorReportes
from monitoring.monitor_tiempo_real import MonitorTiempoReal

class SistemaPredictivoIndustrial(SistemaPredictivoBase):
    """Sistema predictivo industrial modularizado"""
    
    def inicializar_sistema_completo(self) -> ResultadoInicializacion:
        """Implementación específica de inicialización"""
        try:
            self.logger.info("=" * 80)
            self.logger.info("INICIANDO SISTEMA PREDICTIVO INDUSTRIAL MODULARIZADO")
            self.logger.info("=" * 80)
            
            # 1. Cargar datos
            resultado_carga = self._cargar_datos_historicos()
            if resultado_carga['estado'] != 'OK':
                return ResultadoInicializacion(
                    carga_datos=resultado_carga['estado'],
                    calidad_datos='SKIP', analisis_historico='SKIP',
                    entrenamiento='SKIP', validacion='SKIP', produccion='SKIP'
                )
            
            # 2. Validar calidad
            resultado_calidad = self._validar_calidad_datos()
            
            # 3. Análisis histórico
            resultado_historico = self._analizar_indicadores_historicos()
            
            # 4. Entrenar modelos
            resultado_entrenamiento = self._entrenar_modelos_predictivos()
            
            # 5. Validar modelos
            resultado_validacion = self._validar_modelos()
            
            # 6. Configurar producción
            resultado_produccion = self._configurar_produccion()
            
            resultado_final = ResultadoInicializacion(
                carga_datos=resultado_carga['estado'],
                calidad_datos=resultado_calidad['estado'],
                analisis_historico=resultado_historico['estado'],
                entrenamiento=resultado_entrenamiento['estado'],
                validacion=resultado_validacion['estado'],
                produccion=resultado_produccion['estado']
            )
            
            if resultado_final.esta_ok():
                self.estado.en_produccion = True
                self.logger.info("✅ SISTEMA COMPLETAMENTE OPERATIVO")
            
            return resultado_final
            
        except Exception as e:
            self.logger.error(f"Error crítico inicializando sistema: {e}")
            return ResultadoInicializacion(
                carga_datos='ERROR', calidad_datos='ERROR', analisis_historico='ERROR',
                entrenamiento='ERROR', validacion='ERROR', produccion='ERROR',
                error_critico=str(e)
            )
    
    def realizar_prediccion_completa(self, horizonte_horas: int = None) -> ResultadoPrediccion:
        """Implementación específica de predicción"""
        if horizonte_horas is None:
            horizonte_horas = self.config['horizonte_horas']
            
        if not self.estado.en_produccion:
            return ResultadoPrediccion(
                timestamp=datetime.now(),
                horizonte_horas=horizonte_horas,
                predicciones={},
                total_predicciones=0,
                reporte_usuario="",
                error="Sistema no está en producción"
            )
        
        try:
            # Usar datos recientes
            datos_recientes = self.datos_historicos.tail(100)
            
            # Realizar predicciones
            predicciones = self.gestor_predicciones.predecir_todos_indicadores(
                datos_recientes, horizonte_horas
            )
            
            # Generar reporte
            reporte = GeneradorReportes.generar_reporte_usuario_simple(
                predicciones, horizonte_horas
            )
            
            total_predicciones = len([p for p in predicciones.values() if p is not None])
            
            return ResultadoPrediccion(
                timestamp=datetime.now(),
                horizonte_horas=horizonte_horas,
                predicciones=predicciones,
                total_predicciones=total_predicciones,
                reporte_usuario=reporte
            )
            
        except Exception as e:
            return ResultadoPrediccion(
                timestamp=datetime.now(),
                horizonte_horas=horizonte_horas,
                predicciones={},
                total_predicciones=0,
                reporte_usuario="",
                error=str(e)
            )

def main():
    """Punto de entrada principal modularizado"""
    print("SISTEMA PREDICTIVO INDUSTRIAL MODULARIZADO")
    print("=" * 50)
    
    config = ConfiguracionSistema.get_config()
    ruta_datos = config['ruta_datos']
    
    if not os.path.exists(ruta_datos):
        print(f"ERROR: Archivo {ruta_datos} no encontrado")
        return
    
    try:
        # Inicializar sistema
        sistema = SistemaPredictivoIndustrial(ruta_datos, config)
        
        # Inicialización
        resultado_init = sistema.inicializar_sistema_completo()
        
        # Mostrar resultados
        print("\nRESULTADOS DE INICIALIZACIÓN:")
        print(f"  Carga datos: {resultado_init.carga_datos}")
        print(f"  Calidad datos: {resultado_init.calidad_datos}")
        print(f"  Entrenamiento: {resultado_init.entrenamiento}")
        print(f"  Validación: {resultado_init.validacion}")
        print(f"  Producción: {resultado_init.produccion}")
        
        # Si está operativo, hacer predicción
        if resultado_init.esta_ok():
            print("\nREALIZANDO PREDICCIÓN...")
            resultado_pred = sistema.realizar_prediccion_completa()
            
            if resultado_pred.es_exitosa():
                print("\nREPORTE:")
                print("-" * 40)
                print(resultado_pred.reporte_usuario)
                print("-" * 40)
                
                # Monitoreo opcional
                monitor = MonitorTiempoReal(sistema)
                monitor.iniciar_monitoreo()
        
        print("✅ Sistema ejecutado completamente")
        
    except Exception as e:
        print(f"ERROR: {e}")

if __name__ == "__main__":
    main()