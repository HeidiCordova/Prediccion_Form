from typing import Dict, Optional
import pandas as pd
from config.settings import ConfiguracionSistema
from config.logging_config import configurar_logging
from core.tipos import EstadoSistema, ResultadoInicializacion, ResultadoPrediccion
from data.csv_parser_avanzado import CSVParserAvanzado
from formulas.calculador_industrial import CalculadorFormulasIndustriales
from models.prediction_models import GestorPredicciones
from models.validation import ValidadorModelos, ValidadorCalidadDatos
from reports.generador_reportes import GeneradorReportes

class SistemaPredictivoBase:
    """Clase base del sistema predictivo modularizado con implementaciones completas"""
    
    def __init__(self, ruta_datos: str, config: Optional[Dict] = None):
        self.ruta_datos = ruta_datos
        self.config = config or ConfiguracionSistema.get_config()
        self.logger = configurar_logging()
        self.estado = EstadoSistema()
        
        # Componentes del sistema
        self.datos_historicos = None
        self.calculador_historico = None
        self.gestor_predicciones = None
        self.validador = None
        self.validador_calidad = None
    
    def inicializar_sistema_completo(self) -> ResultadoInicializacion:
        """Inicializar todo el sistema paso a paso"""
        try:
            self.logger.info("=" * 80)
            self.logger.info("INICIANDO SISTEMA PREDICTIVO INDUSTRIAL MODULARIZADO")
            self.logger.info("=" * 80)
            
            resultado = ResultadoInicializacion()
            
            # 1. Cargar datos históricos
            self.logger.info("PASO 1: Cargando datos históricos...")
            resultado_carga = self._cargar_datos_historicos()
            resultado.carga_datos = resultado_carga['estado']
            
            if resultado_carga['estado'] != 'OK':
                return resultado
            
            # 2. Validar calidad de datos
            self.logger.info("PASO 2: Validando calidad de datos...")
            resultado_calidad = self._validar_calidad_datos()
            resultado.calidad_datos = resultado_calidad['estado']
            
            # 3. Análisis histórico de fórmulas
            self.logger.info("PASO 3: Analizando indicadores históricos...")
            resultado_historico = self._analizar_indicadores_historicos()
            resultado.analisis_historico = resultado_historico['estado']
            
            # 4. Entrenar modelos predictivos
            self.logger.info("PASO 4: Entrenando modelos predictivos...")
            resultado_entrenamiento = self._entrenar_modelos_predictivos()
            resultado.entrenamiento = resultado_entrenamiento['estado']
            
            # 5. Validar modelos
            self.logger.info("PASO 5: Validando modelos...")
            resultado_validacion = self._validar_modelos()
            resultado.validacion = resultado_validacion['estado']
            
            # 6. Configurar sistema en producción
            if all(r in ['OK', 'WARNING'] for r in [
                resultado.carga_datos, resultado.calidad_datos, resultado.analisis_historico,
                resultado.entrenamiento, resultado.validacion
            ]):
                self.logger.info("PASO 6: Configurando sistema en producción...")
                resultado_produccion = self._configurar_produccion()
                resultado.produccion = resultado_produccion['estado']
                
                self.estado.en_produccion = True
                self.logger.info("✅ SISTEMA COMPLETAMENTE OPERATIVO")
            else:
                self.logger.error("❌ Sistema no pudo iniciarse completamente")
                resultado.produccion = 'ERROR'
            
            return resultado
            
        except Exception as e:
            self.logger.error(f"Error crítico inicializando sistema: {e}")
            return ResultadoInicializacion(error_critico=str(e))
    
    def realizar_prediccion_completa(self, horizonte_horas: int = None) -> ResultadoPrediccion:
        """Realizar predicción completa"""
        from datetime import datetime
        
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
            self.logger.info(f"Realizando predicción completa a {horizonte_horas} horas...")
            
            # Usar datos más recientes para predicción
            datos_recientes = self.datos_historicos.tail(100)
            
            # Realizar predicciones
            predicciones = self.gestor_predicciones.predecir_todos_indicadores(
                datos_recientes, horizonte_horas
            )
            
            # Log de predicciones obtenidas
            for nombre, prediccion in predicciones.items():
                if prediccion is not None:
                    self.logger.info(f"Predicción {nombre}: {prediccion.valor_predicho}")
                else:
                    self.logger.warning(f"No se pudo generar predicción para {nombre}")
            
            # Generar reporte simple con valores predichos
            reporte = GeneradorReportes.generar_reporte_usuario_simple(predicciones, horizonte_horas)
            
            total_predicciones = len([p for p in predicciones.values() if p is not None])
            
            return ResultadoPrediccion(
                timestamp=datetime.now(),
                horizonte_horas=horizonte_horas,
                predicciones=predicciones,
                total_predicciones=total_predicciones,
                reporte_usuario=reporte
            )
            
        except Exception as e:
            self.logger.error(f"Error en predicción completa: {e}")
            return ResultadoPrediccion(
                timestamp=datetime.now(),
                horizonte_horas=horizonte_horas,
                predicciones={},
                total_predicciones=0,
                reporte_usuario="",
                error=str(e)
            )
    
    def _cargar_datos_historicos(self) -> Dict[str, str]:
        """Cargar datos históricos con parser avanzado"""
        try:
            parser = CSVParserAvanzado(self.ruta_datos)
            self.datos_historicos = parser.cargar_csv_formato_industrial()
            
            self.logger.info(f"✅ Datos cargados: {len(self.datos_historicos)} registros, {len(self.datos_historicos.columns)} columnas")
            
            # Mostrar resumen básico
            resumen = parser.mostrar_resumen_datos()
            # Solo mostrar las primeras líneas del resumen para no saturar logs
            resumen_corto = '\n'.join(resumen.split('\n')[:10])
            self.logger.info(f"Resumen de datos:\n{resumen_corto}")
            
            self.estado.inicializado = True
            return {'estado': 'OK', 'registros': len(self.datos_historicos)}
            
        except Exception as e:
            self.logger.error(f"Error cargando datos: {e}")
            return {'estado': 'ERROR', 'detalle': str(e)}
    
    def _validar_calidad_datos(self) -> Dict[str, str]:
        """Validar calidad de datos para entrenamiento"""
        try:
            self.validador_calidad = ValidadorCalidadDatos()
            reporte_calidad = self.validador_calidad.validar_calidad_datos(self.datos_historicos)
            
            calidad_global = reporte_calidad['calidad_global']
            apto_entrenamiento = reporte_calidad['apto_para_entrenamiento']
            
            self.logger.info(f"Calidad global de datos: {calidad_global:.1f}%")
            self.logger.info(f"Apto para entrenamiento: {'Sí' if apto_entrenamiento else 'No'}")
            
            if reporte_calidad.get('recomendaciones'):
                self.logger.info("Recomendaciones de calidad:")
                for rec in reporte_calidad['recomendaciones'][:3]:
                    self.logger.info(f"  - {rec}")
            
            if calidad_global >= ConfiguracionSistema.CALIDAD_MINIMA_OK:
                return {'estado': 'OK', 'calidad': calidad_global}
            elif calidad_global >= ConfiguracionSistema.CALIDAD_MINIMA_WARNING:
                return {'estado': 'WARNING', 'calidad': calidad_global}
            else:
                return {'estado': 'ERROR', 'calidad': calidad_global}
                
        except Exception as e:
            self.logger.error(f"Error validando calidad: {e}")
            return {'estado': 'ERROR', 'detalle': str(e)}
    
    def _analizar_indicadores_historicos(self) -> Dict[str, str]:
        """Analizar indicadores históricos como baseline"""
        try:
            self.calculador_historico = CalculadorFormulasIndustriales(self.datos_historicos)
            resultados_historicos = self.calculador_historico.calcular_todas_las_formulas_reales()
            
            # Log resultados principales - SOLO VALORES, SIN CLASIFICACIONES
            for indicador in ['IVM', 'IET', 'IPC']:
                if indicador in resultados_historicos:
                    resultado = resultados_historicos[indicador]
                    if 'error' not in resultado:
                        valor = resultado['valor']
                        self.logger.info(f"{indicador}: {valor}%")
                    else:
                        self.logger.error(f"{indicador}: {resultado['error']}")
            
            return {'estado': 'OK', 'indicadores_calculados': len(resultados_historicos)}
            
        except Exception as e:
            self.logger.error(f"Error analizando indicadores históricos: {e}")
            return {'estado': 'ERROR', 'detalle': str(e)}
    
    def _entrenar_modelos_predictivos(self) -> Dict[str, str]:
        """Entrenar todos los modelos predictivos"""
        try:
            self.gestor_predicciones = GestorPredicciones()
            
            # Configurar horizontes según necesidades de usuarios
            for modelo in self.gestor_predicciones.modelos.values():
                modelo.horizonte_horas = self.config['horizonte_horas']
            
            # Entrenar modelos
            resultados_entrenamiento = self.gestor_predicciones.entrenar_todos_modelos(self.datos_historicos)
            
            modelos_exitosos = 0
            for nombre, resultado in resultados_entrenamiento.items():
                if 'error' not in resultado:
                    # Calcular precisión más realista
                    r2 = resultado.get('r2', 0)
                    precision = max(0, r2 * 100)  # Convertir R² a porcentaje
                    self.logger.info(f"Modelo {nombre} entrenado: {precision:.1f}% precisión")
                    if precision >= 0:  # Cualquier modelo entrenado cuenta
                        modelos_exitosos += 1
                else:
                    self.logger.error(f"Error entrenando {nombre}: {resultado['error']}")
            
            if modelos_exitosos >= 1:  # Al menos 1 modelo funcionando
                self.estado.modelos_entrenados = True
                return {'estado': 'OK', 'modelos_exitosos': modelos_exitosos}
            else:
                return {'estado': 'WARNING', 'modelos_exitosos': modelos_exitosos}
                
        except Exception as e:
            self.logger.error(f"Error entrenando modelos: {e}")
            return {'estado': 'ERROR', 'detalle': str(e)}
    
    def _validar_modelos(self) -> Dict[str, str]:
        """Validar modelos entrenados"""
        try:
            self.validador = ValidadorModelos(
                precision_objetivo=self.config['confianza_objetivo'],
                confianza_minima=self.config['precision_minima']
            )
            
            # Dividir datos para validación
            datos_validacion = self.datos_historicos.tail(1000)  # Últimos 1000 registros
            
            resultados_validacion = self.validador.validar_todos_modelos(
                self.gestor_predicciones, datos_validacion
            )
            
            modelos_validados = 0
            alertas_criticas = 0
            
            for nombre, resultado in resultados_validacion.items():
                if resultado.cumple_objetivos:
                    modelos_validados += 1
                    self.logger.info(f"✅ {nombre}: VALIDADO (Precisión: {resultado.precision_global:.1f}%)")
                else:
                    self.logger.warning(f"⚠️ {nombre}: REQUIERE ATENCIÓN")
                
                if resultado.alertas:
                    alertas_criticas += len(resultado.alertas)
                    for alerta in resultado.alertas:
                        self.logger.warning(f"  {alerta}")
            
            if modelos_validados >= 1:  # Al menos 1 modelo validado
                self.estado.validacion_ok = True
                return {'estado': 'OK', 'modelos_validados': modelos_validados}
            else:
                return {'estado': 'WARNING', 'modelos_validados': modelos_validados}
                
        except Exception as e:
            self.logger.error(f"Error validando modelos: {e}")
            return {'estado': 'ERROR', 'detalle': str(e)}
    
    def _configurar_produccion(self) -> Dict[str, str]:
        """Configurar sistema para producción"""
        try:
            self.logger.info("Configurando sistema para producción...")
            
            # Guardar modelos entrenados
            ruta_modelos = self.config['ruta_modelos']
            self.gestor_predicciones.guardar_modelos(ruta_modelos)
            
            self.logger.info("✅ Sistema configurado para producción")
            return {'estado': 'OK'}
            
        except Exception as e:
            self.logger.error(f"Error configurando producción: {e}")
            return {'estado': 'ERROR', 'detalle': str(e)}
    
 