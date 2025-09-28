#!/usr/bin/env python3
"""
SISTEMA COMPLETO DE PREDICCIÓN INDUSTRIAL
Integra análisis histórico + predicción + validación + monitoreo

Basado en necesidades identificadas:
- Predicción 15min-24h con 70-90%+ confianza
- Alertas en tiempo real
- Reducción de complejidad conceptual
- Prioridad: IVM, IET, IIPNP
"""

import os
import sys
import datetime
import numpy as np
import pandas as pd
import logging
import warnings
from typing import Dict, List, Optional

warnings.filterwarnings('ignore')

# Importar módulos desarrollados
from data.csv_parser_avanzado import CSVParserAvanzado
from formulas.calculador_industrial import CalculadorFormulasIndustrialesArreglado
from models.prediction_models import GestorPredicciones, ModeloPrediccionIVM, ModeloPrediccionIET, ModeloPrediccionIPC
from models.validation import ValidadorModelos, ValidadorCalidadDatos

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('sistema_predictivo.log'),
        logging.StreamHandler()
    ]
)

class SistemaPredictivoIndustrial:
    """Sistema completo de predicción industrial integrado"""
    
    def __init__(self, ruta_datos: str):
        self.ruta_datos = ruta_datos
        self.datos_historicos = None
        self.calculador_historico = None
        self.gestor_predicciones = None
        self.validador = None
        self.validador_calidad = None
        self.logger = logging.getLogger("SistemaPredictivoIndustrial")
        
        # Configuración basada en encuesta de usuarios
        self.config_usuarios = {
            'horizonte_min': 15,  # minutos mínimo
            'horizonte_max': 24,  # horas máximo
            'confianza_objetivo': 80.0,  # % promedio requerido
            'precision_minima': 70.0,  # % mínimo aceptable
            'indicadores_prioritarios': ['IVM', 'IET', 'IIPNP'],
            'frecuencia_alertas': 'tiempo_real',
            'nivel_detalle': 'recomendaciones_accion'  # vs 'tecnico'
        }
        
        self.estado_sistema = {
            'inicializado': False,
            'modelos_entrenados': False,
            'validacion_ok': False,
            'en_produccion': False
        }
    
    def inicializar_sistema_completo(self) -> Dict[str, str]:
        """Inicializar todo el sistema paso a paso"""
        try:
            self.logger.info("=" * 80)
            self.logger.info("INICIANDO SISTEMA PREDICTIVO INDUSTRIAL COMPLETO")
            self.logger.info("=" * 80)
            
            resultados = {}
            
            # 1. Cargar y analizar datos históricos
            self.logger.info("PASO 1: Cargando datos históricos...")
            resultado_carga = self._cargar_datos_historicos()
            resultados['carga_datos'] = resultado_carga['estado']
            
            if resultado_carga['estado'] != 'OK':
                return resultados
            
            # 2. Validar calidad de datos
            self.logger.info("PASO 2: Validando calidad de datos...")
            resultado_calidad = self._validar_calidad_datos()
            resultados['calidad_datos'] = resultado_calidad['estado']
            
            # 3. Análisis histórico de fórmulas
            self.logger.info("PASO 3: Analizando indicadores históricos...")
            resultado_historico = self._analizar_indicadores_historicos()
            resultados['analisis_historico'] = resultado_historico['estado']
            
            # 4. Entrenar modelos predictivos
            self.logger.info("PASO 4: Entrenando modelos predictivos...")
            resultado_entrenamiento = self._entrenar_modelos_predictivos()
            resultados['entrenamiento'] = resultado_entrenamiento['estado']
            
            # 5. Validar modelos
            self.logger.info("PASO 5: Validando modelos...")
            resultado_validacion = self._validar_modelos()
            resultados['validacion'] = resultado_validacion['estado']
            
            # 6. Configurar sistema en producción
            if all(r in ['OK', 'WARNING'] for r in resultados.values()):
                self.logger.info("PASO 6: Configurando sistema en producción...")
                resultado_produccion = self._configurar_produccion()
                resultados['produccion'] = resultado_produccion['estado']
                
                self.estado_sistema['en_produccion'] = True
                self.logger.info("✅ SISTEMA COMPLETAMENTE OPERATIVO")
            else:
                self.logger.error("❌ Sistema no pudo iniciarse completamente")
                resultados['produccion'] = 'ERROR'
            
            return resultados
            
        except Exception as e:
            self.logger.error(f"Error crítico inicializando sistema: {e}")
            return {'error_critico': str(e)}
    
    def _cargar_datos_historicos(self) -> Dict[str, str]:
        """Cargar datos históricos con parser avanzado"""
        try:
            # Usar parser avanzado desarrollado
            parser = CSVParserAvanzado(self.ruta_datos)
            self.datos_historicos = parser.cargar_csv_formato_industrial()
            
            self.logger.info(f"✅ Datos cargados: {len(self.datos_historicos)} registros, {len(self.datos_historicos.columns)} columnas")
            
            # Mostrar resumen
            resumen = parser.mostrar_resumen_datos()
            self.logger.info("Resumen de datos cargados:")
            self.logger.info(resumen)
            
            self.estado_sistema['inicializado'] = True
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
            
            if reporte_calidad['recomendaciones']:
                self.logger.info("Recomendaciones de calidad:")
                for rec in reporte_calidad['recomendaciones'][:3]:
                    self.logger.info(f"  - {rec}")
            
            if calidad_global >= 75:
                return {'estado': 'OK', 'calidad': calidad_global}
            elif calidad_global >= 60:
                return {'estado': 'WARNING', 'calidad': calidad_global}
            else:
                return {'estado': 'ERROR', 'calidad': calidad_global}
                
        except Exception as e:
            self.logger.error(f"Error validando calidad: {e}")
            return {'estado': 'ERROR', 'detalle': str(e)}
    
    def _analizar_indicadores_historicos(self) -> Dict[str, str]:
        """Analizar indicadores históricos como baseline"""
        try:
            self.calculador_historico = CalculadorFormulasIndustrialesArreglado(self.datos_historicos)
            resultados_historicos = self.calculador_historico.calcular_todas_las_formulas_reales()
            
            # Log resultados principales
            for indicador in ['IVM', 'IET', 'IPC']:
                if indicador in resultados_historicos:
                    resultado = resultados_historicos[indicador]
                    if 'error' not in resultado:
                        valor = resultado['valor']
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
                modelo.horizonte_horas = 2  # Compromiso entre 15min y 24h
            
            # Entrenar modelos
            resultados_entrenamiento = self.gestor_predicciones.entrenar_todos_modelos(self.datos_historicos)
            
            modelos_exitosos = 0
            for nombre, resultado in resultados_entrenamiento.items():
                if 'error' not in resultado:
                    precision = resultado.get('precisión', 0)
                    self.logger.info(f"Modelo {nombre} entrenado: {precision:.1f}% precisión")
                    if precision >= self.config_usuarios['precision_minima']:
                        modelos_exitosos += 1
                else:
                    self.logger.error(f"Error entrenando {nombre}: {resultado['error']}")
            
            if modelos_exitosos >= 2:  # Al menos 2 modelos funcionando
                self.estado_sistema['modelos_entrenados'] = True
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
                precision_objetivo=self.config_usuarios['confianza_objetivo'],
                confianza_minima=self.config_usuarios['precision_minima']
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
                    self.logger.info(f"✅ {nombre}: VALIDADO (Precisión: {resultado.precision_global}%)")
                else:
                    self.logger.warning(f"⚠️ {nombre}: REQUIERE ATENCIÓN")
                
                if resultado.alertas:
                    alertas_criticas += len(resultado.alertas)
                    for alerta in resultado.alertas:
                        self.logger.warning(f"  {alerta}")
            
            if modelos_validados >= 2 and alertas_criticas == 0:
                self.estado_sistema['validacion_ok'] = True
                return {'estado': 'OK', 'modelos_validados': modelos_validados}
            elif modelos_validados >= 1:
                return {'estado': 'WARNING', 'modelos_validados': modelos_validados}
            else:
                return {'estado': 'ERROR', 'modelos_validados': modelos_validados}
                
        except Exception as e:
            self.logger.error(f"Error validando modelos: {e}")
            return {'estado': 'ERROR', 'detalle': str(e)}
    
    def _configurar_produccion(self) -> Dict[str, str]:
        """Configurar sistema para producción"""
        try:
            self.logger.info("Configurando sistema para producción...")
            
            # Guardar modelos entrenados
            ruta_modelos = "modelos_entrenados"
            self.gestor_predicciones.guardar_modelos(ruta_modelos)
            
            # Configurar monitoreo automático
            # En un sistema real, esto configuraría tareas programadas
            
            self.logger.info("✅ Sistema configurado para producción")
            return {'estado': 'OK'}
            
        except Exception as e:
            self.logger.error(f"Error configurando producción: {e}")
            return {'estado': 'ERROR', 'detalle': str(e)}
    
   
    def realizar_prediccion_completa(self, horizonte_horas: int = 2) -> Dict[str, any]:
        """Realizar predicción completa con todos los indicadores - SOLO VALORES NUMÉRICOS"""
        if not self.estado_sistema['en_produccion']:
            raise ValueError("Sistema no está en producción. Ejecutar inicializar_sistema_completo() primero.")
        
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
            reporte = self._generar_reporte_usuario_simple(predicciones, horizonte_horas)
            
            resultado = {
                'timestamp': datetime.datetime.now(),
                'horizonte_horas': horizonte_horas,
                'predicciones': predicciones,
                'reporte_usuario': reporte,
                'total_predicciones': len([p for p in predicciones.values() if p is not None])
            }
            
            self.logger.info(f"Predicción completada - {resultado['total_predicciones']} indicadores predichos")
            
            return resultado
            
        except Exception as e:
            self.logger.error(f"Error en predicción completa: {e}")
            return {'error': str(e)}
    
   
    def _generar_reporte_usuario_simple(self, predicciones, horizonte_horas) -> str:
        """Generar reporte simple solo con valores predichos"""
        output = []
        
        # Encabezado simple
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
        output.append(f"PREDICCIÓN INDUSTRIAL - {timestamp}")
        output.append(f"Horizonte: {horizonte_horas} horas")
        output.append("")
        
        # Mostrar predicciones obtenidas
        output.append("VALORES PREDICHOS:")
        output.append("-" * 30)
        
        for indicador in ['IVM', 'IET', 'IPC']:
            if indicador in predicciones and predicciones[indicador] is not None:
                pred = predicciones[indicador]
                anomalia_tag = " (ANOMALÍA)" if pred.es_anomalia else ""
                output.append(f"{indicador}: {pred.valor_predicho:.2f} (Confianza: {pred.confianza:.1f}%){anomalia_tag}")
            else:
                output.append(f"{indicador}: No disponible")
        
        # Próxima actualización
        proxima_actualizacion = datetime.datetime.now() + datetime.timedelta(hours=1)
        output.append("")
        output.append(f"Próxima actualización: {proxima_actualizacion.strftime('%H:%M')}")
        
        return '\n'.join(output)
    

    def monitoreo_tiempo_real(self, intervalo_minutos: int = 60):
        """Monitoreo en tiempo real simplificado - solo valores"""
        self.logger.info(f"Iniciando monitoreo cada {intervalo_minutos} minutos...")
        
        try:
            resultado = self.realizar_prediccion_completa(horizonte_horas=2)
            
            if 'error' not in resultado:
                # Mostrar reporte simple
                print("\n" + "="*60)
                print("REPORTE AUTOMÁTICO DEL SISTEMA")
                print("="*60)
                print(resultado['reporte_usuario'])
                print("="*60)
                
                # Log adicional de predicciones con anomalías
                for nombre, pred in resultado['predicciones'].items():
                    if pred is not None and pred.es_anomalia:
                        self.logger.warning(f"ANOMALÍA detectada en {nombre}: {pred.valor_predicho}")
            else:
                self.logger.error(f"Error en monitoreo: {resultado['error']}")
            
            return resultado
            
        except Exception as e:
            self.logger.error(f"Error en monitoreo tiempo real: {e}")
            return {'error': str(e)}
    
    
    
    def generar_reporte_completo_sistema(self) -> str:
        """Generar reporte completo del estado del sistema"""
        output = []
        
        output.append("=" * 80)
        output.append("REPORTE COMPLETO DEL SISTEMA PREDICTIVO INDUSTRIAL")
        output.append(f"Generado: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        output.append("=" * 80)
        
        # Estado del sistema
        output.append("\n[ESTADO DEL SISTEMA]")
        for componente, estado in self.estado_sistema.items():
            emoji = "✅" if estado else "❌"
            output.append(f"  {emoji} {componente.replace('_', ' ').title()}: {'OK' if estado else 'PENDIENTE'}")
        
        # Datos históricos
        if self.datos_historicos is not None:
            output.append(f"\n[DATOS HISTÓRICOS]")
            output.append(f"  Total registros: {len(self.datos_historicos):,}")
            output.append(f"  Columnas: {len(self.datos_historicos.columns)}")
            
            if 'Fecha' in self.datos_historicos.columns:
                fecha_min = self.datos_historicos['Fecha'].min()
                fecha_max = self.datos_historicos['Fecha'].max()
                output.append(f"  Período: {fecha_min.strftime('%Y-%m-%d')} a {fecha_max.strftime('%Y-%m-%d')}")
        
        # Configuración de usuarios
        output.append(f"\n[CONFIGURACIÓN USUARIOS]")
        output.append(f"  Horizonte mínimo: {self.config_usuarios['horizonte_min']} minutos")
        output.append(f"  Horizonte máximo: {self.config_usuarios['horizonte_max']} horas")
        output.append(f"  Confianza objetivo: {self.config_usuarios['confianza_objetivo']}%")
        output.append(f"  Precisión mínima: {self.config_usuarios['precision_minima']}%")
        output.append(f"  Indicadores prioritarios: {', '.join(self.config_usuarios['indicadores_prioritarios'])}")
        
        # Estado modelos
        if self.gestor_predicciones:
            output.append(f"\n[MODELOS PREDICTIVOS]")
            for nombre, modelo in self.gestor_predicciones.modelos.items():
                estado_modelo = "✅ ENTRENADO" if modelo.modelo_entrenado else "❌ NO ENTRENADO"
                output.append(f"  {nombre}: {estado_modelo}")
                
                if modelo.modelo_entrenado and hasattr(modelo, 'metricas_entrenamiento'):
                    precision = modelo.metricas_entrenamiento.get('precisión', 0)
                    output.append(f"    Precisión: {precision:.1f}%")
        
        return '\n'.join(output)

def main():
    """Función principal para ejecutar sistema completo"""
    print("🏭 SISTEMA PREDICTIVO INDUSTRIAL AVANZADO")
    print("=" * 60)
    
    # Archivo de datos
    ruta_datos = "datos.csv"  # Ajustar según ubicación real
    
    if not os.path.exists(ruta_datos):
        print(f"❌ ERROR: Archivo {ruta_datos} no encontrado")
        return
    
    try:
        # Inicializar sistema
        sistema = SistemaPredictivoIndustrial(ruta_datos)
        
        # Inicialización completa
        print("\n🚀 Inicializando sistema completo...")
        resultados_init = sistema.inicializar_sistema_completo()
        
        # Mostrar resultados de inicialización
        print("\n📊 RESULTADOS DE INICIALIZACIÓN:")
        for etapa, estado in resultados_init.items():
            emoji = "✅" if estado == 'OK' else "⚠️" if estado == 'WARNING' else "❌"
            print(f"  {emoji} {etapa.replace('_', ' ').title()}: {estado}")
        
        # Si sistema está operativo, hacer demostración
        if sistema.estado_sistema['en_produccion']:
            print("\n🎯 REALIZANDO PREDICCIÓN DE DEMOSTRACIÓN...")
            
            # Predicción a 2 horas
            resultado_prediccion = sistema.realizar_prediccion_completa(horizonte_horas=2)
            
            if 'error' not in resultado_prediccion:
                print("\n📱 REPORTE PARA USUARIO FINAL:")
                print("-" * 40)
                print(resultado_prediccion['reporte_usuario'])
                print("-" * 40)
                
                # Simular monitoreo en tiempo real
                print("\n🔄 INICIANDO MONITOREO AUTOMÁTICO...")
                sistema.monitoreo_tiempo_real(intervalo_minutos=60)
            else:
                print(f"❌ Error en predicción: {resultado_prediccion['error']}")
        
        # Reporte completo del sistema
        print("\n📋 GENERANDO REPORTE COMPLETO DEL SISTEMA...")
        reporte_completo = sistema.generar_reporte_completo_sistema()
        
        # Guardar reporte
        with open(f"reporte_sistema_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.txt", 'w', encoding='utf-8') as f:
            f.write(reporte_completo)
        
        print("✅ Sistema ejecutado completamente. Revisar logs para detalles.")
        
    except Exception as e:
        print(f"❌ ERROR CRÍTICO: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()