import os
import sys
import numpy as np
import pandas as pd
import datetime
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
import warnings
warnings.filterwarnings('ignore')

# Configurar logging
logging.basicConfig(level=logging.INFO)

@dataclass
class ResultadoValidacion:
    """Resultado estructurado de validación"""
    precision_global: float
    confianza_promedio: float
    error_absoluto_promedio: float
    r2_score: float
    validacion_cruzada_scores: List[float]
    metricas_por_turno: Dict[str, float]
    metricas_por_producto: Dict[str, float]
    degradacion_detectada: bool
    cumple_objetivos: bool
    timestamp: datetime.datetime
    recomendaciones: List[str]
    alertas: List[str]

class ValidadorModelos:
    """Validador principal para modelos predictivos industriales"""
    
    def __init__(self, precision_objetivo: float = 80.0, confianza_minima: float = 70.0):
        self.precision_objetivo = precision_objetivo
        self.confianza_minima = confianza_minima
        self.logger = logging.getLogger("ValidadorModelos")
        self.historial_validaciones = []
        
        # Umbrales de alerta
        self.umbrales = {
            'precision_critica': 60.0,
            'degradacion_maxima': 10.0,  # % de pérdida respecto al mejor resultado
            'variabilidad_maxima': 15.0,  # CV máximo aceptable
            'error_maximo_relativo': 25.0  # % de error máximo aceptable
        }
    
    def validar_modelo_completo(self, modelo, datos_test: pd.DataFrame, datos_validacion_temporal: pd.DataFrame = None) -> ResultadoValidacion:
        """Validación completa de un modelo predictivo"""
        try:
            self.logger.info(f"Iniciando validación completa del modelo {modelo.nombre_indicador}")
            
            # 1. Validación básica de precisión
            precision_global = self._validar_precision_basica(modelo, datos_test)
            
            # 2. Validación cruzada temporal
            scores_cv = self._validacion_cruzada_temporal(modelo, datos_test)
            
            # 3. Validación por segmentos
            metricas_turnos = self._validar_por_turnos(modelo, datos_test)
            metricas_productos = self._validar_por_productos(modelo, datos_test)
            
            # 4. Detección de degradación
            degradacion = self._detectar_degradacion_modelo(modelo, datos_test, datos_validacion_temporal)
            
            # 5. Cálculo de métricas agregadas
            confianza_promedio = np.mean(scores_cv) if scores_cv else precision_global
            error_absoluto = self._calcular_error_absoluto_promedio(modelo, datos_test)
            r2 = self._calcular_r2_score(modelo, datos_test)
            
            # 6. Evaluación de cumplimiento de objetivos
            cumple_objetivos = (
                precision_global >= self.precision_objetivo and
                confianza_promedio >= self.confianza_minima and
                not degradacion
            )
            
            # 7. Generar recomendaciones y alertas
            recomendaciones = self._generar_recomendaciones(precision_global, confianza_promedio, degradacion)
            alertas = self._generar_alertas(precision_global, confianza_promedio, degradacion)
            
            resultado = ResultadoValidacion(
                precision_global=round(precision_global, 2),
                confianza_promedio=round(confianza_promedio, 2),
                error_absoluto_promedio=round(error_absoluto, 2),
                r2_score=round(r2, 3),
                validacion_cruzada_scores=scores_cv,
                metricas_por_turno=metricas_turnos,
                metricas_por_producto=metricas_productos,
                degradacion_detectada=degradacion,
                cumple_objetivos=cumple_objetivos,
                timestamp=datetime.datetime.now(),
                recomendaciones=recomendaciones,
                alertas=alertas
            )
            
            # Guardar en historial
            self.historial_validaciones.append({
                'modelo': modelo.nombre_indicador,
                'timestamp': resultado.timestamp,
                'precision': resultado.precision_global,
                'confianza': resultado.confianza_promedio,
                'cumple_objetivos': resultado.cumple_objetivos
            })
            
            self.logger.info(f"Validación completada. Precisión: {precision_global:.1f}%, "
                           f"Confianza: {confianza_promedio:.1f}%, Cumple objetivos: {cumple_objetivos}")
            
            return resultado
            
        except Exception as e:
            self.logger.error(f"Error en validación completa: {e}")
            raise
    
    def _validar_precision_basica(self, modelo, datos_test: pd.DataFrame) -> float:
        """Validar precisión básica del modelo"""
        try:
            if not modelo.modelo_entrenado:
                return 0.0
            
            # Realizar predicciones en datos de test
            predicciones_realizadas = []
            valores_reales = []
            
            # Simulación de predicciones (adaptar según la estructura real del modelo)
            for i in range(min(100, len(datos_test))):  # Limitar para eficiencia
                try:
                    muestra = datos_test.iloc[i:i+1]
                    prediccion = modelo.predecir(muestra)
                    
                    # Obtener valor real basado en el tipo de indicador
                    valor_real = self._obtener_valor_real(modelo.nombre_indicador, muestra)
                    
                    if valor_real is not None:
                        predicciones_realizadas.append(prediccion.valor_predicho)
                        valores_reales.append(valor_real)
                        
                except Exception as e:
                    continue  # Saltar muestras problemáticas
            
            if len(predicciones_realizadas) == 0:
                return 0.0
            
            # Calcular precisión basada en error relativo
            errores_relativos = []
            for real, pred in zip(valores_reales, predicciones_realizadas):
                if real != 0:
                    error_relativo = abs((real - pred) / real) * 100
                    errores_relativos.append(error_relativo)
            
            if not errores_relativos:
                return 0.0
            
            error_promedio = np.mean(errores_relativos)
            precision = max(0, 100 - error_promedio)
            
            return precision
            
        except Exception as e:
            self.logger.error(f"Error calculando precisión básica: {e}")
            return 0.0
    
    def _validacion_cruzada_temporal(self, modelo, datos: pd.DataFrame) -> List[float]:
        """Validación cruzada considerando la naturaleza temporal de los datos"""
        try:
            scores = []
            
            # Dividir datos en ventanas temporales
            if 'Fecha' in datos.columns:
                datos_ordenados = datos.sort_values('Fecha')
                n_splits = min(5, len(datos_ordenados) // 100)  # Máximo 5 splits
                
                if n_splits < 2:
                    return []
                
                tscv = TimeSeriesSplit(n_splits=n_splits)
                
                for train_idx, test_idx in tscv.split(datos_ordenados):
                    try:
                        datos_train = datos_ordenados.iloc[train_idx]
                        datos_test = datos_ordenados.iloc[test_idx]
                        
                        # Re-entrenar modelo en datos de entrenamiento
                        modelo_temp = type(modelo)()
                        modelo_temp.entrenar(datos_train)
                        
                        # Evaluar en datos de test
                        precision_fold = self._validar_precision_basica(modelo_temp, datos_test)
                        scores.append(precision_fold)
                        
                    except Exception as e:
                        continue
            
            return scores
            
        except Exception as e:
            self.logger.error(f"Error en validación cruzada temporal: {e}")
            return []
    
    def _validar_por_turnos(self, modelo, datos: pd.DataFrame) -> Dict[str, float]:
        """Validar rendimiento del modelo por turnos"""
        metricas_turnos = {}
        
        try:
            if 'Id Turno' in datos.columns:
                turnos_unicos = datos['Id Turno'].dropna().unique()
                
                for turno in turnos_unicos:
                    datos_turno = datos[datos['Id Turno'] == turno]
                    if len(datos_turno) >= 10:  # Mínimo de datos por turno
                        precision_turno = self._validar_precision_basica(modelo, datos_turno)
                        metricas_turnos[f"Turno_{int(turno)}"] = round(precision_turno, 2)
            
        except Exception as e:
            self.logger.error(f"Error validando por turnos: {e}")
        
        return metricas_turnos
    
    def _validar_por_productos(self, modelo, datos: pd.DataFrame) -> Dict[str, float]:
        """Validar rendimiento del modelo por productos"""
        metricas_productos = {}
        
        try:
            if 'Id Producto' in datos.columns:
                productos_unicos = datos['Id Producto'].dropna().unique()[:10]  # Limitar a 10 productos
                
                for producto in productos_unicos:
                    datos_producto = datos[datos['Id Producto'] == producto]
                    if len(datos_producto) >= 10:
                        precision_producto = self._validar_precision_basica(modelo, datos_producto)
                        metricas_productos[f"Producto_{int(producto)}"] = round(precision_producto, 2)
        
        except Exception as e:
            self.logger.error(f"Error validando por productos: {e}")
        
        return metricas_productos
    
    def _detectar_degradacion_modelo(self, modelo, datos_actuales: pd.DataFrame, datos_historicos: pd.DataFrame = None) -> bool:
        """Detectar si el modelo ha degradado su rendimiento"""
        try:
            # Verificar degradación basada en historial
            if len(self.historial_validaciones) >= 3:
                precisiones_recientes = [v['precision'] for v in self.historial_validaciones[-3:] if v['modelo'] == modelo.nombre_indicador]
                
                if len(precisiones_recientes) >= 2:
                    mejor_precision = max(precisiones_recientes)
                    precision_actual = precisiones_recientes[-1]
                    
                    degradacion_porcentual = ((mejor_precision - precision_actual) / mejor_precision) * 100
                    
                    if degradacion_porcentual > self.umbrales['degradacion_maxima']:
                        return True
            
            # Verificar variabilidad excesiva
            if datos_historicos is not None and len(datos_historicos) > 50:
                precision_historica = self._validar_precision_basica(modelo, datos_historicos)
                precision_actual = self._validar_precision_basica(modelo, datos_actuales)
                
                variabilidad = abs(precision_historica - precision_actual)
                if variabilidad > self.umbrales['variabilidad_maxima']:
                    return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error detectando degradación: {e}")
            return False
    
    def _obtener_valor_real(self, indicador: str, datos: pd.DataFrame) -> Optional[float]:
        """Obtener valor real del indicador para comparar con predicción"""
        try:
            if indicador == 'IVM':
                if 'T. de Microparadas' in datos.columns:
                    microparadas = datos['T. de Microparadas'].iloc[0]
                    # Simplificación: usar valor actual como proxy del valor real
                    return microparadas * 0.5  # Factor de ajuste
                    
            elif indicador == 'IET':
                if 'T. Disponible' in datos.columns and 'T. de Microparadas' in datos.columns:
                    disponible = datos['T. Disponible'].iloc[0]
                    microparadas = datos['T. de Microparadas'].iloc[0]
                    if disponible > 0:
                        return ((disponible - microparadas) / disponible) * 100
                        
            elif indicador == 'IPC':
                if 'Conteo' in datos.columns:
                    conteo = datos['Conteo'].iloc[0]
                    return conteo * 0.8  # Factor de ajuste
            
            return None
            
        except Exception as e:
            return None
    
    def _calcular_error_absoluto_promedio(self, modelo, datos: pd.DataFrame) -> float:
        """Calcular error absoluto promedio"""
        try:
            errores = []
            for i in range(min(50, len(datos))):
                muestra = datos.iloc[i:i+1]
                prediccion = modelo.predecir(muestra)
                valor_real = self._obtener_valor_real(modelo.nombre_indicador, muestra)
                
                if valor_real is not None:
                    error = abs(prediccion.valor_predicho - valor_real)
                    errores.append(error)
            
            return np.mean(errores) if errores else 0.0
            
        except Exception as e:
            return 0.0
    
    def _calcular_r2_score(self, modelo, datos: pd.DataFrame) -> float:
        """Calcular R² score"""
        try:
            predicciones = []
            valores_reales = []
            
            for i in range(min(100, len(datos))):
                muestra = datos.iloc[i:i+1]
                prediccion = modelo.predecir(muestra)
                valor_real = self._obtener_valor_real(modelo.nombre_indicador, muestra)
                
                if valor_real is not None:
                    predicciones.append(prediccion.valor_predicho)
                    valores_reales.append(valor_real)
            
            if len(predicciones) >= 10:
                return r2_score(valores_reales, predicciones)
            else:
                return 0.0
                
        except Exception as e:
            return 0.0
    
    def _generar_recomendaciones(self, precision: float, confianza: float, degradacion: bool) -> List[str]:
        """Generar recomendaciones basadas en resultados de validación"""
        recomendaciones = []
        
        if precision < self.umbrales['precision_critica']:
            recomendaciones.append("CRÍTICO: Re-entrenar modelo con más datos")
            recomendaciones.append("Revisar calidad de datos de entrada")
            recomendaciones.append("Considerar cambio de algoritmo")
        
        elif precision < self.precision_objetivo:
            recomendaciones.append("Optimizar hiperparámetros del modelo")
            recomendaciones.append("Aumentar datos de entrenamiento")
            recomendaciones.append("Implementar feature engineering adicional")
        
        if confianza < self.confianza_minima:
            recomendaciones.append("Mejorar estabilidad del modelo")
            recomendaciones.append("Implementar ensemble de modelos")
        
        if degradacion:
            recomendaciones.append("Re-entrenar modelo inmediatamente")
            recomendaciones.append("Investigar cambios en patrones de datos")
            recomendaciones.append("Implementar monitoreo continuo")
        
        if not recomendaciones:
            recomendaciones.append("Modelo funcionando correctamente")
            recomendaciones.append("Mantener monitoreo regular")
        
        return recomendaciones
    
    def _generar_alertas(self, precision: float, confianza: float, degradacion: bool) -> List[str]:
        """Generar alertas críticas"""
        alertas = []
        
        if precision < self.umbrales['precision_critica']:
            alertas.append(f"🔴 ALERTA CRÍTICA: Precisión muy baja ({precision:.1f}%)")
        
        if confianza < self.confianza_minima:
            alertas.append(f"🟡 ALERTA: Confianza insuficiente ({confianza:.1f}%)")
        
        if degradacion:
            alertas.append("🔴 ALERTA CRÍTICA: Degradación del modelo detectada")
        
        return alertas
    
    def validar_todos_modelos(self, gestor_predicciones, datos_test: pd.DataFrame) -> Dict[str, ResultadoValidacion]:
        """Validar todos los modelos del gestor de predicciones"""
        resultados_validacion = {}
        
        for nombre_modelo, modelo in gestor_predicciones.modelos.items():
            try:
                if modelo.modelo_entrenado:
                    self.logger.info(f"Validando modelo {nombre_modelo}...")
                    resultado = self.validar_modelo_completo(modelo, datos_test)
                    resultados_validacion[nombre_modelo] = resultado
                else:
                    self.logger.warning(f"Modelo {nombre_modelo} no está entrenado")
                    
            except Exception as e:
                self.logger.error(f"Error validando modelo {nombre_modelo}: {e}")
        
        return resultados_validacion
    
    def generar_reporte_validacion_completo(self, resultados_validacion: Dict[str, ResultadoValidacion]) -> str:
        """Generar reporte completo de validación"""
        output = []
        
        output.append("=" * 80)
        output.append("REPORTE DE VALIDACIÓN DE MODELOS PREDICTIVOS")
        output.append(f"Generado: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        output.append("=" * 80)
        
        # Resumen ejecutivo
        modelos_criticos = sum(1 for r in resultados_validacion.values() if r.alertas)
        modelos_cumpliendo = sum(1 for r in resultados_validacion.values() if r.cumple_objetivos)
        
        output.append(f"\n[RESUMEN EJECUTIVO]")
        output.append(f"Total modelos evaluados: {len(resultados_validacion)}")
        output.append(f"Modelos cumpliendo objetivos: {modelos_cumpliendo}/{len(resultados_validacion)}")
        output.append(f"Modelos con alertas críticas: {modelos_criticos}")
        output.append(f"Objetivo precisión: {self.precision_objetivo}%")
        output.append(f"Objetivo confianza: {self.confianza_minima}%")
        
        # Validación por modelo
        for nombre, resultado in resultados_validacion.items():
            estado_global = "✅ APROBADO" if resultado.cumple_objetivos else "❌ REQUIERE ATENCIÓN"
            
            output.append(f"\n[{nombre}] {estado_global}")
            output.append(f"  Precisión global: {resultado.precision_global}%")
            output.append(f"  Confianza promedio: {resultado.confianza_promedio}%")
            output.append(f"  Error absoluto promedio: {resultado.error_absoluto_promedio}")
            output.append(f"  R² Score: {resultado.r2_score}")
            output.append(f"  Degradación detectada: {'Sí' if resultado.degradacion_detectada else 'No'}")
            
            # Alertas críticas
            if resultado.alertas:
                output.append(f"  Alertas:")
                for alerta in resultado.alertas:
                    output.append(f"    {alerta}")
            
            # Métricas por segmento
            if resultado.metricas_por_turno:
                output.append(f"  Rendimiento por turno:")
                for turno, precision in resultado.metricas_por_turno.items():
                    output.append(f"    {turno}: {precision}%")
            
            # Recomendaciones principales
            if resultado.recomendaciones:
                output.append(f"  Recomendaciones principales:")
                for rec in resultado.recomendaciones[:3]:  # Top 3
                    output.append(f"    - {rec}")
        
        # Tendencias históricas
        if self.historial_validaciones:
            output.append(f"\n[TENDENCIAS HISTÓRICAS]")
            
            for modelo in set(v['modelo'] for v in self.historial_validaciones):
                validaciones_modelo = [v for v in self.historial_validaciones if v['modelo'] == modelo]
                if len(validaciones_modelo) >= 2:
                    precision_inicial = validaciones_modelo[0]['precision']
                    precision_actual = validaciones_modelo[-1]['precision']
                    tendencia = precision_actual - precision_inicial
                    
                    simbolo = "📈" if tendencia > 0 else "📉" if tendencia < 0 else "➡️"
                    output.append(f"  {modelo}: {simbolo} {tendencia:+.1f}% (de {precision_inicial:.1f}% a {precision_actual:.1f}%)")
        
        # Recomendaciones globales
        output.append(f"\n[RECOMENDACIONES GLOBALES]")
        if modelos_criticos > 0:
            output.append("- Priorizar re-entrenamiento de modelos con alertas críticas")
            output.append("- Implementar monitoreo continuo automático")
        
        if modelos_cumpliendo == len(resultados_validacion):
            output.append("- Todos los modelos cumplen objetivos: mantener monitoreo regular")
        else:
            output.append("- Establecer plan de mejora para modelos no conformes")
        
        output.append("- Considerar implementación de A/B testing para nuevas versiones")
        
        return '\n'.join(output)
    
    def monitoreo_continuo(self, gestor_predicciones, datos_stream: pd.DataFrame, intervalo_validacion_horas: int = 24):
        """Implementar monitoreo continuo de modelos en producción"""
        self.logger.info(f"Iniciando monitoreo continuo cada {intervalo_validacion_horas} horas")
        
        # Esta función sería llamada periódicamente en un sistema real
        # Por ahora, implementamos la lógica base
        
        try:
            resultados = self.validar_todos_modelos(gestor_predicciones, datos_stream)
            
            # Verificar alertas críticas
            alertas_criticas = []
            for nombre, resultado in resultados.items():
                if resultado.alertas:
                    alertas_criticas.extend([f"{nombre}: {alerta}" for alerta in resultado.alertas])
            
            if alertas_criticas:
                self.logger.critical(f"ALERTAS CRÍTICAS DETECTADAS: {alertas_criticas}")
                # Aquí se implementaría notificación automática
            
            return resultados
            
        except Exception as e:
            self.logger.error(f"Error en monitoreo continuo: {e}")
            return {}

class ValidadorCalidadDatos:
    """Validador específico para calidad de datos de entrada"""
    
    def __init__(self):
        self.logger = logging.getLogger("ValidadorCalidadDatos")
    
    def validar_calidad_datos(self, datos: pd.DataFrame) -> Dict[str, Any]:
        """Validar calidad de datos para entrenamiento/predicción"""
        reporte_calidad = {
            'timestamp': datetime.datetime.now(),
            'total_registros': len(datos),
            'columnas_criticas': self._verificar_columnas_criticas(datos),
            'completitud': self._calcular_completitud(datos),
            'consistencia_temporal': self._verificar_consistencia_temporal(datos),
            'outliers_detectados': self._detectar_outliers(datos),
            'calidad_global': 0.0,
            'apto_para_entrenamiento': False,
            'recomendaciones': []
        }
        
        # Calcular calidad global
        calidad_global = self._calcular_calidad_global(reporte_calidad)
        reporte_calidad['calidad_global'] = calidad_global
        reporte_calidad['apto_para_entrenamiento'] = calidad_global >= 75.0
        
        # Generar recomendaciones
        reporte_calidad['recomendaciones'] = self._generar_recomendaciones_calidad(reporte_calidad)
        
        return reporte_calidad
    
    def _verificar_columnas_criticas(self, datos: pd.DataFrame) -> Dict[str, bool]:
        """Verificar presencia de columnas críticas"""
        columnas_criticas = [
            'T. de Microparadas', 'T. Disponible', 'Conteo', 
            'T. de P. No Programada', 'Id Turno', 'Fecha'
        ]
        
        return {col: col in datos.columns for col in columnas_criticas}
    
    def _calcular_completitud(self, datos: pd.DataFrame) -> Dict[str, float]:
        """Calcular completitud de datos por columna"""
        completitud = {}
        for col in datos.columns:
            pct_completo = (datos[col].notna().sum() / len(datos)) * 100
            completitud[col] = round(pct_completo, 2)
        
        return completitud
    
    def _verificar_consistencia_temporal(self, datos: pd.DataFrame) -> Dict[str, Any]:
        """Verificar consistencia temporal de los datos"""
        consistencia = {
            'tiene_fechas': 'Fecha' in datos.columns,
            'orden_cronologico': False,
            'gaps_detectados': 0,
            'frecuencia_detectada': 'Desconocida'
        }
        
        if 'Fecha' in datos.columns:
            fechas = pd.to_datetime(datos['Fecha']).dropna()
            if len(fechas) > 1:
                consistencia['orden_cronologico'] = fechas.is_monotonic_increasing
                
                # Detectar frecuencia
                diff = fechas.diff().dropna()
                if len(diff) > 0:
                    frecuencia_comun = diff.mode()
                    if len(frecuencia_comun) > 0:
                        consistencia['frecuencia_detectada'] = str(frecuencia_comun.iloc[0])
        
        return consistencia
    
    def _detectar_outliers(self, datos: pd.DataFrame) -> Dict[str, int]:
        """Detectar outliers en columnas numéricas"""
        outliers = {}
        
        columnas_numericas = datos.select_dtypes(include=[np.number]).columns
        
        for col in columnas_numericas:
            valores = datos[col].dropna()
            if len(valores) > 10:
                Q1 = valores.quantile(0.25)
                Q3 = valores.quantile(0.75)
                IQR = Q3 - Q1
                
                limite_inferior = Q1 - 1.5 * IQR
                limite_superior = Q3 + 1.5 * IQR
                
                outliers_count = len(valores[(valores < limite_inferior) | (valores > limite_superior)])
                outliers[col] = outliers_count
        
        return outliers
    
    def _calcular_calidad_global(self, reporte: Dict[str, Any]) -> float:
        """Calcular puntuación global de calidad"""
        puntuacion = 0.0
        
        # Columnas críticas (30%)
        columnas_criticas = reporte['columnas_criticas']
        pct_columnas_criticas = sum(columnas_criticas.values()) / len(columnas_criticas) * 100
        puntuacion += pct_columnas_criticas * 0.3
        
        # Completitud promedio (40%)
        completitud = reporte['completitud']
        completitud_promedio = np.mean(list(completitud.values()))
        puntuacion += completitud_promedio * 0.4
        
        # Consistencia temporal (20%)
        consistencia = reporte['consistencia_temporal']
        if consistencia['tiene_fechas'] and consistencia['orden_cronologico']:
            puntuacion += 20
        elif consistencia['tiene_fechas']:
            puntuacion += 10
        
        # Outliers (10%)
        outliers = reporte['outliers_detectados']
        if outliers:
            pct_outliers = sum(outliers.values()) / reporte['total_registros'] * 100
            puntuacion_outliers = max(0, 10 - pct_outliers)  # Penalizar outliers excesivos
            puntuacion += puntuacion_outliers
        else:
            puntuacion += 10
        
        return min(100, puntuacion)
    
    def _generar_recomendaciones_calidad(self, reporte: Dict[str, Any]) -> List[str]:
        """Generar recomendaciones para mejorar calidad de datos"""
        recomendaciones = []
        
        # Columnas faltantes
        columnas_faltantes = [k for k, v in reporte['columnas_criticas'].items() if not v]
        if columnas_faltantes:
            recomendaciones.append(f"Obtener columnas faltantes: {', '.join(columnas_faltantes)}")
        
        # Completitud baja
        completitud_baja = [k for k, v in reporte['completitud'].items() if v < 80]
        if completitud_baja:
            recomendaciones.append(f"Mejorar completitud en: {', '.join(completitud_baja[:3])}")
        
        # Problemas temporales
        if not reporte['consistencia_temporal']['orden_cronologico']:
            recomendaciones.append("Corregir orden cronológico de los datos")
        
        # Outliers excesivos
        outliers_excesivos = [k for k, v in reporte['outliers_detectados'].items() if v > reporte['total_registros'] * 0.05]
        if outliers_excesivos:
            recomendaciones.append(f"Revisar outliers en: {', '.join(outliers_excesivos[:2])}")
        
        if reporte['calidad_global'] < 75:
            recomendaciones.append("Calidad insuficiente para entrenamiento confiable")
        
        return recomendaciones