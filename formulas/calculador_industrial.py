# ============================================================================
# CALCULADOR DE FÓRMULAS INDUSTRIALES - VERSIÓN ARREGLADA
# Solución para error de logger + análisis completo de datos reales
# ============================================================================

import pandas as pd
import numpy as np
import datetime
import logging
from typing import Dict, List, Optional, Any

# Configurar logging global
logging.basicConfig(level=logging.INFO)

# ============================================================================
# CALCULADOR DE FÓRMULAS INDUSTRIALES ARREGLADO
# ============================================================================

class CalculadorFormulasIndustrialesArreglado:
    """Calculador de fórmulas industriales con manejo mejorado"""
    
    def __init__(self, datos: pd.DataFrame):
        self.datos = datos
        self.logger = logging.getLogger(__name__)  # ← AQUÍ ESTABA EL ERROR
        self.mapeo_columnas = self._crear_mapeo_inteligente()
        self.estadisticas_datos = self._calcular_estadisticas_generales()
    
    def _crear_mapeo_inteligente(self) -> Dict[str, str]:
        """Crear mapeo inteligente de columnas críticas"""
        mapeo = {}
        
        # Mapeo directo para las columnas encontradas
        mapeo_directo = {
            'microparadas': 'T. de Microparadas',
            'disponible': 'T. Disponible', 
            'conteo': 'Conteo',
            'parada_no_programada': 'T. de P. No Programada',
            'turno': 'Id Turno',
            'fecha': 'Fecha',
            'merma': 'Merma',
            'id': 'id',
            'producto': 'Id Producto',
            'lote': 'Id Lote',
            'por_segundo': 'por segundo',
            'litros_unidad': 'Litros x unidad'
        }
        
        # Verificar que las columnas existan
        for clave, nombre_columna in mapeo_directo.items():
            if nombre_columna in self.datos.columns:
                mapeo[clave] = nombre_columna
                self.logger.info(f"Mapeo: {clave} -> {nombre_columna}")
            else:
                self.logger.warning(f"Columna no encontrada: {nombre_columna}")
        
        return mapeo
    
    def _calcular_estadisticas_generales(self) -> Dict:
        """Calcular estadísticas generales de los datos"""
        stats = {
            'total_registros': len(self.datos),
            'periodo_analisis': self._obtener_periodo_datos(),
            'turnos_unicos': self.datos.get('Id Turno', pd.Series()).dropna().nunique(),
            'productos_unicos': self.datos.get('Id Producto', pd.Series()).dropna().nunique(),
            'registros_con_microparadas': 0,
            'registros_con_produccion': 0
        }
        
        # Estadísticas específicas
        if 'T. de Microparadas' in self.datos.columns:
            microparadas_validas = self.datos['T. de Microparadas'].dropna()
            stats['registros_con_microparadas'] = len(microparadas_validas[microparadas_validas > 0])
        
        if 'Conteo' in self.datos.columns:
            produccion_valida = self.datos['Conteo'].dropna()
            stats['registros_con_produccion'] = len(produccion_valida[produccion_valida > 0])
        
        return stats
    
    def calcular_ivm_datos_reales(self, filtros: Optional[Dict] = None) -> Dict:
        """
        Calcular IVM usando datos reales de microparadas
        IVM = (σ/μ) × 100% - Índice de Variabilidad de Microparadas
        """
        col_microparadas = self.mapeo_columnas.get('microparadas')
        
        if not col_microparadas:
            return {
                'error': 'Columna de microparadas no encontrada',
                'columnas_disponibles': list(self.datos.columns)
            }
        
        try:
            # Aplicar filtros si se proporcionan
            df_trabajo = self._aplicar_filtros(filtros) if filtros else self.datos
            
            # Obtener datos de microparadas válidos
            microparadas = df_trabajo[col_microparadas].dropna()
            microparadas = microparadas[microparadas >= 0]
            
            if len(microparadas) == 0:
                return {
                    'valor': 0,
                    'interpretacion': 'Sin datos válidos de microparadas',
                    'datos_utilizados': 0
                }
            
            # Calcular componentes estadísticos
            media = microparadas.mean()
            desviacion_std = microparadas.std()
            mediana = microparadas.median()
            q25 = microparadas.quantile(0.25)
            q75 = microparadas.quantile(0.75)
            minimo = microparadas.min()
            maximo = microparadas.max()
            
            # Calcular IVM
            ivm = (desviacion_std / media) * 100 if media > 0 else 0
            
            # Análisis de distribución
            registros_cero = len(microparadas[microparadas == 0])
            registros_altos = len(microparadas[microparadas > media + 2 * desviacion_std])
            
            componentes = {
                'sigma_desviacion': round(desviacion_std, 2),
                'mu_media': round(media, 2),
                'mediana': round(mediana, 2),
                'q25': round(q25, 2),
                'q75': round(q75, 2),
                'minimo': round(minimo, 2),
                'maximo': round(maximo, 2),
                'rango': round(maximo - minimo, 2),
                'registros_cero': registros_cero,
                'registros_outliers': registros_altos,
                'coef_variacion': round(ivm, 2),
                'formula': '(σ/μ) × 100%'
            }
            
            # Interpretación avanzada
            interpretacion = self._interpretar_ivm_avanzado(ivm, componentes)
            
            return {
                'valor': round(ivm, 1),
                'componentes': componentes,
                'interpretacion': interpretacion,
                'datos_utilizados': len(microparadas),
                'periodo_analizado': self._obtener_periodo_datos(df_trabajo),
                'recomendaciones': self._generar_recomendaciones_ivm(ivm, componentes)
            }
            
        except Exception as e:
            self.logger.error(f"Error calculando IVM: {e}")
            return {'error': f'Error calculando IVM: {e}'}
    
    def calcular_iet_datos_reales(self, filtros: Optional[Dict] = None) -> Dict:
        """
        Calcular IET usando datos reales
        IET = ((T.Disponible - T.Microparadas) / T.Disponible) × 100%
        """
        col_disponible = self.mapeo_columnas.get('disponible')
        col_microparadas = self.mapeo_columnas.get('microparadas')
        
        if not col_disponible or not col_microparadas:
            return {
                'error': f'Columnas faltantes - Disponible: {col_disponible}, Microparadas: {col_microparadas}'
            }
        
        try:
            df_trabajo = self._aplicar_filtros(filtros) if filtros else self.datos
            
            # Filtrar datos válidos
            datos_validos = df_trabajo[
                (df_trabajo[col_disponible].notna()) & 
                (df_trabajo[col_disponible] > 0) &
                (df_trabajo[col_microparadas].notna()) &
                (df_trabajo[col_microparadas] >= 0)
            ].copy()
            
            if len(datos_validos) == 0:
                return {'valor': 0, 'interpretacion': 'Sin datos válidos para IET'}
            
            # Calcular métricas totales
            tiempo_disponible_total = datos_validos[col_disponible].sum()
            tiempo_microparadas_total = datos_validos[col_microparadas].sum()
            tiempo_efectivo_total = tiempo_disponible_total - tiempo_microparadas_total
            
            # IET global
            iet_global = (tiempo_efectivo_total / tiempo_disponible_total) * 100 if tiempo_disponible_total > 0 else 0
            
            # Análisis por períodos
            datos_validos['IET_Individual'] = (
                (datos_validos[col_disponible] - datos_validos[col_microparadas]) / 
                datos_validos[col_disponible]
            ) * 100
            
            iet_promedio = datos_validos['IET_Individual'].mean()
            iet_mediana = datos_validos['IET_Individual'].median()
            iet_std = datos_validos['IET_Individual'].std()
            
            # Análisis de períodos críticos
            periodos_criticos = len(datos_validos[datos_validos['IET_Individual'] < 50])
            periodos_excelentes = len(datos_validos[datos_validos['IET_Individual'] > 90])
            
            componentes = {
                't_disponible_total': round(tiempo_disponible_total, 2),
                't_microparadas_total': round(tiempo_microparadas_total, 2),
                't_efectivo_total': round(tiempo_efectivo_total, 2),
                'iet_promedio': round(iet_promedio, 2),
                'iet_mediana': round(iet_mediana, 2),
                'iet_desviacion': round(iet_std, 2),
                'periodos_criticos': periodos_criticos,
                'periodos_excelentes': periodos_excelentes,
                'eficiencia_promedio': round((tiempo_efectivo_total / tiempo_disponible_total) * 100, 2),
                'formula': '((T.Disponible - T.Microparadas) / T.Disponible) × 100%'
            }
            
            interpretacion = self._interpretar_iet_avanzado(iet_global, componentes)
            
            return {
                'valor': round(iet_global, 1),
                'componentes': componentes,
                'interpretacion': interpretacion,
                'datos_utilizados': len(datos_validos),
                'periodo_analizado': self._obtener_periodo_datos(df_trabajo),
                'recomendaciones': self._generar_recomendaciones_iet(iet_global, componentes)
            }
            
        except Exception as e:
            self.logger.error(f"Error calculando IET: {e}")
            return {'error': f'Error calculando IET: {e}'}
    
    def calcular_ipc_datos_reales(self, filtros: Optional[Dict] = None) -> Dict:
        """
        Calcular IPC usando datos reales de conteo
        IPC = (Promedio.Conteo / Máximo.Conteo) × 100%
        """
        col_conteo = self.mapeo_columnas.get('conteo')
        
        if not col_conteo:
            return {'error': 'Columna de conteo no encontrada'}
        
        try:
            df_trabajo = self._aplicar_filtros(filtros) if filtros else self.datos
            
            # Obtener datos de conteo válidos
            conteos = df_trabajo[col_conteo].dropna()
            conteos = conteos[conteos >= 0]
            
            if len(conteos) == 0:
                return {'valor': 0, 'interpretacion': 'Sin datos de conteo válidos'}
            
            # Métricas básicas
            conteo_promedio = conteos.mean()
            conteo_maximo = conteos.max()
            conteo_minimo = conteos.min()
            conteo_total = conteos.sum()
            conteo_mediana = conteos.median()
            conteo_std = conteos.std()
            
            # Calcular IPC
            ipc = (conteo_promedio / conteo_maximo) * 100 if conteo_maximo > 0 else 0
            
            # Análisis de productividad
            registros_alta_produccion = len(conteos[conteos > conteo_promedio + conteo_std])
            registros_baja_produccion = len(conteos[conteos < conteo_promedio - conteo_std])
            registros_cero_produccion = len(conteos[conteos == 0])
            
            # Análisis por turno si está disponible
            analisis_turnos = {}
            if 'Id Turno' in df_trabajo.columns:
                turnos_stats = df_trabajo.groupby('Id Turno')[col_conteo].agg(['sum', 'mean', 'count']).round(2)
                analisis_turnos = turnos_stats.to_dict()
            
            componentes = {
                'conteo_promedio': round(conteo_promedio, 2),
                'conteo_maximo': round(conteo_maximo, 2),
                'conteo_minimo': round(conteo_minimo, 2),
                'conteo_mediana': round(conteo_mediana, 2),
                'conteo_total': round(conteo_total, 2),
                'conteo_desviacion': round(conteo_std, 2),
                'registros_alta_prod': registros_alta_produccion,
                'registros_baja_prod': registros_baja_produccion,
                'registros_cero_prod': registros_cero_produccion,
                'analisis_turnos': analisis_turnos,
                'formula': '(Promedio.Conteo / Máximo.Conteo) × 100%'
            }
            
            interpretacion = self._interpretar_ipc_avanzado(ipc, componentes)
            
            return {
                'valor': round(ipc, 1),
                'componentes': componentes,
                'interpretacion': interpretacion,
                'datos_utilizados': len(conteos),
                'periodo_analizado': self._obtener_periodo_datos(df_trabajo),
                'recomendaciones': self._generar_recomendaciones_ipc(ipc, componentes)
            }
            
        except Exception as e:
            self.logger.error(f"Error calculando IPC: {e}")
            return {'error': f'Error calculando IPC: {e}'}
    
    def calcular_todas_las_formulas_reales(self, filtros: Optional[Dict] = None) -> Dict:
        """Calcular todas las fórmulas usando datos reales"""
        self.logger.info("Calculando todas las fórmulas con datos reales...")
        
        resultados = {
            'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'estadisticas_datos': self.estadisticas_datos,
            'mapeo_columnas': self.mapeo_columnas,
            'filtros_aplicados': filtros or 'Ninguno'
        }
        
        # Calcular cada fórmula
        formulas = ['IVM', 'IET', 'IPC']
        
        for formula in formulas:
            try:
                if formula == 'IVM':
                    resultado = self.calcular_ivm_datos_reales(filtros)
                elif formula == 'IET':
                    resultado = self.calcular_iet_datos_reales(filtros)
                elif formula == 'IPC':
                    resultado = self.calcular_ipc_datos_reales(filtros)
                
                resultados[formula] = resultado
                
                if 'error' not in resultado:
                    self.logger.info(f"{formula} calculado: {resultado['valor']}%")
                else:
                    self.logger.error(f"Error en {formula}: {resultado['error']}")
                    
            except Exception as e:
                self.logger.error(f"Error crítico calculando {formula}: {e}")
                resultados[formula] = {'error': f'Error crítico: {e}'}
        
        return resultados
    
    def generar_reporte_industrial_completo(self, filtros: Optional[Dict] = None) -> str:
        """Generar reporte completo de análisis industrial"""
        resultados = self.calcular_todas_las_formulas_reales(filtros)
        
        output = []
        
        # Cabecera
        output.append("=" * 80)
        output.append("REPORTE INDUSTRIAL COMPLETO - DATOS REALES")
        output.append(f"Generado: {resultados['timestamp']}")
        output.append("=" * 80)
        
        # Estadísticas de datos
        stats = resultados['estadisticas_datos']
        output.append(f"\n[ESTADISTICAS DE DATOS]")
        output.append("-" * 50)
        output.append(f"Total registros analizados: {stats['total_registros']:,}")
        output.append(f"Período de análisis: {stats['periodo_analisis']}")
        output.append(f"Turnos únicos: {stats['turnos_unicos']}")
        output.append(f"Productos únicos: {stats['productos_unicos']}")
        output.append(f"Registros con microparadas: {stats['registros_con_microparadas']:,}")
        output.append(f"Registros con producción: {stats['registros_con_produccion']:,}")
        
        # Resultados de fórmulas
        output.append(f"\n[INDICADORES INDUSTRIALES CALCULADOS]")
        output.append("-" * 50)
        
        for formula in ['IVM', 'IET', 'IPC']:
            if formula in resultados:
                resultado = resultados[formula]
                
                if 'error' in resultado:
                    output.append(f"\n{formula}: ERROR - {resultado['error']}")
                    continue
                
                # Información básica
                estado = self._determinar_estado_alerta(resultado['valor'])
                output.append(f"\n{formula} (Datos Reales): {resultado['valor']}% - {estado}")
                output.append(f"    Registros utilizados: {resultado['datos_utilizados']:,}")
                output.append(f"    Interpretación: {resultado['interpretacion']}")
                
                # Componentes específicos
                comp = resultado['componentes']
                if formula == 'IVM':
                    output.append(f"    σ (Desviación): {comp['sigma_desviacion']}")
                    output.append(f"    μ (Media): {comp['mu_media']} minutos")
                    output.append(f"    Registros con microparadas: {comp['registros_cero']:,}")
                    output.append(f"    Outliers detectados: {comp['registros_outliers']:,}")
                    
                elif formula == 'IET':
                    output.append(f"    Tiempo disponible total: {comp['t_disponible_total']:,} min")
                    output.append(f"    Tiempo microparadas total: {comp['t_microparadas_total']:,} min")
                    output.append(f"    Eficiencia promedio: {comp['eficiencia_promedio']}%")
                    output.append(f"    Períodos críticos (<50%): {comp['periodos_criticos']:,}")
                    
                elif formula == 'IPC':
                    output.append(f"    Conteo promedio: {comp['conteo_promedio']:,}")
                    output.append(f"    Conteo máximo: {comp['conteo_maximo']:,}")
                    output.append(f"    Producción total: {comp['conteo_total']:,}")
                    output.append(f"    Períodos sin producción: {comp['registros_cero_prod']:,}")
                
                # Recomendaciones
                if 'recomendaciones' in resultado:
                    output.append(f"    Recomendaciones: {'; '.join(resultado['recomendaciones'])}")
        
        # Resumen final
        output.append(f"\n[RESUMEN EJECUTIVO]")
        output.append("-" * 50)
        
        # Identificar indicador más crítico
        valores_formulas = {}
        for formula in ['IVM', 'IET', 'IPC']:
            if formula in resultados and 'valor' in resultados[formula]:
                valores_formulas[formula] = resultados[formula]['valor']
        
        if valores_formulas:
            # Para IVM, mayor valor es peor; para IET e IPC, menor valor es peor
            ivm_critico = valores_formulas.get('IVM', 0) > 50
            iet_critico = valores_formulas.get('IET', 100) < 70
            ipc_critico = valores_formulas.get('IPC', 100) < 75
            
            criticos = []
            if ivm_critico:
                criticos.append("IVM (alta variabilidad)")
            if iet_critico:
                criticos.append("IET (baja eficiencia)")
            if ipc_critico:
                criticos.append("IPC (baja capacidad)")
            
            if criticos:
                output.append(f"Indicadores críticos: {', '.join(criticos)}")
            else:
                output.append("Todos los indicadores en rangos aceptables")
        
        # Período analizado
        if 'periodo_analizado' in resultados.get('IVM', {}):
            periodo = resultados['IVM']['periodo_analizado']
            output.append(f"Período analizado: {periodo}")
        
        # Footer
        timestamp_fin = datetime.datetime.now()
        output.append(f"\n[REPORTE FINALIZADO]: {timestamp_fin.strftime('%Y-%m-%d %H:%M:%S')}")
        
        return '\n'.join(output)
    
    # ========================================================================
    # MÉTODOS AUXILIARES
    # ========================================================================
    
    def _aplicar_filtros(self, filtros: Dict) -> pd.DataFrame:
        """Aplicar filtros específicos a los datos"""
        df_filtrado = self.datos.copy()
        
        for columna, valor in filtros.items():
            if columna in df_filtrado.columns:
                if isinstance(valor, (list, tuple)):
                    df_filtrado = df_filtrado[df_filtrado[columna].isin(valor)]
                else:
                    df_filtrado = df_filtrado[df_filtrado[columna] == valor]
                    
        self.logger.info(f"Filtros aplicados: {filtros}. Registros resultantes: {len(df_filtrado)}")
        return df_filtrado
    
    def _obtener_periodo_datos(self, df: Optional[pd.DataFrame] = None) -> str:
        """Obtener información del período de datos"""
        if df is None:
            df = self.datos
            
        if 'Fecha' in df.columns and df['Fecha'].notna().sum() > 0:
            fechas_validas = df['Fecha'].dropna()
            fecha_min = fechas_validas.min()
            fecha_max = fechas_validas.max()
            dias = (fecha_max - fecha_min).days + 1
            return f"{fecha_min.strftime('%Y-%m-%d')} a {fecha_max.strftime('%Y-%m-%d')} ({dias} días)"
        else:
            return "Período no disponible"
    
    def _interpretar_ivm_avanzado(self, valor: float, componentes: Dict) -> str:
        base = self._interpretar_variabilidad_basico(valor)
        
        # Análisis adicional basado en componentes
        outliers_pct = (componentes['registros_outliers'] / componentes.get('total_registros', 1)) * 100
        
        if outliers_pct > 10:
            return f"{base}. Detectados picos anómalos ({outliers_pct:.1f}% outliers)"
        elif componentes['registros_cero'] > 1000:
            return f"{base}. Muchos períodos sin microparadas ({componentes['registros_cero']:,})"
        else:
            return base
    
    def _interpretar_iet_avanzado(self, valor: float, componentes: Dict) -> str:
        base = self._interpretar_eficiencia_basico(valor)
        
        periodos_criticos = componentes['periodos_criticos']
        total_periodos = componentes.get('registros_analizados', 1)
        pct_criticos = (periodos_criticos / total_periodos) * 100
        
        if pct_criticos > 20:
            return f"{base}. Alto porcentaje de períodos críticos ({pct_criticos:.1f}%)"
        else:
            return base
    
    def _interpretar_ipc_avanzado(self, valor: float, componentes: Dict) -> str:
        base = self._interpretar_capacidad_basico(valor)
        
        cero_prod_pct = (componentes['registros_cero_prod'] / componentes.get('total_registros', 1)) * 100
        
        if cero_prod_pct > 15:
            return f"{base}. Muchos períodos sin producción ({cero_prod_pct:.1f}%)"
        else:
            return base
    
    def _generar_recomendaciones_ivm(self, valor: float, componentes: Dict) -> List[str]:
        recomendaciones = []
        
        if valor > 50:
            recomendaciones.append("Investigar causas de alta variabilidad")
            recomendaciones.append("Implementar mantenimiento preventivo")
        elif valor > 25:
            recomendaciones.append("Monitorear tendencias de microparadas")
        
        if componentes['registros_outliers'] > 100:
            recomendaciones.append("Analizar picos anómalos específicos")
        
        return recomendaciones
    
    def _generar_recomendaciones_iet(self, valor: float, componentes: Dict) -> List[str]:
        recomendaciones = []
        
        if valor < 70:
            recomendaciones.append("Reducir tiempo de microparadas")
            recomendaciones.append("Optimizar procesos de producción")
        elif valor < 85:
            recomendaciones.append("Identificar oportunidades de mejora")
        
        if componentes['periodos_criticos'] > 1000:
            recomendaciones.append("Analizar períodos de baja eficiencia")
        
        return recomendaciones
    
    def _generar_recomendaciones_ipc(self, valor: float, componentes: Dict) -> List[str]:
        recomendaciones = []
        
        if valor < 75:
            recomendaciones.append("Optimizar capacidad de producción")
            recomendaciones.append("Revisar procesos operativos")
        
        if componentes['registros_cero_prod'] > 5000:
            recomendaciones.append("Reducir períodos sin producción")
        
        return recomendaciones
    
    def _interpretar_variabilidad_basico(self, valor: float) -> str:
        if valor <= 25:
            return "Variabilidad baja - operación estable"
        elif valor <= 50:
            return "Variabilidad moderada - requiere monitoreo"
        else:
            return "Variabilidad alta - intervención necesaria"
    
    def _interpretar_eficiencia_basico(self, valor: float) -> str:
        if valor >= 85:
            return "Eficiencia excelente"
        elif valor >= 70:
            return "Eficiencia buena - margen de mejora"
        else:
            return "Eficiencia baja - acción requerida"
    
    def _interpretar_capacidad_basico(self, valor: float) -> str:
        if valor >= 90:
            return "Capacidad óptima"
        elif valor >= 75:
            return "Capacidad buena"
        else:
            return "Capacidad limitada - optimización requerida"
    
    def _determinar_estado_alerta(self, valor: float) -> str:
        """Determinar estado de alerta basado en umbrales"""
        if valor <= 25:
            return "NORMAL"
        elif valor <= 50:
            return "ATENCION_REQUERIDA"
        else:
            return "ALERTA_CRITICA"


# ============================================================================
# FUNCIÓN PRINCIPAL PARA USAR CON LOS DATOS CARGADOS
# ============================================================================

def procesar_datos_industriales_completo(datos: pd.DataFrame, filtros: Optional[Dict] = None) -> str:
    """Función principal para procesar los datos industriales cargados"""
    
    # Inicializar calculador arreglado
    calculador = CalculadorFormulasIndustrialesArreglado(datos)
    
    # Generar reporte completo
    reporte = calculador.generar_reporte_industrial_completo(filtros)
    
    return reporte

