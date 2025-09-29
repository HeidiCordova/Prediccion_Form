import pandas as pd
import numpy as np
import datetime
import logging
from typing import Dict, List, Optional, Any, Tuple
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.model_selection import KFold
import warnings
warnings.filterwarnings('ignore')

# Configurar logging global
logging.basicConfig(level=logging.INFO)

# ============================================================================
# CALCULADOR DE FÓRMULAS INDUSTRIALES 
# ============================================================================

class CalculadorFormulasIndustriales:
    """Calculador de fórmulas industriales con manejo completo de outliers y validación"""
    
    def __init__(self, datos: pd.DataFrame):
        self.datos = datos
        self.logger = logging.getLogger(__name__)
        self.mapeo_columnas = self._crear_mapeo_inteligente()
        self.estadisticas_datos = self._calcular_estadisticas_generales()
        
        # Umbrales realistas para datos industriales
        self.umbrales = {
            'microparadas_max_minutos': 480,  # 8 horas máximo
            'tiempo_disponible_max_minutos': 1440,  # 24 horas máximo
            'conteo_max_unidades': 100000,  # 100K unidades máximo
            'percentil_outliers': 99.5,  # Eliminar top 0.5%
            'min_registros_validos': 10  # Mínimo para cálculos
        }
    
    def _crear_mapeo_inteligente(self) -> Dict[str, str]:
        """Crear mapeo inteligente de columnas críticas"""
        mapeo = {}
        
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
        
        for clave, nombre_columna in mapeo_directo.items():
            if nombre_columna in self.datos.columns:
                mapeo[clave] = nombre_columna
                self.logger.info(f"Mapeo: {clave} -> {nombre_columna}")
        
        return mapeo
    
    def diagnosticar_datos_completo(self) -> str:
        """Diagnóstico completo de calidad de datos"""
        output = []
        output.append("=" * 80)
        output.append("DIAGNÓSTICO COMPLETO DE DATOS INDUSTRIALES")
        output.append("=" * 80)
        
        # Análisis de microparadas
        if 'T. de Microparadas' in self.datos.columns:
            micro = self.datos['T. de Microparadas'].dropna()
            output.append(f"\n[MICROPARADAS - {len(micro):,} registros]")
            output.append(f"  Min: {micro.min():.2f} min")
            output.append(f"  Max: {micro.max():.2f} min")
            output.append(f"  Media: {micro.mean():.2f} min")
            output.append(f"  Mediana: {micro.median():.2f} min")
            output.append(f"  Desv. Std: {micro.std():.2f} min")
            
            output.append(f"\n  Distribución por rangos:")
            output.append(f"    = 0 min: {len(micro[micro == 0]):,} ({len(micro[micro == 0])/len(micro)*100:.1f}%)")
            output.append(f"    0-5 min: {len(micro[(micro > 0) & (micro <= 5)]):,} ({len(micro[(micro > 0) & (micro <= 5)])/len(micro)*100:.1f}%)")
            output.append(f"    5-15 min: {len(micro[(micro > 5) & (micro <= 15)]):,} ({len(micro[(micro > 5) & (micro <= 15)])/len(micro)*100:.1f}%)")
            output.append(f"    15-60 min: {len(micro[(micro > 15) & (micro <= 60)]):,} ({len(micro[(micro > 15) & (micro <= 60)])/len(micro)*100:.1f}%)")
            output.append(f"    > 60 min: {len(micro[micro > 60]):,} ({len(micro[micro > 60])/len(micro)*100:.1f}%)")
            output.append(f"    > 480 min: {len(micro[micro > 480]):,} (OUTLIERS EXTREMOS)")
            
            output.append(f"\n  Percentiles:")
            for p in [50, 75, 90, 95, 99]:
                output.append(f"    P{p}: {micro.quantile(p/100):.2f} min")
        
        # Análisis de tiempo disponible
        if 'T. Disponible' in self.datos.columns:
            disp = self.datos['T. Disponible'].dropna()
            output.append(f"\n[TIEMPO DISPONIBLE - {len(disp):,} registros]")
            output.append(f"  Media: {disp.mean():.2f} min")
            output.append(f"  Valores > 0: {len(disp[disp > 0]):,} ({len(disp[disp > 0])/len(disp)*100:.1f}%)")
            output.append(f"  Valores > 1440 min (24h): {len(disp[disp > 1440]):,}")
        
        # Análisis de conteos
        if 'Conteo' in self.datos.columns:
            conteo = self.datos['Conteo'].dropna()
            output.append(f"\n[CONTEO PRODUCCIÓN - {len(conteo):,} registros]")
            output.append(f"  Min: {conteo.min():.0f}")
            output.append(f"  Max: {conteo.max():.0f}")
            output.append(f"  Media: {conteo.mean():.2f}")
            output.append(f"  Mediana: {conteo.median():.2f}")
            output.append(f"  P95: {conteo.quantile(0.95):.2f}")
            output.append(f"  Valores = 0: {len(conteo[conteo == 0]):,} ({len(conteo[conteo == 0])/len(conteo)*100:.1f}%)")
        
        # Análisis de registros válidos para IET
        if 'T. Disponible' in self.datos.columns and 'T. de Microparadas' in self.datos.columns:
            validos = self.datos[
                (self.datos['T. Disponible'].notna()) & 
                (self.datos['T. Disponible'] > 0) &
                (self.datos['T. de Microparadas'].notna()) &
                (self.datos['T. de Microparadas'] >= 0) &
                (self.datos['T. de Microparadas'] <= self.datos['T. Disponible'])
            ]
            output.append(f"\n[REGISTROS VÁLIDOS PARA IET]")
            output.append(f"  Total válidos: {len(validos):,} de {len(self.datos):,} ({len(validos)/len(self.datos)*100:.1f}%)")
            
            if len(validos) > 0:
                iet_muestra = ((validos['T. Disponible'] - validos['T. de Microparadas']) / validos['T. Disponible']) * 100
                output.append(f"  IET promedio: {iet_muestra.mean():.1f}%")
                output.append(f"  IET mediana: {iet_muestra.median():.1f}%")
        
        return '\n'.join(output)
    
    def _limpiar_datos_microparadas(self, microparadas: pd.Series, excluir_ceros: bool = False) -> pd.Series:
        """Limpiar datos de microparadas con opción de excluir ceros"""
        datos_originales = len(microparadas)
        
        # 1. Eliminar valores negativos
        microparadas = microparadas[microparadas >= 0]
        
        # 2. Opción: Excluir ceros si la mayoría son ceros
        if excluir_ceros:
            microparadas = microparadas[microparadas > 0]
            self.logger.info(f"Excluyendo valores cero de microparadas")
        
        # 3. Eliminar valores absurdamente altos
        microparadas = microparadas[microparadas <= self.umbrales['microparadas_max_minutos']]
        
        # 4. Eliminar outliers usando IQR
        Q1 = microparadas.quantile(0.25)
        Q3 = microparadas.quantile(0.75)
        IQR = Q3 - Q1
        
        limite_inferior = Q1 - 3.0 * IQR
        limite_superior = Q3 + 3.0 * IQR
        
        microparadas = microparadas[
            (microparadas >= max(0, limite_inferior)) & 
            (microparadas <= limite_superior)
        ]
        
        # 5. Eliminar percentil más alto
        percentil_max = microparadas.quantile(self.umbrales['percentil_outliers'] / 100)
        microparadas = microparadas[microparadas <= percentil_max]
        
        datos_finales = len(microparadas)
        eliminados = datos_originales - datos_finales
        
        self.logger.info(f"Limpieza microparadas: {datos_originales} -> {datos_finales} ({eliminados} eliminados, {eliminados/datos_originales*100:.1f}%)")
        
        return microparadas
    
    def _limpiar_datos_conteo(self, conteos: pd.Series) -> pd.Series:
        """Limpiar datos de conteo"""
        datos_originales = len(conteos)
        
        conteos = conteos[conteos >= 0]
        conteos = conteos[conteos <= self.umbrales['conteo_max_unidades']]
        
        percentil_max = conteos.quantile(self.umbrales['percentil_outliers'] / 100)
        conteos = conteos[conteos <= percentil_max]
        
        datos_finales = len(conteos)
        self.logger.info(f"Limpieza conteos: {datos_originales} -> {datos_finales}")
        
        return conteos
    
    def _calcular_estadisticas_generales(self) -> Dict:
        """Calcular estadísticas generales"""
        stats = {
            'total_registros': len(self.datos),
            'periodo_analisis': self._obtener_periodo_datos(),
            'turnos_unicos': self.datos.get('Id Turno', pd.Series()).dropna().nunique(),
            'productos_unicos': self.datos.get('Id Producto', pd.Series()).dropna().nunique(),
            'registros_con_microparadas': 0,
            'registros_con_produccion': 0
        }
        
        if 'T. de Microparadas' in self.datos.columns:
            stats['registros_con_microparadas'] = len(
                self.datos[self.datos['T. de Microparadas'] > 0]
            )
        
        if 'Conteo' in self.datos.columns:
            stats['registros_con_produccion'] = len(
                self.datos[self.datos['Conteo'] > 0]
            )
        
        return stats
    
    def calcular_ivm_datos_reales(self, filtros: Optional[Dict] = None, excluir_ceros: bool = None) -> Dict:
        """
        Calcular IVM con opción de excluir ceros automáticamente
        IVM = (σ/μ) × 100%
        """
        col_microparadas = self.mapeo_columnas.get('microparadas')
        
        if not col_microparadas:
            return {'error': 'Columna de microparadas no encontrada'}
        
        try:
            df_trabajo = self._aplicar_filtros(filtros) if filtros else self.datos
            microparadas_raw = df_trabajo[col_microparadas].dropna()
            
            # Auto-detectar si hay muchos ceros
            pct_ceros = (len(microparadas_raw[microparadas_raw == 0]) / len(microparadas_raw)) * 100
            
            if excluir_ceros is None:
                excluir_ceros = pct_ceros > 50  # Auto-excluir si >50% son ceros
            
            microparadas = self._limpiar_datos_microparadas(microparadas_raw, excluir_ceros)
            
            if len(microparadas) < self.umbrales['min_registros_validos']:
                return {
                    'valor': 0,
                    'interpretacion': f'Datos insuficientes (solo {len(microparadas)} registros válidos)',
                    'datos_utilizados': len(microparadas)
                }
            
            media = microparadas.mean()
            desviacion_std = microparadas.std()
            mediana = microparadas.median()
            
            ivm = (desviacion_std / media) * 100 if media > 0 else 0
            
            componentes = {
                'sigma_desviacion': round(desviacion_std, 2),
                'mu_media': round(media, 2),
                'mediana': round(mediana, 2),
                'minimo': round(microparadas.min(), 2),
                'maximo': round(microparadas.max(), 2),
                'datos_eliminados_limpieza': len(microparadas_raw) - len(microparadas),
                'porcentaje_ceros_original': round(pct_ceros, 1),
                'ceros_excluidos': excluir_ceros,
                'coef_variacion': round(ivm, 2),
                'formula': '(σ/μ) × 100%' + (' [sin ceros]' if excluir_ceros else '')
            }
            
            self.logger.info(f"IVM: Media={media:.2f}, Std={desviacion_std:.2f}, IVM={ivm:.1f}% (ceros excluidos: {excluir_ceros})")
            
            return {
                'valor': round(ivm, 1),
                'componentes': componentes,
                'interpretacion': self._interpretar_ivm_avanzado(ivm, componentes),
                'datos_utilizados': len(microparadas),
                'periodo_analizado': self._obtener_periodo_datos(df_trabajo),
                'recomendaciones': self._generar_recomendaciones_ivm(ivm, componentes)
            }
            
        except Exception as e:
            self.logger.error(f"Error calculando IVM: {e}")
            return {'error': f'Error calculando IVM: {e}'}
    
    def calcular_iet_datos_reales(self, filtros: Optional[Dict] = None) -> Dict:
        """
        Calcular IET con validación robusta
        IET = ((T.Disponible - T.Microparadas) / T.Disponible) × 100%
        """
        col_disponible = self.mapeo_columnas.get('disponible')
        col_microparadas = self.mapeo_columnas.get('microparadas')
        
        if not col_disponible or not col_microparadas:
            return {
                'error': f'Columnas requeridas no encontradas',
                'detalles': f'Disponible: {col_disponible}, Microparadas: {col_microparadas}'
            }
        
        try:
            df_trabajo = self._aplicar_filtros(filtros) if filtros else self.datos
            
            # Validación estricta
            datos_validos = df_trabajo[
                (df_trabajo[col_disponible].notna()) & 
                (df_trabajo[col_disponible] > 0) &
                (df_trabajo[col_disponible] <= self.umbrales['tiempo_disponible_max_minutos']) &
                (df_trabajo[col_microparadas].notna()) &
                (df_trabajo[col_microparadas] >= 0) &
                (df_trabajo[col_microparadas] <= df_trabajo[col_disponible])
            ].copy()
            
            if len(datos_validos) < self.umbrales['min_registros_validos']:
                return {
                    'error': f'Registros válidos insuficientes: {len(datos_validos)}',
                    'valor': 0,
                    'detalles': 'Verifique columnas T. Disponible y T. de Microparadas'
                }
            
            tiempo_disponible_total = datos_validos[col_disponible].sum()
            tiempo_microparadas_total = datos_validos[col_microparadas].sum()
            tiempo_efectivo_total = tiempo_disponible_total - tiempo_microparadas_total
            
            iet_global = (tiempo_efectivo_total / tiempo_disponible_total) * 100 if tiempo_disponible_total > 0 else 0
            
            datos_validos['IET_Individual'] = (
                (datos_validos[col_disponible] - datos_validos[col_microparadas]) / 
                datos_validos[col_disponible]
            ) * 100
            
            componentes = {
                't_disponible_total': round(tiempo_disponible_total, 2),
                't_microparadas_total': round(tiempo_microparadas_total, 2),
                't_efectivo_total': round(tiempo_efectivo_total, 2),
                'iet_promedio': round(datos_validos['IET_Individual'].mean(), 2),
                'iet_mediana': round(datos_validos['IET_Individual'].median(), 2),
                'registros_validos': len(datos_validos),
                'formula': '((T.Disponible - T.Microparadas) / T.Disponible) × 100%'
            }
            
            self.logger.info(f"IET: {iet_global:.1f}% ({len(datos_validos):,} registros válidos)")
            
            return {
                'valor': round(iet_global, 1),
                'componentes': componentes,
                'interpretacion': self._interpretar_iet_avanzado(iet_global, componentes),
                'datos_utilizados': len(datos_validos),
                'periodo_analizado': self._obtener_periodo_datos(df_trabajo)
            }
            
        except Exception as e:
            self.logger.error(f"Error calculando IET: {e}")
            return {'error': f'Error calculando IET: {e}'}
    
    def calcular_ipc_datos_reales(self, filtros: Optional[Dict] = None) -> Dict:
        """
        Calcular IPC usando percentil 95 como referencia
        IPC = (Promedio / Percentil95) × 100%
        """
        col_conteo = self.mapeo_columnas.get('conteo')
        
        if not col_conteo:
            return {'error': 'Columna de conteo no encontrada'}
        
        try:
            df_trabajo = self._aplicar_filtros(filtros) if filtros else self.datos
            
            conteos_raw = df_trabajo[col_conteo].dropna()
            conteos = self._limpiar_datos_conteo(conteos_raw)
            
            if len(conteos) < self.umbrales['min_registros_validos']:
                return {
                    'valor': 0,
                    'interpretacion': f'Datos insuficientes ({len(conteos)} registros)'
                }
            
            conteo_promedio = conteos.mean()
            percentil_95 = conteos.quantile(0.95)
            conteo_maximo = conteos.max()
            
            # IPC usando percentil 95 como referencia (más realista)
            ipc = (conteo_promedio / percentil_95) * 100 if percentil_95 > 0 else 0
            
            componentes = {
                'conteo_promedio': round(conteo_promedio, 2),
                'conteo_p95': round(percentil_95, 2),
                'conteo_maximo': round(conteo_maximo, 2),
                'conteo_mediana': round(conteos.median(), 2),
                'conteo_total': round(conteos.sum(), 2),
                'datos_eliminados_limpieza': len(conteos_raw) - len(conteos),
                'formula': '(Promedio / Percentil95) × 100%'
            }
            
            self.logger.info(f"IPC: {ipc:.1f}% (Promedio={conteo_promedio:.0f}, P95={percentil_95:.0f})")
            
            return {
                'valor': round(ipc, 1),
                'componentes': componentes,
                'interpretacion': self._interpretar_ipc_avanzado(ipc, componentes),
                'datos_utilizados': len(conteos),
                'periodo_analizado': self._obtener_periodo_datos(df_trabajo)
            }
            
        except Exception as e:
            self.logger.error(f"Error calculando IPC: {e}")
            return {'error': f'Error calculando IPC: {e}'}
    
    def calcular_todas_las_formulas_reales(self, filtros: Optional[Dict] = None) -> Dict:
        """Calcular todas las fórmulas"""
        self.logger.info("=" * 80)
        self.logger.info("CALCULANDO TODAS LAS FÓRMULAS CON DATOS REALES")
        self.logger.info("=" * 80)
        
        resultados = {
            'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'estadisticas_datos': self.estadisticas_datos,
            'umbrales_aplicados': self.umbrales
        }
        
        resultados['IVM'] = self.calcular_ivm_datos_reales(filtros)
        resultados['IET'] = self.calcular_iet_datos_reales(filtros)
        resultados['IPC'] = self.calcular_ipc_datos_reales(filtros)
        
        return resultados
    
    def generar_reporte_completo(self, filtros: Optional[Dict] = None) -> str:
        """Generar reporte completo"""
        resultados = self.calcular_todas_las_formulas_reales(filtros)
        
        output = []
        output.append("=" * 80)
        output.append("REPORTE INDUSTRIAL COMPLETO - VERSIÓN FINAL")
        output.append(f"Generado: {resultados['timestamp']}")
        output.append("=" * 80)
        
        # Estadísticas
        stats = resultados['estadisticas_datos']
        output.append(f"\n[ESTADÍSTICAS GENERALES]")
        output.append(f"  Total registros: {stats['total_registros']:,}")
        output.append(f"  Período: {stats['periodo_analisis']}")
        output.append(f"  Registros con microparadas: {stats['registros_con_microparadas']:,}")
        output.append(f"  Registros con producción: {stats['registros_con_produccion']:,}")
        
        # Resultados
        output.append(f"\n[INDICADORES CALCULADOS]")
        output.append("=" * 80)
        
        for formula in ['IVM', 'IET', 'IPC']:
            resultado = resultados[formula]
            
            if 'error' in resultado:
                output.append(f"\n{formula}: ERROR - {resultado.get('error', 'Desconocido')}")
                if 'detalles' in resultado:
                    output.append(f"  Detalles: {resultado['detalles']}")
                continue
            
            output.append(f"\n{formula}: {resultado['valor']}%")
            output.append(f"  Datos utilizados: {resultado['datos_utilizados']:,}")
            output.append(f"  Interpretación: {resultado['interpretacion']}")
            
            if 'componentes' in resultado:
                comp = resultado['componentes']
                output.append(f"  Fórmula: {comp.get('formula', 'N/A')}")
                
                if formula == 'IVM':
                    output.append(f"  Media: {comp['mu_media']} min, Desv: {comp['sigma_desviacion']}")
                    output.append(f"  % Ceros original: {comp['porcentaje_ceros_original']}%")
                    output.append(f"  Outliers eliminados: {comp['datos_eliminados_limpieza']:,}")
                
                elif formula == 'IET':
                    output.append(f"  T. Disponible: {comp['t_disponible_total']:,} min")
                    output.append(f"  T. Microparadas: {comp['t_microparadas_total']:,} min")
                    output.append(f"  Registros válidos: {comp['registros_validos']:,}")
                
                elif formula == 'IPC':
                    output.append(f"  Promedio: {comp['conteo_promedio']:,}")
                    output.append(f"  P95: {comp['conteo_p95']:,}")
                    output.append(f"  Outliers eliminados: {comp['datos_eliminados_limpieza']:,}")
        
        output.append(f"\n" + "=" * 80)
        output.append(f"Reporte generado: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        return '\n'.join(output)
    
    # Métodos auxiliares
    def _aplicar_filtros(self, filtros: Dict) -> pd.DataFrame:
        """Aplicar filtros"""
        df = self.datos.copy()
        for col, val in filtros.items():
            if col in df.columns:
                df = df[df[col].isin(val)] if isinstance(val, (list, tuple)) else df[df[col] == val]
        return df
    
    def _obtener_periodo_datos(self, df: Optional[pd.DataFrame] = None) -> str:
        """Obtener período"""
        if df is None:
            df = self.datos
        if 'Fecha' in df.columns:
            fechas = df['Fecha'].dropna()
            if len(fechas) > 0:
                return f"{fechas.min().strftime('%Y-%m-%d')} a {fechas.max().strftime('%Y-%m-%d')}"
        return "N/A"
    
    def _interpretar_ivm_avanzado(self, valor: float, componentes: Dict) -> str:
        if valor <= 25:
            return "Variabilidad baja - operación estable"
        elif valor <= 50:
            return "Variabilidad moderada"
        elif valor <= 100:
            return "Variabilidad alta - requiere atención"
        else:
            return "Variabilidad muy alta - intervención urgente"
    
    def _interpretar_iet_avanzado(self, valor: float, componentes: Dict) -> str:
        if valor >= 90:
            return "Eficiencia excelente"
        elif valor >= 80:
            return "Eficiencia muy buena"
        elif valor >= 70:
            return "Eficiencia buena"
        else:
            return "Eficiencia baja - acción requerida"
    
    def _interpretar_ipc_avanzado(self, valor: float, componentes: Dict) -> str:
        if valor >= 85:
            return "Capacidad óptima"
        elif valor >= 70:
            return "Capacidad buena"
        elif valor >= 50:
            return "Capacidad aceptable"
        else:
            return "Capacidad limitada - optimización necesaria"
    
    def _generar_recomendaciones_ivm(self, valor: float, componentes: Dict) -> List[str]:
        recs = []
        if valor > 100:
            recs.append("Investigar causas de variabilidad extrema")
        if componentes['porcentaje_ceros_original'] > 50:
            recs.append("Alta proporción de períodos sin microparadas")
        return recs


# ============================================================================
# FUNCIÓN PRINCIPAL
# ============================================================================

def procesar_datos_completo(datos: pd.DataFrame, mostrar_diagnostico: bool = True) -> str:
    """Función principal para procesar datos industriales"""
    
    calculador = CalculadorFormulasIndustriales(datos)
    
    output = []
    
    if mostrar_diagnostico:
        output.append(calculador.diagnosticar_datos_completo())
        output.append("\n\n")
    
    output.append(calculador.generar_reporte_completo())
    
    return '\n'.join(output)