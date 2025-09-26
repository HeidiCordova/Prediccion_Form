import pandas as pd
import numpy as np
import datetime
import os
import re
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import logging

warnings.filterwarnings('ignore')
class CSVParserAvanzado:
    """Parser avanzado para CSV con formatos específicos industriales"""
    
    def __init__(self, ruta_archivo: str):
        self.ruta_archivo = ruta_archivo
        self.datos_raw = None
        self.datos_procesados = None
        self.separador_detectado = None
        self.encoding_detectado = 'utf-8'
        
        # Configurar logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def detectar_formato_csv(self) -> Dict[str, Any]:
        """Detectar formato específico del CSV industrial"""
        try:
            # Leer las primeras líneas para análisis
            with open(self.ruta_archivo, 'r', encoding='utf-8', errors='ignore') as f:
                primeras_lineas = [f.readline().strip() for _ in range(5)]
            
            # Analizar primera línea (encabezado)
            primera_linea = primeras_lineas[0] if primeras_lineas else ""
            
            # Detectar separadores posibles
            separadores_posibles = [';', ',', '\t', '|', ':']
            separador_detectado = None
            max_columnas = 0
            
            for sep in separadores_posibles:
                # Contar columnas con este separador
                if '"' in primera_linea:
                    # Caso especial: comillas dobles con separador
                    partes = self._parsear_linea_con_comillas(primera_linea, sep)
                    num_columnas = len(partes)
                else:
                    num_columnas = len(primera_linea.split(sep))
                
                self.logger.info(f"Separador '{sep}': {num_columnas} columnas detectadas")
                
                if num_columnas > max_columnas:
                    max_columnas = num_columnas
                    separador_detectado = sep
            
            # Información del formato detectado
            formato_info = {
                'separador': separador_detectado,
                'tiene_comillas': '"' in primera_linea,
                'columnas_esperadas': max_columnas,
                'encabezado_original': primera_linea,
                'total_lineas': len(primeras_lineas)
            }
            
            self.logger.info(f"Formato detectado: {formato_info}")
            return formato_info
            
        except Exception as e:
            self.logger.error(f"Error detectando formato: {e}")
            return {'separador': ';', 'tiene_comillas': True, 'columnas_esperadas': 0}
    
    def _parsear_linea_con_comillas(self, linea: str, separador: str) -> List[str]:
        """Parsear línea que tiene comillas dobles y separadores específicos"""
        partes = []
        
        # Método 1: Usar regex para encontrar contenido entre comillas
        if '"' in linea:
            # Patrón para encontrar contenido entre comillas
            patron = r'"([^"]*)"'
            matches = re.findall(patron, linea)
            
            if matches:
                partes = matches
            else:
                # Fallback: split normal
                partes = linea.split(separador)
        else:
            partes = linea.split(separador)
        
        # Limpiar espacios y comillas adicionales
        partes_limpias = []
        for parte in partes:
            parte_limpia = parte.strip().strip('"').strip()
            if parte_limpia:  # Solo agregar si no está vacía
                partes_limpias.append(parte_limpia)
        
        return partes_limpias
    
    def cargar_csv_formato_industrial(self) -> pd.DataFrame:
        """Cargar CSV con formato específico industrial"""
        try:
            self.logger.info(f"Analizando formato de: {self.ruta_archivo}")
            
            # 1. Detectar formato
            formato_info = self.detectar_formato_csv()
            separador = formato_info['separador']
            
            if formato_info['columnas_esperadas'] <= 1:
                self.logger.warning("Formato CSV problemático detectado. Intentando parsing manual...")
                return self._parsear_manual_csv()
            
            # 2. Intentar carga estándar con separador detectado
            try:
                self.datos_raw = pd.read_csv(
                    self.ruta_archivo, 
                    sep=separador,
                    encoding='utf-8',
                    quotechar='"',
                    skipinitialspace=True,
                    na_values=['\\N', 'NULL', 'null', '', 'NaN', 'nan']
                )
                
                if len(self.datos_raw.columns) > 1:
                    self.logger.info(f"CSV cargado exitosamente: {len(self.datos_raw)} filas, {len(self.datos_raw.columns)} columnas")
                    self.separador_detectado = separador
                    return self._procesar_datos_cargados()
                else:
                    raise ValueError("Carga estándar falló - solo 1 columna detectada")
                    
            except Exception as e:
                self.logger.warning(f"Carga estándar falló: {e}. Intentando parsing manual...")
                return self._parsear_manual_csv()
                
        except Exception as e:
            self.logger.error(f"Error crítico cargando CSV: {e}")
            raise
    
    def _parsear_manual_csv(self) -> pd.DataFrame:
        """Parsing manual para CSV con formato específico"""
        try:
            self.logger.info("Iniciando parsing manual del CSV...")
            
            # Leer archivo línea por línea
            with open(self.ruta_archivo, 'r', encoding='utf-8', errors='ignore') as f:
                lineas = f.readlines()
            
            self.logger.info(f"Total líneas leídas: {len(lineas)}")
            
            if not lineas:
                raise ValueError("Archivo CSV vacío")
            
            # Procesar encabezado (primera línea)
            encabezado_raw = lineas[0].strip()
            self.logger.info(f"Encabezado raw: {encabezado_raw[:200]}...")
            
            # Detectar separador en el encabezado
            if ';' in encabezado_raw and '"' in encabezado_raw:
                # Caso específico: separador ; con comillas dobles
                columnas = self._parsear_linea_con_comillas(encabezado_raw, ';')
                separador_usado = ';'
            elif ',' in encabezado_raw and '"' in encabezado_raw:
                columnas = self._parsear_linea_con_comillas(encabezado_raw, ',')
                separador_usado = ','
            else:
                # Intentar separadores uno por uno
                for sep in [';', ',', '\t', '|']:
                    test_columnas = encabezado_raw.split(sep)
                    if len(test_columnas) > 5:  # Asumir que hay al menos 5 columnas
                        columnas = [col.strip().strip('"') for col in test_columnas]
                        separador_usado = sep
                        break
                else:
                    raise ValueError("No se pudo detectar separador válido")
            
            self.logger.info(f"Columnas detectadas ({len(columnas)}): {columnas[:5]}...")
            self.separador_detectado = separador_usado
            
            # Procesar datos (resto de líneas)
            datos_filas = []
            lineas_procesadas = 0
            lineas_con_error = 0
            
            for i, linea in enumerate(lineas[1:], 1):  # Saltar encabezado
                try:
                    linea = linea.strip()
                    if not linea:
                        continue
                    
                    # Parsear línea con el mismo método que el encabezado
                    if '"' in linea:
                        valores = self._parsear_linea_con_comillas(linea, separador_usado)
                    else:
                        valores = [val.strip() for val in linea.split(separador_usado)]
                    
                    # Ajustar número de valores al número de columnas
                    while len(valores) < len(columnas):
                        valores.append('')  # Rellenar con valores vacíos
                    
                    if len(valores) > len(columnas):
                        valores = valores[:len(columnas)]  # Truncar si hay más valores
                    
                    datos_filas.append(valores)
                    lineas_procesadas += 1
                    
                    # Log de progreso cada 10,000 líneas
                    if lineas_procesadas % 10000 == 0:
                        self.logger.info(f"Procesadas {lineas_procesadas} líneas...")
                        
                except Exception as e:
                    lineas_con_error += 1
                    if lineas_con_error <= 5:  # Solo log primeros errores
                        self.logger.warning(f"Error en línea {i}: {e}")
                    continue
            
            self.logger.info(f"Parsing completado: {lineas_procesadas} líneas procesadas, {lineas_con_error} con errores")
            
            # Crear DataFrame
            df = pd.DataFrame(datos_filas, columns=columnas)
            
            # Limpiar nombres de columnas
            df.columns = [self._limpiar_nombre_columna(col) for col in df.columns]
            
            self.datos_raw = df
            return self._procesar_datos_cargados()
            
        except Exception as e:
            self.logger.error(f"Error en parsing manual: {e}")
            raise
    
    def _limpiar_nombre_columna(self, nombre: str) -> str:
        """Limpiar nombres de columnas"""
        # Remover comillas, espacios extra, caracteres especiales
        nombre_limpio = nombre.strip().strip('"').strip("'").strip()
        
        # Reemplazar caracteres problemáticos
        nombre_limpio = nombre_limpio.replace('\n', ' ').replace('\r', ' ')
        
        # Normalizar espacios múltiples
        nombre_limpio = ' '.join(nombre_limpio.split())
        
        return nombre_limpio
    
    def _procesar_datos_cargados(self) -> pd.DataFrame:
        """Procesar datos después de cargar"""
        try:
            self.logger.info("Procesando datos cargados...")
            
            df = self.datos_raw.copy()
            
            # 1. Mostrar información básica
            self.logger.info(f"Datos antes del procesamiento: {len(df)} filas, {len(df.columns)} columnas")
            self.logger.info(f"Columnas: {list(df.columns)}")
            
            # 2. Reemplazar valores nulos específicos
            valores_nulos = ['\\N', 'NULL', 'null', '', 'NaN', 'nan', 'None', 'none']
            df = df.replace(valores_nulos, np.nan)
            
            # 3. Identificar columnas numéricas por contenido
            columnas_numericas = []
            for col in df.columns:
                # Intentar conversión a numérico en una muestra
                muestra = df[col].dropna().head(100)
                if len(muestra) > 0:
                    try:
                        pd.to_numeric(muestra, errors='coerce')
                        # Si más del 70% se puede convertir, es numérica
                        numericos = pd.to_numeric(muestra, errors='coerce').dropna()
                        if len(numericos) / len(muestra) > 0.7:
                            columnas_numericas.append(col)
                    except:
                        pass
            
            self.logger.info(f"Columnas numéricas detectadas: {columnas_numericas}")
            
            # 4. Convertir columnas numéricas
            for col in columnas_numericas:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # 5. Procesar fechas (buscar columnas con 'fecha' en el nombre)
            columnas_fecha = [col for col in df.columns if 'fecha' in col.lower()]
            for col in columnas_fecha:
                try:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                    self.logger.info(f"Columna '{col}' convertida a fecha")
                except:
                    pass
            
            # 6. Eliminar filas completamente vacías
            filas_antes = len(df)
            df = df.dropna(how='all')
            filas_despues = len(df)
            
            if filas_antes != filas_despues:
                self.logger.info(f"Eliminadas {filas_antes - filas_despues} filas vacías")
            
            self.datos_procesados = df
            self.logger.info(f"Datos procesados: {len(df)} filas, {len(df.columns)} columnas")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error procesando datos: {e}")
            raise
    
    def mostrar_resumen_datos(self) -> str:
        """Mostrar resumen de los datos cargados"""
        if self.datos_procesados is None:
            return "No hay datos cargados"
        
        df = self.datos_procesados
        output = []
        
        output.append("=" * 60)
        output.append("RESUMEN DE DATOS CARGADOS")
        output.append("=" * 60)
        
        output.append(f"Archivo: {Path(self.ruta_archivo).name}")
        output.append(f"Separador detectado: '{self.separador_detectado}'")
        output.append(f"Total filas: {len(df):,}")
        output.append(f"Total columnas: {len(df.columns)}")
        
        output.append(f"\nCOLUMNAS DISPONIBLES:")
        output.append("-" * 30)
        for i, col in enumerate(df.columns, 1):
            tipo_datos = str(df[col].dtype)
            valores_no_nulos = df[col].count()
            pct_completo = (valores_no_nulos / len(df)) * 100
            
            output.append(f"{i:2d}. {col}")
            output.append(f"    Tipo: {tipo_datos}")
            output.append(f"    Datos: {valores_no_nulos:,} ({pct_completo:.1f}% completo)")
            
            # Mostrar muestra de valores únicos para columnas no numéricas
            if not pd.api.types.is_numeric_dtype(df[col]):
                valores_unicos = df[col].dropna().unique()[:3]
                output.append(f"    Muestra: {list(valores_unicos)}")
            
            output.append("")
        
        # Estadísticas generales
        output.append("ESTADISTICAS GENERALES:")
        output.append("-" * 30)
        output.append(f"Filas con datos completos: {df.dropna().shape[0]:,}")
        output.append(f"Porcentaje de completitud: {(df.dropna().shape[0] / len(df)) * 100:.1f}%")
        
        # Columnas críticas para fórmulas industriales
        columnas_criticas = [
            'T. de Microparadas', 'T. Disponible', 'Conteo', 
            'T. de P. No Programada', 'Id Turno', 'Fecha'
        ]
        
        output.append(f"\nCOLUMNAS CRITICAS ENCONTRADAS:")
        output.append("-" * 30)
        for col_critica in columnas_criticas:
            if col_critica in df.columns:
                output.append(f"✓ {col_critica}")
            else:
                # Buscar columnas similares
                similar = self._buscar_columna_similar(col_critica, df.columns)
                if similar:
                    output.append(f"? {col_critica} (similar: {similar})")
                else:
                    output.append(f"✗ {col_critica} (no encontrada)")
        
        return '\n'.join(output)
    
    def _buscar_columna_similar(self, objetivo: str, columnas_disponibles: List[str]) -> Optional[str]:
        """Buscar columna con nombre similar"""
        objetivo_clean = objetivo.lower().replace(' ', '').replace('.', '')
        
        for col in columnas_disponibles:
            col_clean = col.lower().replace(' ', '').replace('.', '')
            
            # Verificar coincidencia parcial
            if objetivo_clean in col_clean or col_clean in objetivo_clean:
                return col
        
        return None