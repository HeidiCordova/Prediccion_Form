import os
import sys
import time
import datetime
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import warnings
import logging
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import joblib

warnings.filterwarnings('ignore')

# Configurar logging
logging.basicConfig(level=logging.INFO)

@dataclass
class PrediccionResultado:
    """Resultado de predicción estructurado - Solo valores numéricos"""
    valor_predicho: float
    confianza: float
    es_anomalia: bool
    factores_influencia: Dict[str, float]
    horizonte_temporal: str
    timestamp: datetime.datetime
    metricas_modelo: Dict[str, float]

class ModeloPrediccionBase(ABC):
    """Modelo base abstracto para predicciones industriales"""
    
    def __init__(self, nombre_indicador: str, horizonte_horas: int = 2):
        self.nombre_indicador = nombre_indicador
        self.horizonte_horas = horizonte_horas
        self.modelo = None
        self.scaler = StandardScaler()
        self.modelo_entrenado = False
        self.metricas_entrenamiento = {}
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        self.detector_anomalias = None
        self.contamination_calculada = None
    
    def _calcular_contamination_dinamico(self, datos: np.ndarray) -> float:
        """Calcular contamination dinámicamente usando método IQR"""
        try:
            # Método IQR para detectar outliers
            Q1 = np.percentile(datos, 25)
            Q3 = np.percentile(datos, 75)
            IQR = Q3 - Q1
            
            # Límites para outliers
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Contar outliers
            outliers = ((datos < lower_bound) | (datos > upper_bound))
            contamination_rate = np.sum(outliers) / len(datos)
            
            # Límites razonables (entre 0.01 y 0.3)
            contamination_final = max(0.01, min(0.3, contamination_rate))
            
            self.logger.info(f"Contamination calculada para {self.nombre_indicador}: {contamination_final:.3f} "
                           f"({np.sum(outliers)} outliers de {len(datos)} datos)")
            
            return contamination_final
            
        except Exception as e:
            self.logger.warning(f"Error calculando contamination dinámico: {e}. Usando valor por defecto 0.05")
            return 0.05
    
    @abstractmethod
    def entrenar(self, datos: pd.DataFrame) -> Dict[str, float]:
        """Entrenar modelo con datos históricos"""
        pass
    
    @abstractmethod
    def predecir(self, datos_entrada: pd.DataFrame) -> PrediccionResultado:
        """Realizar predicción"""
        pass
    
    def validar_modelo(self, datos_test: pd.DataFrame) -> Dict[str, float]:
        """Validar precisión del modelo"""
        if not self.modelo_entrenado:
            return {'error': 'Modelo no entrenado'}
        
        try:
            return self.metricas_entrenamiento
        except Exception as e:
            self.logger.error(f"Error validando modelo: {e}")
            return {'error': str(e)}

class ModeloPrediccionIVM(ModeloPrediccionBase):
    """Modelo específico para predicción de IVM (Índice de Variabilidad de Microparadas)"""
    
    def __init__(self, horizonte_horas: int = 2):
        super().__init__('IVM', horizonte_horas)
        self.modelo = RandomForestRegressor(n_estimators=100, random_state=42)
    
    def _crear_caracteristicas_temporales(self, datos: pd.DataFrame) -> pd.DataFrame:
        """Crear características temporales para predicción"""
        df = datos.copy()
        
        # Asegurar que Fecha es datetime
        if 'Fecha' in df.columns:
            df['Fecha'] = pd.to_datetime(df['Fecha'])
            df = df.sort_values('Fecha')
            
            # Características temporales
            df['hora'] = df['Fecha'].dt.hour
            df['dia_semana'] = df['Fecha'].dt.dayofweek
            df['es_fin_semana'] = (df['dia_semana'] >= 5).astype(int)
            
            # Ventanas móviles de microparadas
            if 'T. de Microparadas' in df.columns:
                df['microparadas_media_1h'] = df['T. de Microparadas'].rolling(window=12, min_periods=1).mean()
                df['microparadas_std_1h'] = df['T. de Microparadas'].rolling(window=12, min_periods=1).std()
                df['microparadas_max_1h'] = df['T. de Microparadas'].rolling(window=12, min_periods=1).max()
                
                # Tendencia
                df['microparadas_tendencia'] = df['T. de Microparadas'].rolling(window=6).apply(
                    lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0
                )
            
            # Características de producción
            if 'Conteo' in df.columns:
                df['produccion_velocidad'] = df['Conteo'].rolling(window=6, min_periods=1).mean()
                df['produccion_estable'] = (df['Conteo'].rolling(window=6).std() < df['Conteo'].rolling(window=6).mean() * 0.1).astype(int)
            
            # Características de turnos
            if 'Id Turno' in df.columns:
                df['cambio_turno'] = (df['Id Turno'] != df['Id Turno'].shift(1)).astype(int)
        
        return df
    
    def entrenar(self, datos: pd.DataFrame) -> Dict[str, float]:
        """Entrenar modelo de predicción IVM"""
        try:
            self.logger.info(f"Entrenando modelo IVM con {len(datos)} registros")
            
            # Crear características
            df_features = self._crear_caracteristicas_temporales(datos)
            
            # Calcular IVM objetivo (ventana futura)
            if 'T. de Microparadas' in df_features.columns:
                # IVM = (std/mean) * 100 en ventana de tiempo
                ventana = 12  # 1 hora (asumiendo datos cada 5 min)
                df_features['ivm_objetivo'] = df_features['T. de Microparadas'].rolling(
                    window=ventana, min_periods=6
                ).apply(lambda x: (x.std() / x.mean()) * 100 if x.mean() > 0 else 0).shift(-ventana)
            
            # Seleccionar características para entrenamiento
            feature_cols = [
                'hora', 'dia_semana', 'es_fin_semana',
                'microparadas_media_1h', 'microparadas_std_1h', 'microparadas_max_1h',
                'microparadas_tendencia', 'produccion_velocidad', 'produccion_estable',
                'cambio_turno'
            ]
            
            # Filtrar columnas que existen
            feature_cols = [col for col in feature_cols if col in df_features.columns]
            
            # Preparar datos de entrenamiento
            df_clean = df_features[feature_cols + ['ivm_objetivo']].dropna()
            
            if len(df_clean) < 100:
                raise ValueError(f"Datos insuficientes para entrenamiento: {len(df_clean)}")
            
            X = df_clean[feature_cols]
            y = df_clean['ivm_objetivo']
            
            # Dividir en entrenamiento y prueba
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Escalar características
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Entrenar modelo principal
            self.modelo.fit(X_train_scaled, y_train)
            
            # Calcular contamination dinámicamente basado en datos de entrenamiento
            self.contamination_calculada = self._calcular_contamination_dinamico(X_train_scaled.flatten())
            
            # Crear detector de anomalías con contamination calculada
            self.detector_anomalias = IsolationForest(
                contamination=self.contamination_calculada, 
                random_state=42
            )
            self.detector_anomalias.fit(X_train_scaled)
            
            # Calcular métricas
            y_pred = self.modelo.predict(X_test_scaled)
            
            self.metricas_entrenamiento = {
                'mae': mean_absolute_error(y_test, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                'r2': r2_score(y_test, y_pred),
                'registros_entrenamiento': len(X_train),
                'registros_prueba': len(X_test),
                'caracteristicas_usadas': len(feature_cols),
                'contamination_calculada': self.contamination_calculada
            }
            
            self.modelo_entrenado = True
            self.feature_cols = feature_cols
            
            self.logger.info(f"Modelo IVM entrenado. R²: {self.metricas_entrenamiento['r2']:.3f}, "
                           f"MAE: {self.metricas_entrenamiento['mae']:.2f}, "
                           f"Contamination: {self.contamination_calculada:.3f}")
            
            return self.metricas_entrenamiento
            
        except Exception as e:
            self.logger.error(f"Error entrenando modelo IVM: {e}")
            raise
    
    def predecir(self, datos_entrada: pd.DataFrame) -> PrediccionResultado:
        """Predecir IVM futuro"""
        if not self.modelo_entrenado:
            raise ValueError("Modelo no entrenado")
        
        try:
            # Crear características temporales
            df_features = self._crear_caracteristicas_temporales(datos_entrada)
            
            # Usar últimos datos disponibles
            datos_recientes = df_features.tail(1)
            
            # Preparar características
            X_pred = datos_recientes[self.feature_cols].fillna(0)
            X_pred_scaled = self.scaler.transform(X_pred)
            
            # Realizar predicción
            valor_predicho = self.modelo.predict(X_pred_scaled)[0]
            
            # Detectar anomalías solo si el detector está entrenado
            es_anomalia = False
            if self.detector_anomalias is not None:
                es_anomalia = self.detector_anomalias.predict(X_pred_scaled)[0] == -1
            
            # Calcular confianza basada en R²
            r2_score_val = self.metricas_entrenamiento.get('r2', 0)
            confianza_base = max(0, min(100, r2_score_val * 100))
            confianza = confianza_base * (0.8 if es_anomalia else 1.0)
            
            # Factores de influencia (importancia de características)
            factores_influencia = {}
            if hasattr(self.modelo, 'feature_importances_'):
                for i, col in enumerate(self.feature_cols):
                    factores_influencia[col] = round(self.modelo.feature_importances_[i] * 100, 1)
            
            return PrediccionResultado(
                valor_predicho=round(valor_predicho, 2),
                confianza=round(confianza, 1),
                es_anomalia=es_anomalia,
                factores_influencia=factores_influencia,
                horizonte_temporal=f"{self.horizonte_horas} horas",
                timestamp=datetime.datetime.now(),
                metricas_modelo=self.metricas_entrenamiento
            )
            
        except Exception as e:
            self.logger.error(f"Error en predicción IVM: {e}")
            raise

class ModeloPrediccionIET(ModeloPrediccionBase):
    """Modelo específico para predicción de IET (Índice de Eficiencia Temporal)"""
    
    def __init__(self, horizonte_horas: int = 2):
        super().__init__('IET', horizonte_horas)
        self.modelo = RandomForestRegressor(n_estimators=100, random_state=42)
    
    def entrenar(self, datos: pd.DataFrame) -> Dict[str, float]:
        """Entrenar modelo de predicción IET"""
        try:
            self.logger.info(f"Entrenando modelo IET con {len(datos)} registros")
            
            # Filtrar datos válidos
            datos_validos = datos[
                (datos['T. Disponible'].notna()) & 
                (datos['T. Disponible'] > 0) &
                (datos['T. de Microparadas'].notna()) &
                (datos['T. de Microparadas'] >= 0)
            ].copy()
            
            if len(datos_validos) < 100:
                raise ValueError("Datos insuficientes para IET")
            
            # Calcular IET actual
            datos_validos['IET_actual'] = (
                (datos_validos['T. Disponible'] - datos_validos['T. de Microparadas']) / 
                datos_validos['T. Disponible']
            ) * 100
            
            # Crear características temporales básicas
            if 'Fecha' in datos_validos.columns:
                datos_validos['Fecha'] = pd.to_datetime(datos_validos['Fecha'])
                datos_validos['hora'] = datos_validos['Fecha'].dt.hour
                datos_validos['dia_semana'] = datos_validos['Fecha'].dt.dayofweek
            
            # Características de eficiencia
            datos_validos['eficiencia_media_1h'] = datos_validos['IET_actual'].rolling(window=12, min_periods=1).mean()
            datos_validos['tiempo_disponible_ratio'] = datos_validos['T. Disponible'] / datos_validos['T. Disponible'].mean()
            datos_validos['microparadas_ratio'] = datos_validos['T. de Microparadas'] / datos_validos['T. de Microparadas'].mean()
            
            # Crear objetivo (IET futuro)
            datos_validos['IET_objetivo'] = datos_validos['IET_actual'].shift(-12)  # 1 hora futura
            
            # Características de entrenamiento
            feature_cols = ['hora', 'dia_semana', 'eficiencia_media_1h', 'tiempo_disponible_ratio', 'microparadas_ratio']
            feature_cols = [col for col in feature_cols if col in datos_validos.columns]
            
            # Preparar datos
            df_clean = datos_validos[feature_cols + ['IET_objetivo']].dropna()
            
            if len(df_clean) < 50:
                raise ValueError("Datos insuficientes después de limpieza")
            
            X = df_clean[feature_cols]
            y = df_clean['IET_objetivo']
            
            # Dividir datos
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Escalar y entrenar
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            self.modelo.fit(X_train_scaled, y_train)
            
            # Calcular contamination dinámicamente
            self.contamination_calculada = self._calcular_contamination_dinamico(X_train_scaled.flatten())
            
            # Crear detector de anomalías
            self.detector_anomalias = IsolationForest(
                contamination=self.contamination_calculada, 
                random_state=42
            )
            self.detector_anomalias.fit(X_train_scaled)
            
            # Métricas
            y_pred = self.modelo.predict(X_test_scaled)
            self.metricas_entrenamiento = {
                'mae': mean_absolute_error(y_test, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                'r2': r2_score(y_test, y_pred),
                'registros_entrenamiento': len(X_train),
                'contamination_calculada': self.contamination_calculada
            }
            
            self.modelo_entrenado = True
            self.feature_cols = feature_cols
            
            self.logger.info(f"Modelo IET entrenado. R²: {self.metricas_entrenamiento['r2']:.3f}, "
                           f"Contamination: {self.contamination_calculada:.3f}")
            
            return self.metricas_entrenamiento
            
        except Exception as e:
            self.logger.error(f"Error entrenando modelo IET: {e}")
            raise
    
    def predecir(self, datos_entrada: pd.DataFrame) -> PrediccionResultado:
        """Predecir IET futuro"""
        if not self.modelo_entrenado:
            raise ValueError("Modelo IET no entrenado")
        
        try:
            # Filtrar y calcular IET actual
            datos_validos = datos_entrada[
                (datos_entrada['T. Disponible'].notna()) & 
                (datos_entrada['T. Disponible'] > 0) &
                (datos_entrada['T. de Microparadas'].notna()) &
                (datos_entrada['T. de Microparadas'] >= 0)
            ].copy()
            
            if len(datos_validos) == 0:
                raise ValueError("No hay datos válidos para predicción IET")
            
            # Calcular IET actual
            datos_validos['IET_actual'] = (
                (datos_validos['T. Disponible'] - datos_validos['T. de Microparadas']) / 
                datos_validos['T. Disponible']
            ) * 100
            
            # Características temporales
            if 'Fecha' in datos_validos.columns:
                datos_validos['Fecha'] = pd.to_datetime(datos_validos['Fecha'])
                datos_validos['hora'] = datos_validos['Fecha'].dt.hour
                datos_validos['dia_semana'] = datos_validos['Fecha'].dt.dayofweek
            
            # Características de eficiencia
            datos_validos['eficiencia_media_1h'] = datos_validos['IET_actual'].rolling(window=12, min_periods=1).mean()
            datos_validos['tiempo_disponible_ratio'] = datos_validos['T. Disponible'] / datos_validos['T. Disponible'].mean()
            datos_validos['microparadas_ratio'] = datos_validos['T. de Microparadas'] / datos_validos['T. de Microparadas'].mean()
            
            # Usar últimos datos
            datos_recientes = datos_validos.tail(1)
            X_pred = datos_recientes[self.feature_cols].fillna(0)
            X_pred_scaled = self.scaler.transform(X_pred)
            
            # Predicción
            valor_predicho = self.modelo.predict(X_pred_scaled)[0]
            es_anomalia = False
            if self.detector_anomalias is not None:
                es_anomalia = self.detector_anomalias.predict(X_pred_scaled)[0] == -1
            
            # Confianza
            r2_score_val = self.metricas_entrenamiento.get('r2', 0)
            confianza_base = max(0, min(100, r2_score_val * 100))
            confianza = confianza_base * (0.8 if es_anomalia else 1.0)
            
            # Factores de influencia
            factores_influencia = {}
            if hasattr(self.modelo, 'feature_importances_'):
                for i, col in enumerate(self.feature_cols):
                    factores_influencia[col] = round(self.modelo.feature_importances_[i] * 100, 1)
            
            return PrediccionResultado(
                valor_predicho=round(valor_predicho, 2),
                confianza=round(confianza, 1),
                es_anomalia=es_anomalia,
                factores_influencia=factores_influencia,
                horizonte_temporal=f"{self.horizonte_horas} horas",
                timestamp=datetime.datetime.now(),
                metricas_modelo=self.metricas_entrenamiento
            )
            
        except Exception as e:
            self.logger.error(f"Error en predicción IET: {e}")
            raise

class ModeloPrediccionIPC(ModeloPrediccionBase):
    """Modelo específico para predicción de IPC (Índice de Productividad de Conteo)"""
    
    def __init__(self, horizonte_horas: int = 2):
        super().__init__('IPC', horizonte_horas)
        self.modelo = RandomForestRegressor(n_estimators=100, random_state=42)
    
    def entrenar(self, datos: pd.DataFrame) -> Dict[str, float]:
        """Entrenar modelo de predicción IPC"""
        try:
            self.logger.info(f"Entrenando modelo IPC con {len(datos)} registros")
            
            # Filtrar datos válidos para IPC
            datos_validos = datos[
                (datos['Conteo'].notna()) & 
                (datos['Conteo'] > 0) &
                (datos['T. Disponible'].notna()) &
                (datos['T. Disponible'] > 0)
            ].copy()
            
            if len(datos_validos) < 100:
                raise ValueError("Datos insuficientes para IPC")
            
            # Calcular IPC actual (productividad por minuto disponible)
            datos_validos['IPC_actual'] = (datos_validos['Conteo'] / datos_validos['T. Disponible']) * 100
            
            # Características temporales
            if 'Fecha' in datos_validos.columns:
                datos_validos['Fecha'] = pd.to_datetime(datos_validos['Fecha'])
                datos_validos['hora'] = datos_validos['Fecha'].dt.hour
                datos_validos['dia_semana'] = datos_validos['Fecha'].dt.dayofweek
            
            # Características de productividad
            datos_validos['conteo_media_1h'] = datos_validos['Conteo'].rolling(window=12, min_periods=1).mean()
            datos_validos['productividad_ratio'] = datos_validos['IPC_actual'] / datos_validos['IPC_actual'].mean()
            
            # Crear objetivo (IPC futuro)
            datos_validos['IPC_objetivo'] = datos_validos['IPC_actual'].shift(-12)
            
            # Características de entrenamiento
            feature_cols = ['hora', 'dia_semana', 'conteo_media_1h', 'productividad_ratio']
            feature_cols = [col for col in feature_cols if col in datos_validos.columns]
            
            # Preparar datos
            df_clean = datos_validos[feature_cols + ['IPC_objetivo']].dropna()
            
            if len(df_clean) < 50:
                raise ValueError("Datos insuficientes después de limpieza")
            
            X = df_clean[feature_cols]
            y = df_clean['IPC_objetivo']
            
            # Dividir y entrenar
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            self.modelo.fit(X_train_scaled, y_train)
            
            # Calcular contamination dinámicamente
            self.contamination_calculada = self._calcular_contamination_dinamico(X_train_scaled.flatten())
            
            # Crear detector de anomalías
            self.detector_anomalias = IsolationForest(
                contamination=self.contamination_calculada, 
                random_state=42
            )
            self.detector_anomalias.fit(X_train_scaled)
            
            # Métricas
            y_pred = self.modelo.predict(X_test_scaled)
            self.metricas_entrenamiento = {
                'mae': mean_absolute_error(y_test, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                'r2': r2_score(y_test, y_pred),
                'registros_entrenamiento': len(X_train),
                'contamination_calculada': self.contamination_calculada
            }
            
            self.modelo_entrenado = True
            self.feature_cols = feature_cols
            
            self.logger.info(f"Modelo IPC entrenado. R²: {self.metricas_entrenamiento['r2']:.3f}, "
                           f"Contamination: {self.contamination_calculada:.3f}")
            
            return self.metricas_entrenamiento
            
        except Exception as e:
            self.logger.error(f"Error entrenando modelo IPC: {e}")
            raise
    
    def predecir(self, datos_entrada: pd.DataFrame) -> PrediccionResultado:
        """Predecir IPC futuro"""
        if not self.modelo_entrenado:
            raise ValueError("Modelo IPC no entrenado")
        
        try:
            # Implementación similar a IET pero para IPC
            datos_validos = datos_entrada[
                (datos_entrada['Conteo'].notna()) & 
                (datos_entrada['Conteo'] > 0) &
                (datos_entrada['T. Disponible'].notna()) &
                (datos_entrada['T. Disponible'] > 0)
            ].copy()
            
            if len(datos_validos) == 0:
                raise ValueError("No hay datos válidos para predicción IPC")
            
            # Calcular características necesarias
            datos_validos['IPC_actual'] = (datos_validos['Conteo'] / datos_validos['T. Disponible']) * 100
            
            if 'Fecha' in datos_validos.columns:
                datos_validos['Fecha'] = pd.to_datetime(datos_validos['Fecha'])
                datos_validos['hora'] = datos_validos['Fecha'].dt.hour
                datos_validos['dia_semana'] = datos_validos['Fecha'].dt.dayofweek
            
            datos_validos['conteo_media_1h'] = datos_validos['Conteo'].rolling(window=12, min_periods=1).mean()
            datos_validos['productividad_ratio'] = datos_validos['IPC_actual'] / datos_validos['IPC_actual'].mean()
            
            # Predicción
            datos_recientes = datos_validos.tail(1)
            X_pred = datos_recientes[self.feature_cols].fillna(0)
            X_pred_scaled = self.scaler.transform(X_pred)
            
            valor_predicho = self.modelo.predict(X_pred_scaled)[0]
            es_anomalia = False
            if self.detector_anomalias is not None:
                es_anomalia = self.detector_anomalias.predict(X_pred_scaled)[0] == -1
            
            # Confianza
            r2_score_val = self.metricas_entrenamiento.get('r2', 0)
            confianza_base = max(0, min(100, r2_score_val * 100))
            confianza = confianza_base * (0.8 if es_anomalia else 1.0)
            
            # Factores de influencia
            factores_influencia = {}
            if hasattr(self.modelo, 'feature_importances_'):
                for i, col in enumerate(self.feature_cols):
                    factores_influencia[col] = round(self.modelo.feature_importances_[i] * 100, 1)
            
            return PrediccionResultado(
                valor_predicho=round(valor_predicho, 2),
                confianza=round(confianza, 1),
                es_anomalia=es_anomalia,
                factores_influencia=factores_influencia,
                horizonte_temporal=f"{self.horizonte_horas} horas",
                timestamp=datetime.datetime.now(),
                metricas_modelo=self.metricas_entrenamiento
            )
            
        except Exception as e:
            self.logger.error(f"Error en predicción IPC: {e}")
            raise

class GestorPredicciones:
    """Gestor principal para coordinar todos los modelos predictivos"""
    
    def __init__(self):
        self.modelos = {
            'IVM': ModeloPrediccionIVM(),
            'IET': ModeloPrediccionIET(),
            'IPC': ModeloPrediccionIPC()
        }
        self.logger = logging.getLogger("GestorPredicciones")
        self.modelos_entrenados = False
    
    def entrenar_todos_modelos(self, datos: pd.DataFrame) -> Dict[str, Dict]:
        """Entrenar todos los modelos predictivos"""
        resultados_entrenamiento = {}
        
        for nombre, modelo in self.modelos.items():
            try:
                self.logger.info(f"Entrenando modelo {nombre}...")
                resultado = modelo.entrenar(datos)
                resultados_entrenamiento[nombre] = resultado
                r2_score = resultado.get('r2', 0)
                contamination = resultado.get('contamination_calculada', 'N/A')
                self.logger.info(f"Modelo {nombre} entrenado. R²: {r2_score:.3f}, Contamination: {contamination}")
            except Exception as e:
                self.logger.error(f"Error entrenando {nombre}: {e}")
                resultados_entrenamiento[nombre] = {'error': str(e)}
        
        self.modelos_entrenados = True
        return resultados_entrenamiento
    
    def predecir_todos_indicadores(self, datos: pd.DataFrame, horizonte_horas: int = 2) -> Dict[str, PrediccionResultado]:
        """Realizar predicciones para todos los indicadores"""
        if not self.modelos_entrenados:
            raise ValueError("Modelos no entrenados. Ejecutar entrenar_todos_modelos() primero.")
        
        predicciones = {}
        
        for nombre, modelo in self.modelos.items():
            try:
                modelo.horizonte_horas = horizonte_horas
                prediccion = modelo.predecir(datos)
                predicciones[nombre] = prediccion
                anomalia_str = " (ANOMALÍA)" if prediccion.es_anomalia else ""
                self.logger.info(f"Predicción {nombre}: {prediccion.valor_predicho:.2f}{anomalia_str}")
            except Exception as e:
                self.logger.error(f"Error prediciendo {nombre}: {e}")
                predicciones[nombre] = None
        
        return predicciones
    
    def generar_reporte_predicciones(self, predicciones: Dict[str, PrediccionResultado], incluir_detalles: bool = True) -> str:
        """Generar reporte completo de predicciones"""
        output = []
        
        output.append("=" * 80)
        output.append("REPORTE DE PREDICCIONES INDUSTRIALES")
        output.append(f"Generado: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        output.append("=" * 80)
        
        # Resumen ejecutivo
        predicciones_validas = [p for p in predicciones.values() if p is not None]
        anomalias_detectadas = sum(1 for p in predicciones_validas if p.es_anomalia)
        
        output.append(f"\n[RESUMEN EJECUTIVO]")
        output.append(f"Total predicciones: {len(predicciones_validas)}")
        output.append(f"Anomalías detectadas: {anomalias_detectadas}")
        output.append(f"Predicciones normales: {len(predicciones_validas) - anomalias_detectadas}")
        
        # Predicciones detalladas
        for nombre, prediccion in predicciones.items():
            if prediccion is None:
                output.append(f"\n[{nombre}] ERROR - No se pudo generar predicción")
                continue
            
            anomalia_tag = " ⚠️ ANOMALÍA" if prediccion.es_anomalia else ""
            output.append(f"\n[{nombre}]{anomalia_tag}")
            output.append(f"  Valor predicho: {prediccion.valor_predicho:.2f}")
            output.append(f"  Confianza: {prediccion.confianza:.1f}%")
            output.append(f"  Horizonte: {prediccion.horizonte_temporal}")
            
            # Mostrar contamination calculada si está disponible
            if 'contamination_calculada' in prediccion.metricas_modelo:
                cont = prediccion.metricas_modelo['contamination_calculada']
                output.append(f"  Contamination calculada: {cont:.3f}")
            
            if incluir_detalles and prediccion.factores_influencia:
                output.append(f"  Factores más influyentes:")
                factores_ordenados = sorted(prediccion.factores_influencia.items(), 
                                          key=lambda x: x[1], reverse=True)
                for factor, importancia in factores_ordenados[:3]:
                    output.append(f"    - {factor}: {importancia:.1f}%")
        
        return '\n'.join(output)
    
    def guardar_modelos(self, ruta_directorio: str):
        """Guardar modelos entrenados"""
        os.makedirs(ruta_directorio, exist_ok=True)
        
        for nombre, modelo in self.modelos.items():
            if modelo.modelo_entrenado:
                ruta_archivo = os.path.join(ruta_directorio, f"modelo_{nombre.lower()}.joblib")
                joblib.dump(modelo, ruta_archivo)
                self.logger.info(f"Modelo {nombre} guardado en {ruta_archivo}")
    
    def cargar_modelos(self, ruta_directorio: str):
        """Cargar modelos previamente entrenados"""
        for nombre in self.modelos.keys():
            ruta_archivo = os.path.join(ruta_directorio, f"modelo_{nombre.lower()}.joblib")
            if os.path.exists(ruta_archivo):
                self.modelos[nombre] = joblib.load(ruta_archivo)
                self.logger.info(f"Modelo {nombre} cargado desde {ruta_archivo}")
        
        self.modelos_entrenados = True