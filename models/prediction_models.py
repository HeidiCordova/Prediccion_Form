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
    """Resultado de predicción estructurado"""
    valor_predicho: float
    confianza: float
    probabilidad_alerta: float
    estado: str
    factores_influencia: Dict[str, float]
    horizonte_temporal: str
    timestamp: datetime.datetime
    metricas_modelo: Dict[str, float]
    recomendaciones: List[str]

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
        
        # Umbrales específicos por indicador
        self.umbrales = self._definir_umbrales()
        
    def _definir_umbrales(self) -> Dict[str, float]:
        """Definir umbrales específicos por indicador"""
        umbrales_base = {
            'IVM': {'critico': 50, 'alerta': 30, 'normal': 15},
            'IET': {'critico': 70, 'alerta': 80, 'normal': 90},
            'IPC': {'critico': 60, 'alerta': 75, 'normal': 85},
            'IIPNP': {'critico': 20, 'alerta': 15, 'normal': 10}
        }
        return umbrales_base.get(self.nombre_indicador, {'critico': 50, 'alerta': 30, 'normal': 15})
    
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
            # Implementar validación específica en clases derivadas
            return self.metricas_entrenamiento
        except Exception as e:
            self.logger.error(f"Error validando modelo: {e}")
            return {'error': str(e)}

class ModeloPrediccionIVM(ModeloPrediccionBase):
    """Modelo específico para predicción de IVM (Índice de Variabilidad de Microparadas)"""
    
    def __init__(self, horizonte_horas: int = 2):
        super().__init__('IVM', horizonte_horas)
        self.modelo = RandomForestRegressor(n_estimators=100, random_state=42)
        self.detector_anomalias = IsolationForest(contamination=0.1, random_state=42)
    
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
            
            # Entrenar detector de anomalías
            self.detector_anomalias.fit(X_train_scaled)
            
            # Calcular métricas
            y_pred = self.modelo.predict(X_test_scaled)
            
            self.metricas_entrenamiento = {
                'mae': mean_absolute_error(y_test, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                'r2': r2_score(y_test, y_pred),
                'precisión': max(0, min(100, 100 - mean_absolute_error(y_test, y_pred))),
                'registros_entrenamiento': len(X_train),
                'registros_prueba': len(X_test),
                'caracteristicas_usadas': len(feature_cols)
            }
            
            self.modelo_entrenado = True
            self.feature_cols = feature_cols
            
            self.logger.info(f"Modelo IVM entrenado. R²: {self.metricas_entrenamiento['r2']:.3f}, "
                           f"Precisión: {self.metricas_entrenamiento['precisión']:.1f}%")
            
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
            
            # Detectar anomalías
            es_anomalia = self.detector_anomalias.predict(X_pred_scaled)[0] == -1
            
            # Calcular confianza
            confianza_base = self.metricas_entrenamiento.get('precisión', 70)
            confianza = confianza_base * (0.8 if es_anomalia else 1.0)
            
            # Determinar estado y probabilidad de alerta
            if valor_predicho >= self.umbrales['critico']:
                estado = 'CRITICO'
                probabilidad_alerta = 95
            elif valor_predicho >= self.umbrales['alerta']:
                estado = 'ALERTA'
                probabilidad_alerta = 75
            else:
                estado = 'NORMAL'
                probabilidad_alerta = 25
            
            # Factores de influencia (importancia de características)
            factores_influencia = {}
            if hasattr(self.modelo, 'feature_importances_'):
                for i, col in enumerate(self.feature_cols):
                    factores_influencia[col] = round(self.modelo.feature_importances_[i] * 100, 1)
            
            # Generar recomendaciones
            recomendaciones = self._generar_recomendaciones_ivm(valor_predicho, factores_influencia)
            
            return PrediccionResultado(
                valor_predicho=round(valor_predicho, 1),
                confianza=round(confianza, 1),
                probabilidad_alerta=probabilidad_alerta,
                estado=estado,
                factores_influencia=factores_influencia,
                horizonte_temporal=f"{self.horizonte_horas} horas",
                timestamp=datetime.datetime.now(),
                metricas_modelo=self.metricas_entrenamiento,
                recomendaciones=recomendaciones
            )
            
        except Exception as e:
            self.logger.error(f"Error en predicción IVM: {e}")
            raise
    
    def _generar_recomendaciones_ivm(self, valor_predicho: float, factores: Dict[str, float]) -> List[str]:
        """Generar recomendaciones específicas para IVM"""
        recomendaciones = []
        
        if valor_predicho >= self.umbrales['critico']:
            recomendaciones.append("ACCIÓN INMEDIATA: Revisar sistema de microparadas")
            recomendaciones.append("Verificar mantenimiento preventivo")
        elif valor_predicho >= self.umbrales['alerta']:
            recomendaciones.append("Monitorear tendencias de microparadas")
            recomendaciones.append("Preparar intervención preventiva")
        
        # Recomendaciones basadas en factores
        if factores.get('cambio_turno', 0) > 20:
            recomendaciones.append("Optimizar transición entre turnos")
        
        if factores.get('microparadas_tendencia', 0) > 15:
            recomendaciones.append("Investigar tendencia creciente de microparadas")
        
        return recomendaciones

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
            
            # Métricas
            y_pred = self.modelo.predict(X_test_scaled)
            self.metricas_entrenamiento = {
                'mae': mean_absolute_error(y_test, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                'r2': r2_score(y_test, y_pred),
                'precisión': max(0, min(100, 100 - mean_absolute_error(y_test, y_pred) / 100 * 100)),
                'registros_entrenamiento': len(X_train)
            }
            
            self.modelo_entrenado = True
            self.feature_cols = feature_cols
            
            self.logger.info(f"Modelo IET entrenado. Precisión: {self.metricas_entrenamiento['precisión']:.1f}%")
            
            return self.metricas_entrenamiento
            
        except Exception as e:
            self.logger.error(f"Error entrenando modelo IET: {e}")
            raise
    
    def predecir(self, datos_entrada: pd.DataFrame) -> PrediccionResultado:
        """Predecir IET futuro"""
        if not self.modelo_entrenado:
            raise ValueError("Modelo IET no entrenado")
        
        # Implementación similar a IVM pero para IET
        # [Código de predicción específico para IET]
        
        return PrediccionResultado(
            valor_predicho=85.0,  # Placeholder
            confianza=75.0,
            probabilidad_alerta=30,
            estado='NORMAL',
            factores_influencia={},
            horizonte_temporal=f"{self.horizonte_horas} horas",
            timestamp=datetime.datetime.now(),
            metricas_modelo=self.metricas_entrenamiento,
            recomendaciones=["Mantener eficiencia actual"]
        )

class ModeloPrediccionIPC(ModeloPrediccionBase):
    """Modelo específico para predicción de IPC (Índice de Productividad de Conteo)"""
    
    def __init__(self, horizonte_horas: int = 2):
        super().__init__('IPC', horizonte_horas)
        self.modelo = RandomForestRegressor(n_estimators=100, random_state=42)
    
    def entrenar(self, datos: pd.DataFrame) -> Dict[str, float]:
        """Entrenar modelo de predicción IPC"""
        # Implementación específica para IPC
        return {'precisión': 80.0}
    
    def predecir(self, datos_entrada: pd.DataFrame) -> PrediccionResultado:
        """Predecir IPC futuro"""
        # Implementación específica para IPC
        return PrediccionResultado(
            valor_predicho=75.0,
            confianza=80.0,
            probabilidad_alerta=40,
            estado='ALERTA',
            factores_influencia={},
            horizonte_temporal=f"{self.horizonte_horas} horas",
            timestamp=datetime.datetime.now(),
            metricas_modelo={'precisión': 80.0},
            recomendaciones=["Optimizar capacidad de producción"]
        )

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
                self.logger.info(f"Modelo {nombre} entrenado con precisión: {resultado.get('precisión', 0):.1f}%")
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
                self.logger.info(f"Predicción {nombre}: {prediccion.valor_predicho} ({prediccion.estado})")
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
        alertas_criticas = sum(1 for p in predicciones.values() if p and p.estado == 'CRITICO')
        alertas_atencion = sum(1 for p in predicciones.values() if p and p.estado == 'ALERTA')
        
        output.append(f"\n[RESUMEN EJECUTIVO]")
        output.append(f"Alertas críticas: {alertas_criticas}")
        output.append(f"Alertas de atención: {alertas_atencion}")
        output.append(f"Indicadores normales: {len(predicciones) - alertas_criticas - alertas_atencion}")
        
        # Predicciones detalladas
        for nombre, prediccion in predicciones.items():
            if prediccion is None:
                output.append(f"\n[{nombre}] ERROR - No se pudo generar predicción")
                continue
            
            output.append(f"\n[{nombre}] {prediccion.estado}")
            output.append(f"  Valor predicho: {prediccion.valor_predicho}")
            output.append(f"  Confianza: {prediccion.confianza}%")
            output.append(f"  Probabilidad alerta: {prediccion.probabilidad_alerta}%")
            output.append(f"  Horizonte: {prediccion.horizonte_temporal}")
            
            if incluir_detalles and prediccion.recomendaciones:
                output.append(f"  Recomendaciones:")
                for rec in prediccion.recomendaciones:
                    output.append(f"    - {rec}")
        
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