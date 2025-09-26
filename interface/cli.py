import os
import sys
import time
import datetime
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')
from core.config import ConfiguracionSuperusuario
from formulas.production_formulas import FormulasProduccion
from formulas.temporal_targets import ObjetivosTemporales
from models.prediction_models import ModeloPrediccion
class InterfazCLI:
    """Interfaz de línea de comandos sin emojis"""
    
    def __init__(self):
        self.config = ConfiguracionSuperusuario()
        self.formulas = FormulasProduccion()
        self.objetivos = ObjetivosTemporales()
        self.modelo = ModeloPrediccion()
        self.timestamp_inicio = None
        self.timestamp_fin = None
    
    def ejecutar_sistema_completo(self):
        """Ejecutar análisis completo del sistema"""
        self.timestamp_inicio = datetime.datetime.now()
        
        # Generar datos simulados para demostración
        datos_microparadas = [23.5, 28.2, 19.8, 32.1, 25.7, 29.3, 21.4, 26.8]
        datos_historicos = list(range(50, 90, 2)) + list(range(90, 50, -1))
        
        self._mostrar_cabecera()
        self._mostrar_indicadores_prioritarios(datos_microparadas)
        self._mostrar_predicciones_temporales(datos_historicos)
        self._mostrar_componentes_formulas(datos_microparadas)
        self._mostrar_finalizacion()
    
    def _mostrar_cabecera(self):
        """Mostrar cabecera con timestamp obligatorio"""
        print("=" * 80)
        print("SISTEMA PREDICTIVO INDUSTRIAL - RESULTADOS COMPLETOS")
        print(f"Iniciado: {self.timestamp_inicio.strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)
    
    def _mostrar_indicadores_prioritarios(self, datos_microparadas: List[float]):
        """Mostrar indicadores con mayor prioridad según cuestionario"""
        print("\n[INDICADORES PRIORITARIOS]")
        print("-" * 50)
        
        # IVM (7 menciones - prioridad máxima)
        ivm = self.formulas.calcular_ivm(datos_microparadas)
        estado_ivm = self._determinar_estado_alerta(ivm['valor'])
        
        print(f"IVM (Variabilidad Microparadas): {ivm['valor']}% - {estado_ivm}")
        print(f"    Formula: (Desviacion_Std / Media) x 100")
        print(f"    Componentes: Std={ivm['componentes']['sigma_desviacion']}, Media={ivm['componentes']['mu_media']}")
        print(f"    Interpretación: {ivm['interpretacion']}")
        
        # IET (6 menciones)  
        iet = self.formulas.calcular_iet(480, 85)  # 8 horas, 85 min microparadas
        estado_iet = self._determinar_estado_alerta(100 - iet['valor'])
        
        print(f"\nIET (Eficiencia Temporal): {iet['valor']}% - {estado_iet}")
        print(f"    Formula: ((T.Disponible - T.Microparadas) / T.Disponible) x 100")
        print(f"    Componentes: Disponible={iet['componentes']['t_disponible']}, Microparadas={iet['componentes']['t_microparadas']}")
        print(f"    Interpretación: {iet['interpretacion']}")
        
        # IIPNP (6 menciones)
        iipnp = self.formulas.calcular_iipnp(45, 480)
        estado_iipnp = self._determinar_estado_alerta(iipnp['valor'])
        
        print(f"\nIIPNP (Paradas No Programadas): {iipnp['valor']}% - {estado_iipnp}")
        print(f"    Formula: (T.PNoProgramada / T.Disponible) x 100")
        print(f"    Componentes: PNP={iipnp['componentes']['t_parada_no_programada']}, Disponible={iipnp['componentes']['t_disponible']}")
        print(f"    Interpretación: {iipnp['interpretacion']}")
    
    def _mostrar_predicciones_temporales(self, datos_historicos: List[float]):
        """Mostrar predicciones por horizonte temporal del cuestionario"""
        print("\n[PREDICCIONES TEMPORALES - HORIZONTES CUESTIONARIO]")
        print("-" * 50)
        
        # Crear targets para cada horizonte
        target_30min = self.objetivos.crear_targets_30min(datos_historicos)
        target_2h = self.objetivos.crear_targets_2h(datos_historicos)
        target_24h = self.objetivos.crear_targets_24h(datos_historicos)
        
        targets = [target_30min, target_2h, target_24h]
        
        for target in targets:
            print(f"Horizonte {target['horizonte']}: {target['probabilidad']}% probabilidad microparada")
            print(f"    Base usuarios: {target['menciones_usuarios']} menciones ({target['porcentaje_usuarios']}% usuarios)")
            print(f"    Acción: {target['accion_recomendada']}")
            print()
        
        # Mostrar balance de horizontes
        balance = self.objetivos.validar_balance_horizontes(targets)
        print(f"Horizonte prioritario: {balance['horizonte_prioritario']} ({balance['total_menciones']} menciones totales)")
    
    def _mostrar_componentes_formulas(self, datos_microparadas: List[float]):
        """Mostrar componentes detallados de fórmulas"""
        print("\n[COMPONENTES DE FORMULAS]")
        print("-" * 50)
        
        # IVM detallado
        ivm = self.formulas.calcular_ivm(datos_microparadas)
        print("IVM - Índice Variabilidad Microparadas:")
        print(f"    Formula: (σ/μ) × 100%")
        print(f"    σ (Desviación): {ivm['componentes']['sigma_desviacion']} minutos")
        print(f"    μ (Media): {ivm['componentes']['mu_media']} minutos")
        print(f"    Resultado: {ivm['interpretacion']}")
        
        # IET detallado
        print(f"\nIET - Índice Eficiencia Temporal:")
        iet = self.formulas.calcular_iet(480, 85)
        print(f"    Formula: ((T.Disponible - T.Microparadas) / T.Disponible) × 100%")
        print(f"    T.Disponible: {iet['componentes']['t_disponible']} minutos")
        print(f"    T.Microparadas: {iet['componentes']['t_microparadas']} minutos")
        print(f"    T.Efectivo: {iet['componentes']['t_efectivo']} minutos")
        print(f"    Resultado: {iet['interpretacion']}")
        
        # IPC ejemplo
        print(f"\nIPC - Índice Productividad Conteo:")
        ipc = self.formulas.calcular_ipc(85.5, 100)
        print(f"    Formula: (Promedio.Conteo / Máximo.Conteo) × 100%")
        print(f"    Promedio.Conteo: {ipc['componentes']['conteo_promedio']}")
        print(f"    Máximo.Conteo: {ipc['componentes']['conteo_maximo']}")
        print(f"    Resultado: {ipc['interpretacion']}")
    
    def _mostrar_finalizacion(self):
        """Mostrar información de finalización obligatoria"""
        self.timestamp_fin = datetime.datetime.now()
        duracion = (self.timestamp_fin - self.timestamp_inicio).total_seconds()
        
        print(f"\n[FINALIZACION]: {self.timestamp_fin.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"[DURACION]: {duracion:.2f} segundos")
    
    def _determinar_estado_alerta(self, valor: float) -> str:
        """Determinar estado de alerta basado en umbrales"""
        if valor <= self.config.umbrales['verde_max']:
            return "NORMAL"
        elif valor <= self.config.umbrales['amarilla_max']:
            return "ATENCION_REQUERIDA"
        else:
            return "ALERTA_CRITICA"
