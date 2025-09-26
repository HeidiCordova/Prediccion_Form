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
from formulas.calculador_industrial import CalculadorFormulasIndustrialesArreglado
from data.csv_parser_avanzado import CSVParserAvanzado
def main():
    """Programa principal con parser mejorado"""
    
    ruta_archivo = "datos.csv"
    
    
    try:
        print("\n" + "=" * 70)
        print("SISTEMA MEJORADO - PROCESAMIENTO CSV INDUSTRIAL")
        print("=" * 70)
        
        # 1. Parser avanzado
        print("1. Analizando formato del archivo CSV...")
        parser = CSVParserAvanzado(ruta_archivo)
        # Cargar datos
        datos = parser.cargar_csv_formato_industrial()
        print(f"   ✓ CSV cargado exitosamente: {len(datos)} filas, {len(datos.columns)} columnas")
        
        # 2. Mostrar resumen
        print("\n2. Resumen de datos cargados:")
        print(parser.mostrar_resumen_datos())
        
        # 3. Calculador mejorado
        print("\n3. Calculando fórmulas con mapeo inteligente...")
        calculador = CalculadorFormulasIndustrialesArreglado(datos)
        resultados = calculador.calcular_todas_las_formulas_reales()
        
        # 4. Mostrar resultados
        print("\n" + "=" * 60)
        print("RESULTADOS DE FÓRMULAS INDUSTRIALES")
        print("=" * 60)
        
        for formula, resultado in resultados.items():
            if formula in ['IVM', 'IET']:
                print(f"\n{formula}:")
                if 'error' in resultado:
                    print(f"  ERROR: {resultado['error']}")
                else:
                    print(f"  Valor: {resultado['valor']}%")
                    print(f"  Interpretación: {resultado['interpretacion']}")
                    if 'componentes' in resultado:
                        comp = resultado['componentes']
                        if 'registros_validos' in comp:
                            print(f"  Registros válidos: {comp['registros_validos']:,}")
        
        # 5. Mapeo de columnas usado
        print(f"\nMAPEO DE COLUMNAS USADO:")
        print("-" * 30)
        for clave, columna in resultados['mapeo_columnas'].items():
            print(f"  {clave}: {columna}")
        
        print(f"\nProcesamiento completado exitosamente!")
        
    except Exception as e:
        print(f"\nERROR CRÍTICO: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()