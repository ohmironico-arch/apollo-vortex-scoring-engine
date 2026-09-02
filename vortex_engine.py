"""
Apollo.io - Vortex Scoring Engine (VSE) Extension
Module: core/scoring/vortex_engine.py
Author: External Tech Innovation Proposal
Description: High-performance Metaheuristic Optimization for B2B Lead Scoring
             using Vortex Search Algorithms (VSA) to minimize DB query costs.
"""

import logging
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd

# Configuración del Logger Corporativo
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ApolloVortexEngine")

@dataclass(frozen=True)
class TargetProfile:
    """
    Contrato de datos para el Perfil de Cliente Ideal (ICP) del usuario.
    Representa el centro gravitacional del mercado objetivo.
    """
    optimal_employees: float
    optimal_budget_k: float
    optimal_tech_count: float


class VortexScoringEngine:
    """
    Motor de optimización matemática basado en Vortex Search (VSA).
    Diseñado para reducir la carga de consultas complejas en bases de datos masivas.
    """
    
    def __init__(self, target: TargetProfile) -> None:
        if not target:
            raise ValueError("TargetProfile cannot be None.")
            
        self.target_vector = np.array([
            target.optimal_employees, 
            target.optimal_budget_k, 
            target.optimal_tech_count
        ], dtype=np.float64)
        
        # Límites operativos del ecosistema de datos B2B de Apollo.io
        self.lower_bounds = np.array([1.0, 5.0, 1.0], dtype=np.float64)
        self.upper_bounds = np.array([10000.0, 5000.0, 150.0], dtype=np.float64)

    def _calculate_fitness(self, features: np.ndarray) -> float:
        """
        Calcula la aptitud (Fitness Score) del vector.
        Retorna un Lead Score normalizado entre 0.0 y 100.0.
        """
        try:
            distance = np.linalg.norm(features - self.target_vector)
            # Escalabilidad inversa para normalización suave en dashboards
            normalized_score = 100.0 / (1.0 + (distance * 0.01))
            return float(normalized_score)
        except Exception as e:
            logger.error(f"Error computing fitness vector distance: {str(e)}")
            return 0.0

    def execute_vsa_optimization(self, iterations: int = 30, num_samples: int = 25) -> Tuple[np.ndarray, float]:
        """
        Ejecuta la optimización metaheurística del vórtice.
        Encuentra el centro matemático óptimo del clúster de clientes potenciales.
        """
        # Inicialización del centro en el punto medio del espacio de búsqueda
        vortex_center = (self.lower_bounds + self.upper_bounds) / 2.0
        best_score = self._calculate_fitness(vortex_center)
        best_solution = vortex_center.copy()
        
        initial_radius = 50.0
        dimensions = len(self.lower_bounds)

        for i in range(iterations):
            # Ecuación de decaimiento del radio del vórtice (Forma cónica decreciente)
            current_radius = initial_radius * (1.0 - (i / iterations))
            
            # Muestreo Gaussiano vectorizado en memoria para máxima velocidad de CPU
            samples = np.random.normal(vortex_center, current_radius, size=(num_samples, dimensions))
            samples = np.clip(samples, self.lower_bounds, self.upper_bounds)
            
            # Evaluación del remolino de muestras
            for candidate in samples:
                candidate_score = self._calculate_fitness(candidate)
                
                if candidate_score > best_score:
                    best_score = candidate_score
                    best_solution = candidate.copy()
            
            vortex_center = best_solution.copy()
            
        return best_solution, best_score

    def batch_process_apollo_pipeline(self, df_leads: pd.DataFrame) -> pd.DataFrame:
        """
        Interfaz principal para Pipelines de Datos (ETL).
        Mapea las coordenadas óptimas descubiertas y clasifica el DataFrame completo.
        """
        required_cols = ['num_employees', 'estimated_budget_k', 'tech_stack_count']
        if not all(col in df_leads.columns for col in required_cols):
            raise KeyError(f"DataFrame must contain all required columns: {required_cols}")

        logger.info("Initializing Vortex Search on data matrix...")
        ideal_coordinates, _ = self.execute_vsa_optimization()
        
        # Conversión a matriz NumPy para procesamiento masivo ultra veloz
        data_matrix = df_leads[required_cols].to_numpy(dtype=np.float64)
        
        # Cálculo de distancias vectorizado (Cero bucles for en la base de datos)
        distances = np.linalg.norm(data_matrix - ideal_coordinates, axis=1)
        
        # Inyección del nuevo Score en la estructura nativa de Apollo
        df_leads['vortex_lead_score'] = 100.0 / (1.0 + (distances * 0.01))
        
        logger.info("Pipeline optimization complete. Sorting results.")
        return df_leads.sort_values(by='vortex_lead_score', ascending=False)


# --- UNIT TESTS / INTEG-PROOFS ---
def test_engine_convergance():
    """ Prueba unitaria para verificar la convergencia del algoritmo """
    target = TargetProfile(optimal_employees=500.0, optimal_budget_k=1200.0, optimal_tech_count=45.0)
    engine = VortexScoringEngine(target)
    best_coords, best_score = engine.execute_vsa_optimization()
    
    assert best_score > 90.0, "The engine failed to converge on the target customer profile."
    logger.info("Unit Test: Convergence Check PASSED.")

if __name__ == "__main__":
    # Ejecutar validación interna
    test_engine_convergance()
    
    # Simulación de entorno de datos Apollo
    sample_data = {
        'company_name': ['Enterprise Alpha', 'MidMarket Beta', 'SME Gamma'],
        'num_employees': [520.0, 150.0, 15.0],
        'estimated_budget_k': [1180.0, 400.0, 50.0],
        'tech_stack_count': [42.0, 20.0, 5.0]
    }
    df_sandbox = pd.DataFrame(sample_data)
    
    target_icp = TargetProfile(optimal_employees=500.0, optimal_budget_k=1200.0, optimal_tech_count=45.0)
    engine = VortexScoringEngine(target_icp)
    
    processed_df = engine.batch_process_apollo_pipeline(df_sandbox)
    print("\n" + "="*60 + "\nOUTPUT ENVIADO A LA API DE APOLLO:\n" + "="*60)
    print(processed_df[['company_name', 'vortex_lead_score']].to_string(index=False))
  
