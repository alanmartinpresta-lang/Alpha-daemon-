"""AIRE V201.1 Alpha World Core.

Only runtime world layers are exposed here. Validation campaigns, NASA/CFD,
historical engines, benchmarks and test suites are intentionally external.
"""
from .core import WorldCore, WorldState, WorldIdentity, Space3D, SimulationClock
from .environment import PlanetConfig, EnvironmentState, EnvironmentWorld, create_environment, step_environment
from .u2_consolidated import U2DynamicsState, U2ConsolidatedReport, initialize_u2_dynamics, step_u2_dynamics
from .chemistry import ChemicalSpecies, ChemicalReaction, ChemistryNetwork, ChemistryState, create_chemistry, step_chemistry
from .biology import BiologicalParameters, BiologicalCompartment, BiologicalState, create_biology, step_biology
from .ecology import EcologicalResource, EcologicalPopulation, EcologyState, create_ecology, step_ecology
from .u5_consolidated import BiogeochemicalPools, EcologicalRelation, EvolutionTrait, U5ConsolidatedState, create_u5_consolidated, step_u5_consolidated
from .organism import Anatomy, OrganSystems, SensorState, FunctionalState, MotorCommand, Genome, OrganismState, create_organism, create_offspring, mutate_genome, sense_environment, step_organism, attach_organism
from .interdomain_metabolism import RespirationContract, step_coupled_respiration
from .interdomain_uptake import step_nutrient_mineral_uptake
from .physical_coupling import transfer_environment_to_biology
from .world_closure_v192 import MatterClosureState, EnergyClosureState
from .v201_atmospheric_chemistry import step_atmospheric_chemistry

from .agent import AutonomyConfig, AutonomyState, Experience, ACTION_NAMES, sensor_vector, state_key, homeostatic_score, homeostatic_reward, action_command, choose_action, learn, autonomy_digest
