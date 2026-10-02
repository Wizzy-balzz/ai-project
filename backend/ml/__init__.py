# backend.ml package
from backend.ml.dataset import (
    AttackGraphDataset,
    DataValidationReport,
    build_dataset,
    build_node_labels,
    build_edge_features,
    topology_to_pyg,
    validate_pyg_data,
    EDGE_TYPE_CATEGORIES,
    EDGE_SCALAR_FEATURE_NAMES,
    EDGE_FEATURE_DIM,
    ALL_EDGE_FEATURE_NAMES,
)
from backend.ml.model import AttackGNN
from backend.ml.trainer import (
    TrainingConfig,
    TrainingHistory,
    Trainer,
    compute_class_weights,
    set_seeds,
    load_checkpoint,
)
from backend.ml.evaluator import (
    EvaluationResult,
    Evaluator,
    collect_predictions,
)
from backend.ml.path_models import (
    CandidateAttackPath,
    PathConfig,
    PathGenerationSummary,
    PathValidationResult,
    make_path_id,
)
from backend.ml.path_generator import (
    GNNProbabilityProvider,
    PathGenerator,
    PathValidator,
)

__all__ = [
    # Dataset
    "AttackGraphDataset",
    "DataValidationReport",
    "build_dataset",
    "build_node_labels",
    "build_edge_features",
    "topology_to_pyg",
    "validate_pyg_data",
    "EDGE_TYPE_CATEGORIES",
    "EDGE_SCALAR_FEATURE_NAMES",
    "EDGE_FEATURE_DIM",
    "ALL_EDGE_FEATURE_NAMES",
    # Model
    "AttackGNN",
    # Trainer
    "TrainingConfig",
    "TrainingHistory",
    "Trainer",
    "compute_class_weights",
    "set_seeds",
    "load_checkpoint",
    # Evaluator
    "EvaluationResult",
    "Evaluator",
    "collect_predictions",
    # Phase 4 Path Generation
    "PathConfig",
    "CandidateAttackPath",
    "PathValidationResult",
    "PathGenerationSummary",
    "make_path_id",
    "GNNProbabilityProvider",
    "PathGenerator",
    "PathValidator",
]
