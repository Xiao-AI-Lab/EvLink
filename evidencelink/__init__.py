"""EvidenceLink: source-grounded evidence links and coverage-aware selection."""

__version__ = "0.1.0"

from evidencelink.api import (
    PaperPipelineArtifacts,
    PaperPipelineConfig,
    build_candidate_pool_cq,
    build_evidence_needs_bq,
    build_openie_artifact,
    build_support_cache,
    compose_final_evidence_rq,
    run_paper_pipeline,
)
from evidencelink.artifacts import CandidateEvidence, CandidatePoolRecord, Document, OpenIEFact, Question
from evidencelink.datasets import (
    DATASET_ALIASES,
    SUPPORTED_DATASETS,
    DatasetSpec,
    canonical_dataset_name,
    dataset_spec,
    iter_supported_datasets,
    supported_dataset_names,
)
from evidencelink.evidence_needs import mine_evidence_needs_for_question
from evidencelink.index import EvidenceLink, EvidenceLinkIndex, build_evidence_link_index
from evidencelink.induction import build_candidate_pool_records
from evidencelink.integrations import candidates_from_hipporag, candidates_from_lightrag
from evidencelink.prepare_dataset import prepare_benchmark_dataset
from evidencelink.run_evidence_selection import run_evidence_selection
from evidencelink.selector import EvidenceSelection, EvidenceSelector, EvidenceSelectorConfig, select_evidence

__all__ = [
    "CandidateEvidence",
    "CandidatePoolRecord",
    "DATASET_ALIASES",
    "DatasetSpec",
    "Document",
    "EvidenceLink",
    "EvidenceLinkIndex",
    "EvidenceSelection",
    "EvidenceSelector",
    "EvidenceSelectorConfig",
    "OpenIEFact",
    "PaperPipelineArtifacts",
    "PaperPipelineConfig",
    "Question",
    "SUPPORTED_DATASETS",
    "build_candidate_pool_cq",
    "build_candidate_pool_records",
    "build_evidence_needs_bq",
    "build_evidence_link_index",
    "build_openie_artifact",
    "build_support_cache",
    "canonical_dataset_name",
    "candidates_from_hipporag",
    "candidates_from_lightrag",
    "compose_final_evidence_rq",
    "dataset_spec",
    "iter_supported_datasets",
    "mine_evidence_needs_for_question",
    "prepare_benchmark_dataset",
    "run_paper_pipeline",
    "run_evidence_selection",
    "select_evidence",
    "supported_dataset_names",
]
