"""Model-independent face verification scoring and evaluation.

The module consumes externally produced embeddings and image-processing
outcomes. It does not implement a detector, aligner, or recognition model.
All reported verification metrics are conditional on pairs for which both
embeddings are valid; pair/image coverage is reported separately and gates
whether a result is suitable for a privacy-improvement claim.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
import numbers
from typing import Literal

import numpy as np
from sklearn.metrics import roc_auc_score

from src.evaluation.identity_pairs import PairManifest


ImageState = Literal["success", "no_face", "multiple_faces", "unknown"]
StageState = Literal["success", "failure", "not_attempted", "unknown"]


def _finite_fraction(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(f"{name} must be a finite number in [0, 1]")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be a finite number in [0, 1]")
    return result


@dataclass(frozen=True)
class EmbeddingStatus:
    image_id: str
    status: str
    reason: str | None = None
    dimension: int | None = None


@dataclass(frozen=True)
class ImageProcessingRecord:
    """Per-image process outcome supplied by a detector/model adapter.

    Detection, alignment, embedding and inference outcomes are separate so a
    failure at one stage is not mislabeled as a privacy success.
    """
    image_id: str
    attempted: bool
    detection_status: str = "unknown"  # success, no_face, multiple_faces, unknown
    alignment_status: str = "unknown"  # success, failure, not_attempted, unknown
    embedding_status: str = "unknown"  # success, invalid, missing, unknown
    inference_status: str = "unknown"  # success, error, not_attempted, unknown

    def __post_init__(self) -> None:
        if not isinstance(self.image_id, str) or not self.image_id.strip():
            raise ValueError("image_id must be a non-empty string")
        if not isinstance(self.attempted, bool):
            raise ValueError("attempted must be a boolean")
        if self.detection_status not in ("success", "no_face", "multiple_faces", "unknown"):
            raise ValueError("invalid detection_status")
        if self.alignment_status not in ("success", "failure", "not_attempted", "unknown"):
            raise ValueError("invalid alignment_status")
        if self.embedding_status not in ("success", "invalid", "missing", "unknown"):
            raise ValueError("invalid embedding_status")
        if self.inference_status not in ("success", "error", "not_attempted", "unknown"):
            raise ValueError("invalid inference_status")


@dataclass(frozen=True)
class PairScore:
    pair_id: str
    label: str
    similarity: float | None
    status: str
    reason: str | None = None


@dataclass(frozen=True)
class FailureAccounting:
    images_attempted: int
    successful_face_detections: int
    no_face_detections: int
    multiple_face_ambiguities: int
    alignment_failures: int
    invalid_embeddings: int
    inference_errors: int
    missing_pair_endpoints: int
    scoreable_genuine_pairs: int
    unscoreable_genuine_pairs: int
    scoreable_impostor_pairs: int
    unscoreable_impostor_pairs: int
    image_outcomes_complete: bool


@dataclass(frozen=True)
class PairScoringResult:
    manifest: PairManifest
    condition_identifier: str
    scores: tuple[PairScore, ...]
    embedding_statuses: tuple[EmbeddingStatus, ...]
    failure_accounting: FailureAccounting

    @property
    def pair_manifest_hash(self) -> str:
        return self.manifest.manifest_hash


@dataclass(frozen=True)
class VerificationMetrics:
    pair_manifest_hash: str
    dataset_identifier: str
    partition: str
    subset_role: str
    scenario: str
    condition_identifier: str
    metric_direction: str
    metric_population: str
    roc_auc: float | None
    roc_auc_status: str
    target_far: float | None
    threshold: float | None
    achieved_far: float | None
    achieved_tar: float | None
    genuine_scores: tuple[float, ...]
    impostor_scores: tuple[float, ...]
    pair_counts: Mapping[str, int]
    coverage: Mapping[str, float]
    failure_accounting: FailureAccounting
    suitable_for_privacy_claim: bool
    limitations: tuple[str, ...]


@dataclass(frozen=True)
class ThresholdCalibration:
    pair_manifest_hash: str
    parent_split_hash: str
    dataset_identifier: str
    partition: str
    scenario: str
    condition_identifier: str
    calibration_identity_ids: tuple[str, ...]
    target_far: float
    threshold: float | None
    achieved_far: float | None
    achieved_tar: float | None
    scoreable_genuine_pairs: int
    scoreable_impostor_pairs: int
    pair_coverage: float
    class_coverage: Mapping[str, float]
    status: str
    limitations: tuple[str, ...]


@dataclass(frozen=True)
class VerificationComparison:
    original_roc_auc: float | None
    transformed_roc_auc: float | None
    absolute_auc_change: float | None  # transformed minus original
    chance_adjusted_attack_advantage_reduction: float | None
    original_coverage: Mapping[str, float]
    transformed_coverage: Mapping[str, float]
    original_tar: float | None
    original_far: float | None
    transformed_tar: float | None
    transformed_far: float | None
    suitable_for_privacy_claim: bool
    limitations: tuple[str, ...]


@dataclass(frozen=True)
class ROCAUCBootstrapInterval:
    """Identity-dyadic bootstrap interval for conditional ROC-AUC.

    Bounds are omitted unless the point estimate is defined and the configured
    minimum number of bootstrap replicates produced both score classes.
    """
    point_estimate: float | None
    lower_bound: float | None
    upper_bound: float | None
    confidence_level: float
    bootstrap_replicates: int
    valid_replicates: int
    invalid_replicates: int
    minimum_valid_replicates: int
    method: str
    seed: int
    status: str
    limitations: tuple[str, ...]


def _as_vector(value: object, image_id: str) -> tuple[np.ndarray | None, EmbeddingStatus]:
    try:
        vector = np.asarray(value)
    except Exception:
        return None, EmbeddingStatus(image_id, "invalid", "embedding cannot be converted to an array")
    if vector.ndim != 1 or vector.size == 0:
        return None, EmbeddingStatus(image_id, "invalid", "embedding must be a non-empty one-dimensional vector")
    if vector.dtype.kind not in "iuf" or vector.dtype.kind == "b":
        return None, EmbeddingStatus(image_id, "invalid", "embedding must contain real numeric values")
    vector = vector.astype(np.float64, copy=False)
    if not np.all(np.isfinite(vector)):
        return None, EmbeddingStatus(image_id, "invalid", "embedding contains NaN or infinity", int(vector.size))
    norm = float(np.linalg.norm(vector))
    if not math.isfinite(norm) or norm <= 0:
        return None, EmbeddingStatus(image_id, "invalid", "embedding has zero or non-finite L2 norm", int(vector.size))
    return vector, EmbeddingStatus(image_id, "candidate", None, int(vector.size))


def normalize_embeddings(embeddings: Mapping[str, object]) -> tuple[dict[str, np.ndarray], tuple[EmbeddingStatus, ...]]:
    """Validate vectors and L2-normalize only if all valid vectors share a dimension.

    If dimensions disagree, every otherwise-valid vector is marked invalid;
    this avoids choosing a dimension based on mapping order or silently scoring
    only a convenient subset.
    """
    if not isinstance(embeddings, Mapping):
        raise TypeError("embeddings must map image identifiers to vectors")
    candidates: dict[str, np.ndarray] = {}
    statuses: dict[str, EmbeddingStatus] = {}
    for image_id in sorted(embeddings):
        if not isinstance(image_id, str) or not image_id.strip():
            raise ValueError("embedding image identifiers must be non-empty strings")
        vector, status = _as_vector(embeddings[image_id], image_id)
        statuses[image_id] = status
        if vector is not None:
            candidates[image_id] = vector
    dimensions = {int(vector.size) for vector in candidates.values()}
    if len(dimensions) > 1:
        for image_id, vector in candidates.items():
            statuses[image_id] = EmbeddingStatus(image_id, "invalid", "inconsistent embedding dimension", int(vector.size))
        candidates.clear()
    normalized: dict[str, np.ndarray] = {}
    for image_id, vector in candidates.items():
        normalized[image_id] = vector / np.linalg.norm(vector)
        statuses[image_id] = EmbeddingStatus(image_id, "valid", dimension=int(vector.size))
    return normalized, tuple(statuses[key] for key in sorted(statuses))


def score_manifest_pairs(manifest: PairManifest, embeddings: Mapping[str, object], *,
                         condition_identifier: str = "unspecified",
                         image_outcomes: Mapping[str, ImageProcessingRecord] | None = None) -> PairScoringResult:
    """Score every manifest pair, retaining explicit records for invalid/missing endpoints."""
    if not isinstance(manifest, PairManifest):
        raise TypeError("manifest must be a PairManifest")
    if not isinstance(condition_identifier, str) or not condition_identifier.strip():
        raise ValueError("condition_identifier must be a non-empty string")
    normalized, statuses = normalize_embeddings(embeddings)
    status_by_id = {status.image_id: status for status in statuses}
    referenced = {image for pair in manifest.pairs for image in (pair.first_image_id, pair.second_image_id)}
    missing = referenced - normalized.keys()
    all_statuses = list(statuses)
    for image_id in sorted(missing - status_by_id.keys()):
        all_statuses.append(EmbeddingStatus(image_id, "missing", "no embedding supplied"))
    results: list[PairScore] = []
    for pair in manifest.pairs:
        first = normalized.get(pair.first_image_id)
        second = normalized.get(pair.second_image_id)
        if first is None or second is None:
            bad = [pair.first_image_id if first is None else None, pair.second_image_id if second is None else None]
            reasons = [status_by_id.get(image_id, EmbeddingStatus(image_id or "", "missing", "no embedding supplied")).reason
                       for image_id in bad if image_id is not None]
            results.append(PairScore(pair.pair_id, pair.label, None, "unscoreable", "; ".join(r for r in reasons if r) or "invalid endpoint embedding"))
        else:
            similarity = float(np.clip(np.dot(first, second), -1.0, 1.0))
            results.append(PairScore(pair.pair_id, pair.label, similarity, "scoreable"))

    outcomes: dict[str, ImageProcessingRecord] = {}
    if image_outcomes is not None:
        if not isinstance(image_outcomes, Mapping):
            raise TypeError("image_outcomes must map image identifiers to ImageProcessingRecord values")
        for image_id, record in image_outcomes.items():
            if not isinstance(image_id, str) or not isinstance(record, ImageProcessingRecord) or image_id != record.image_id:
                raise ValueError("image_outcomes keys must match ImageProcessingRecord.image_id")
            outcomes[image_id] = record
    else:
        outcomes = {}
    labels = {"genuine": [0, 0], "impostor": [0, 0]}  # scoreable, unscoreable
    for result in results:
        index = 0 if result.status == "scoreable" else 1
        labels[result.label][index] += 1
    missing_endpoint_count = sum(
        int(pair.first_image_id not in embeddings) + int(pair.second_image_id not in embeddings)
        for pair in manifest.pairs
    )
    invalid_embedding_ids = {status.image_id for status in statuses if status.status == "invalid"}
    invalid_embedding_ids.update(image_id for image_id, record in outcomes.items()
                                 if record.embedding_status == "invalid")
    outcomes_complete = image_outcomes is not None and referenced.issubset(outcomes)
    if outcomes_complete:
        outcomes_complete = all(
            record.detection_status != "unknown" and record.alignment_status != "unknown"
            and record.embedding_status != "unknown" and record.inference_status != "unknown"
            for image_id, record in outcomes.items() if image_id in referenced
        )
    failure = FailureAccounting(
        images_attempted=sum(record.attempted for record in outcomes.values()),
        successful_face_detections=sum(record.detection_status == "success" for record in outcomes.values()),
        no_face_detections=sum(record.detection_status == "no_face" for record in outcomes.values()),
        multiple_face_ambiguities=sum(record.detection_status == "multiple_faces" for record in outcomes.values()),
        alignment_failures=sum(record.alignment_status == "failure" for record in outcomes.values()),
        invalid_embeddings=len(invalid_embedding_ids),
        inference_errors=sum(record.inference_status == "error" for record in outcomes.values()),
        missing_pair_endpoints=missing_endpoint_count,
        scoreable_genuine_pairs=labels["genuine"][0], unscoreable_genuine_pairs=labels["genuine"][1],
        scoreable_impostor_pairs=labels["impostor"][0], unscoreable_impostor_pairs=labels["impostor"][1],
        image_outcomes_complete=outcomes_complete,
    )
    return PairScoringResult(manifest, condition_identifier.strip(), tuple(results),
                             tuple(sorted(all_statuses, key=lambda s: s.image_id)), failure)


def _coverage(scoring: PairScoringResult) -> tuple[dict[str, int], dict[str, float], list[float], list[float]]:
    totals = {label: sum(pair.label == label for pair in scoring.scores) for label in ("genuine", "impostor")}
    valid = {label: sum(pair.label == label and pair.status == "scoreable" for pair in scoring.scores)
             for label in ("genuine", "impostor")}
    class_coverage = {label: (valid[label] / totals[label] if totals[label] else 0.0) for label in totals}
    all_total = sum(totals.values())
    all_valid = sum(valid.values())
    coverage = {"overall": all_valid / all_total if all_total else 0.0,
                "genuine": class_coverage["genuine"], "impostor": class_coverage["impostor"]}
    genuine_scores = [pair.similarity for pair in scoring.scores if pair.label == "genuine" and pair.similarity is not None]
    impostor_scores = [pair.similarity for pair in scoring.scores if pair.label == "impostor" and pair.similarity is not None]
    return {"genuine_total": totals["genuine"], "impostor_total": totals["impostor"],
            "genuine_scoreable": valid["genuine"], "impostor_scoreable": valid["impostor"]}, coverage, genuine_scores, impostor_scores


def _far_tar(impostor_scores: Sequence[float], genuine_scores: Sequence[float], threshold: float) -> tuple[float | None, float | None]:
    far = float(np.mean(np.asarray(impostor_scores) >= threshold)) if impostor_scores else None
    tar = float(np.mean(np.asarray(genuine_scores) >= threshold)) if genuine_scores else None
    return far, tar


def calibrate_verification_threshold(calibration: PairScoringResult, *, target_far: float,
                                     minimum_pair_coverage: float) -> ThresholdCalibration:
    """Choose the lowest empirical threshold whose calibration FAR <= target.

    Similarity decisions use ``score >= threshold``. Tied impostor scores stay
    together; threshold candidates are observed impostor values plus the next
    representable float above the maximum. This gives deterministic tie handling.
    """
    if calibration.manifest.subset_role != "calibration":
        raise ValueError("threshold calibration requires a calibration pair manifest")
    target_far = _finite_fraction(target_far, "target_far")
    minimum_pair_coverage = _finite_fraction(minimum_pair_coverage, "minimum_pair_coverage")
    counts, coverage, genuine, impostor = _coverage(calibration)
    limitations: list[str] = []
    if coverage["overall"] < minimum_pair_coverage or min(coverage["genuine"], coverage["impostor"]) < minimum_pair_coverage:
        limitations.append("calibration pair coverage is below the configured minimum")
    if not genuine or not impostor:
        limitations.append("calibration requires at least one scoreable genuine and impostor pair")
    threshold = far = tar = None
    status = "valid"
    if limitations:
        status = "invalid_calibration"
    else:
        max_imp = max(impostor)
        candidates = sorted(set(float(value) for value in impostor))
        candidates.append(float(np.nextafter(max_imp, math.inf)))
        feasible = [(candidate, *_far_tar(impostor, genuine, candidate)) for candidate in candidates]
        feasible = [item for item in feasible if item[1] is not None and item[1] <= target_far]
        if not feasible:
            status = "invalid_calibration"
            limitations.append("no empirical threshold satisfies the configured FAR")
        else:
            threshold, far, tar = min(feasible, key=lambda item: item[0])
    manifest = calibration.manifest
    return ThresholdCalibration(manifest.manifest_hash, manifest.split_hash, manifest.dataset_identifier,
                                manifest.partition, manifest.scenario, calibration.condition_identifier,
                                manifest.identity_ids, target_far,
                                threshold, far, tar, len(genuine), len(impostor), coverage["overall"],
                                {"genuine": coverage["genuine"], "impostor": coverage["impostor"]},
                                status, tuple(limitations))


def evaluate_verification(scoring: PairScoringResult, calibration: ThresholdCalibration, *,
                          minimum_pair_coverage: float) -> VerificationMetrics:
    """Apply a frozen calibration threshold and compute conditional scoring metrics."""
    if scoring.manifest.subset_role != "scoring":
        raise ValueError("verification scoring requires a scoring pair manifest")
    minimum_pair_coverage = _finite_fraction(minimum_pair_coverage, "minimum_pair_coverage")
    manifest = scoring.manifest
    if calibration.status != "valid" or calibration.threshold is None:
        raise ValueError("cannot evaluate scoring data without a valid frozen calibration threshold")
    if (calibration.parent_split_hash != manifest.split_hash or calibration.dataset_identifier != manifest.dataset_identifier
            or calibration.partition != manifest.partition or calibration.scenario != manifest.scenario):
        raise ValueError("calibration and scoring manifests do not belong to the same evaluation protocol")
    if calibration.condition_identifier != scoring.condition_identifier:
        raise ValueError("calibration threshold condition does not match scoring condition")
    if set(calibration.calibration_identity_ids) & set(manifest.identity_ids):
        raise ValueError("calibration and scoring identities overlap")
    if calibration.pair_manifest_hash == manifest.manifest_hash:
        raise ValueError("calibration and scoring must use separate pair manifests")
    counts, coverage, genuine, impostor = _coverage(scoring)
    labels = [1] * len(genuine) + [0] * len(impostor)
    scores = genuine + impostor
    auc: float | None = None
    auc_status = "valid"
    if not genuine or not impostor:
        auc_status = "undefined_missing_scoreable_class"
    else:
        auc = float(roc_auc_score(labels, scores))
    far, tar = _far_tar(impostor, genuine, calibration.threshold)
    limitations = list(calibration.limitations)
    suitable = True
    if auc is None:
        suitable = False
        limitations.append("ROC-AUC is undefined because a scoreable class is absent")
    if coverage["overall"] < minimum_pair_coverage or min(coverage["genuine"], coverage["impostor"]) < minimum_pair_coverage:
        suitable = False
        limitations.append("scoring pair coverage is below the configured minimum")
    if not scoring.failure_accounting.image_outcomes_complete:
        limitations.append("complete per-image detection/alignment/inference outcomes were not supplied")
    return VerificationMetrics(
        manifest.manifest_hash, manifest.dataset_identifier, manifest.partition, manifest.subset_role,
        manifest.scenario, scoring.condition_identifier, "higher_is_better", "successfully_scored_pairs",
        auc, auc_status, calibration.target_far,
        calibration.threshold, far, tar, tuple(genuine), tuple(impostor), counts, coverage,
        scoring.failure_accounting, suitable, tuple(dict.fromkeys(limitations)),
    )


def bootstrap_roc_auc_by_identity(scoring: PairScoringResult, *, confidence_level: float = 0.95,
                                  bootstrap_replicates: int = 1000, seed: int = 0,
                                  minimum_valid_fraction: float = 0.8) -> ROCAUCBootstrapInterval:
    """Estimate a percentile ROC-AUC interval with a dyadic identity bootstrap.

    The resampling unit is the identity vertex, not the pair. One bootstrap
    sample draws the manifest's scoring identities with replacement and gives
    each identity a multiplicity ``m_i``. An observed genuine pair (same
    identity at both endpoints) receives weight ``m_i``. An observed impostor
    pair ``(i, j)`` receives weight ``m_i * m_j``. Thus an identity used in
    many pairs affects every incident pair, including both endpoint roles.
    Pairs and labels are never synthesized or relabeled. The interval estimates
    ROC-AUC conditional on pairs that were successfully scored; coverage and
    detector failures remain separate evaluator outcomes.

    This is a dyadic/vertex bootstrap, not an independent pair bootstrap and
    not the conventional one-way cluster bootstrap that assigns one cluster
    weight to a pair based only on one endpoint. The shared vertex weights
    preserve dependence for pairs sharing either endpoint. Same-identity
    genuine pairs are one identity cluster and therefore receive one, not
    squared, multiplicity.

    A replicate without positive-weight observations from both classes is
    invalid. If fewer than ``max(2, ceil(replicates * minimum_valid_fraction))``
    replicates are valid, bounds are returned as ``None`` with an explicit
    ``insufficient_valid_replicates`` status. Quantiles use NumPy's linear
    percentile convention. This procedure does not correct for identity
    sampling bias, pair-manifest construction, model training overlap, or
    incomplete score coverage; it is not a real-world privacy guarantee.
    """
    if not isinstance(scoring, PairScoringResult):
        raise TypeError("scoring must be a PairScoringResult")
    confidence_level = _finite_fraction(confidence_level, "confidence_level")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be strictly between 0 and 1")
    minimum_valid_fraction = _finite_fraction(minimum_valid_fraction, "minimum_valid_fraction")
    if isinstance(bootstrap_replicates, bool) or not isinstance(bootstrap_replicates, int) or bootstrap_replicates < 1:
        raise ValueError("bootstrap_replicates must be an integer >= 1")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be an integer >= 0")

    method = "identity_dyadic_vertex_bootstrap_percentile_v1"
    minimum_valid_replicates = max(2, math.ceil(bootstrap_replicates * minimum_valid_fraction))
    manifest_pairs = {pair.pair_id: pair for pair in scoring.manifest.pairs}
    score_by_id = {score.pair_id: score for score in scoring.scores}
    if len(score_by_id) != len(scoring.scores) or set(score_by_id) != set(manifest_pairs):
        raise ValueError("scoring results must contain exactly one score for every manifest pair")
    for pair_id, score in score_by_id.items():
        pair = manifest_pairs[pair_id]
        if score.label != pair.label:
            raise ValueError("pair score label does not match its manifest record")
        if score.status == "scoreable":
            if (score.similarity is None or not isinstance(score.similarity, numbers.Real)
                    or not math.isfinite(float(score.similarity))
                    or not -1.0 <= float(score.similarity) <= 1.0):
                raise ValueError("scoreable pair scores must have finite cosine similarity in [-1, 1]")
        elif score.status != "unscoreable":
            raise ValueError("pair score status must be scoreable or unscoreable")

    usable = [
        (manifest_pairs[score.pair_id], score)
        for score in scoring.scores
        if score.status == "scoreable" and score.similarity is not None
    ]
    labels = np.asarray([int(score.label == "genuine") for _, score in usable], dtype=np.int8)
    values = np.asarray([float(score.similarity) for _, score in usable], dtype=np.float64)
    if not usable or set(labels.tolist()) != {0, 1} or not np.all(np.isfinite(values)):
        reason = "ROC-AUC point estimate is undefined without finite scoreable genuine and impostor pairs"
        return ROCAUCBootstrapInterval(None, None, None, confidence_level, bootstrap_replicates,
                                       0, bootstrap_replicates, minimum_valid_replicates, method, seed,
                                       "invalid_point_estimate", (reason,))

    point_estimate = float(roc_auc_score(labels, values))
    identities = tuple(scoring.manifest.identity_ids)
    if not identities:
        return ROCAUCBootstrapInterval(point_estimate, None, None, confidence_level,
                                       bootstrap_replicates, 0, bootstrap_replicates,
                                       minimum_valid_replicates, method, seed,
                                       "insufficient_valid_replicates",
                                       ("manifest contains no scoring identities for resampling",))

    identity_index = {identity: index for index, identity in enumerate(identities)}
    endpoint_indices: list[tuple[int, int]] = []
    for pair, _ in usable:
        try:
            endpoint_indices.append((identity_index[pair.first_identity_id],
                                     identity_index[pair.second_identity_id]))
        except KeyError as exc:
            raise ValueError("scored pair identity is absent from the manifest identity set") from exc

    rng = np.random.default_rng(seed)
    estimates: list[float] = []
    for _ in range(bootstrap_replicates):
        sampled = rng.integers(0, len(identities), size=len(identities))
        multiplicities = np.bincount(sampled, minlength=len(identities))
        weights = np.asarray([
            multiplicities[first] if first == second else multiplicities[first] * multiplicities[second]
            for first, second in endpoint_indices
        ], dtype=np.float64)
        positive = weights > 0
        if not np.any(positive & (labels == 0)) or not np.any(positive & (labels == 1)):
            continue
        estimates.append(float(roc_auc_score(labels[positive], values[positive],
                                             sample_weight=weights[positive])))

    valid_replicates = len(estimates)
    invalid_replicates = bootstrap_replicates - valid_replicates
    if valid_replicates < minimum_valid_replicates:
        return ROCAUCBootstrapInterval(
            point_estimate, None, None, confidence_level, bootstrap_replicates,
            valid_replicates, invalid_replicates, minimum_valid_replicates, method, seed,
            "insufficient_valid_replicates",
            (f"valid bootstrap replicates ({valid_replicates}) are below the required "
             f"minimum ({minimum_valid_replicates})",),
        )
    tail = (1.0 - confidence_level) / 2.0
    lower, upper = np.quantile(np.asarray(estimates, dtype=np.float64), [tail, 1.0 - tail], method="linear")
    return ROCAUCBootstrapInterval(point_estimate, float(lower), float(upper), confidence_level,
                                   bootstrap_replicates, valid_replicates, invalid_replicates,
                                   minimum_valid_replicates, method, seed, "valid", ())


def compare_verification_metrics(original: VerificationMetrics, transformed: VerificationMetrics) -> VerificationComparison:
    """Compare matched score manifests without applying relative reduction to raw AUC."""
    if original.pair_manifest_hash != transformed.pair_manifest_hash:
        raise ValueError("original and transformed results must use the same scoring pair manifest")
    if original.target_far != transformed.target_far:
        raise ValueError("original and transformed results must use the same target FAR")
    limitations: list[str] = []
    if original.roc_auc is None or transformed.roc_auc is None:
        auc_change = None
        limitations.append("AUC change is undefined because one or both ROC-AUC values are unavailable")
    else:
        auc_change = transformed.roc_auc - original.roc_auc
    advantage_reduction = None
    baseline_valid = (original.suitable_for_privacy_claim and original.roc_auc_status == "valid"
                      and original.roc_auc is not None
                      and original.roc_auc > 0.5 and transformed.suitable_for_privacy_claim
                      and transformed.roc_auc_status == "valid" and transformed.roc_auc is not None)
    if baseline_valid:
        # Reduce excess attack advantage above chance; this is not a generic
        # relative reduction on raw AUC and may be negative if attack improves.
        advantage_reduction = ((original.roc_auc - 0.5) - (transformed.roc_auc - 0.5)) / (original.roc_auc - 0.5)
    else:
        limitations.append("original attack baseline is invalid, unsuitable, or not above chance")
    if not transformed.suitable_for_privacy_claim:
        limitations.append("transformed result does not meet configured coverage/metric validity requirements")
    suitable = baseline_valid and transformed.suitable_for_privacy_claim
    if not original.failure_accounting.image_outcomes_complete or not transformed.failure_accounting.image_outcomes_complete:
        limitations.append("image-level failure accounting is incomplete; detection failure is not privacy evidence")
    return VerificationComparison(original.roc_auc, transformed.roc_auc, auc_change, advantage_reduction,
                                  original.coverage, transformed.coverage, original.achieved_tar,
                                  original.achieved_far, transformed.achieved_tar, transformed.achieved_far,
                                  suitable, tuple(dict.fromkeys(limitations)))
