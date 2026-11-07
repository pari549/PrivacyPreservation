"""Identity-disjoint split and bounded face-verification pair protocols.

This module is dataset- and model-independent. Serialized manifests contain
identity and image identifiers, which may be sensitive derived dataset data;
keep them internal unless the dataset terms explicitly permit publication.
Hashes here provide reproducible integrity checks, not authentication.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
import numbers
import random
from types import MappingProxyType


PARTITIONS = ("development", "optimization", "validation", "final_test")
SCENARIOS = ("scenario_a_both_transformed", "scenario_b_original_reference_transformed_probe")
SAMPLING_STRATEGY = "python_random_round_robin_v1"


def _identifier(value: object, name: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"{name} must be a non-empty string without surrounding whitespace")
    return value


def _integer(value: object, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _canonical_json(value: object) -> str:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"value is not canonical JSON data: {exc}") from exc


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def validate_identity_inventory(identity_to_images: Mapping[str, Sequence[str]]) -> dict[str, tuple[str, ...]]:
    """Validate and normalize an identity -> image-identifier mapping.

    Image identifiers must be globally unique. Input order is discarded so
    equivalent mappings produce identical inventory and split hashes.
    """
    if not isinstance(identity_to_images, Mapping) or not identity_to_images:
        raise ValueError("identity_to_images must be a non-empty mapping")
    normalized: dict[str, tuple[str, ...]] = {}
    seen_images: set[str] = set()
    for raw_identity, raw_images in identity_to_images.items():
        identity = _identifier(raw_identity, "identity identifier")
        if isinstance(raw_images, (str, bytes)) or not isinstance(raw_images, Sequence) or not raw_images:
            raise ValueError(f"identity {identity!r} must map to a non-empty sequence of image identifiers")
        images = tuple(_identifier(image, "image identifier") for image in raw_images)
        for image in images:
            if image in seen_images:
                raise ValueError(f"duplicate image identifier: {image!r}")
            seen_images.add(image)
        normalized[identity] = tuple(sorted(images))
    return dict(sorted(normalized.items()))


def inventory_sha256(identity_to_images: Mapping[str, Sequence[str]]) -> str:
    """Hash a canonical inventory of identifiers, not image contents."""
    inventory = validate_identity_inventory(identity_to_images)
    return _sha256({identity: list(images) for identity, images in inventory.items()})


def _stable_rank(seed: int, namespace: str, identifier: str) -> tuple[str, str]:
    digest = hashlib.sha256(f"{seed}\0{namespace}\0{identifier}".encode("utf-8")).hexdigest()
    return digest, identifier


def _allocate(keys: tuple[str, ...], size: int, *, proportions: Mapping[str, float] | None = None,
              counts: Mapping[str, int] | None = None) -> dict[str, int]:
    if (proportions is None) == (counts is None):
        raise ValueError("provide exactly one of proportions or counts")
    selected = proportions if proportions is not None else counts
    assert selected is not None
    if not isinstance(selected, Mapping) or set(selected) != set(keys):
        raise ValueError(f"allocation must specify exactly: {', '.join(keys)}")
    if counts is not None:
        allocation = {key: _integer(counts[key], f"count for {key}") for key in keys}
        if sum(allocation.values()) != size:
            raise ValueError("explicit counts must sum to the number of identities")
        return allocation
    values: dict[str, float] = {}
    for key in keys:
        value = proportions[key]  # type: ignore[index]
        if isinstance(value, bool) or not isinstance(value, numbers.Real) or not math.isfinite(float(value)) or value < 0:
            raise ValueError(f"proportion for {key} must be finite and non-negative")
        values[key] = float(value)
    if not math.isclose(sum(values.values()), 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("proportions must sum to 1")
    raw = {key: values[key] * size for key in keys}
    allocation = {key: math.floor(raw[key]) for key in keys}
    remainder = size - sum(allocation.values())
    order = sorted(keys, key=lambda key: (-(raw[key] - allocation[key]), keys.index(key)))
    for key in order[:remainder]:
        allocation[key] += 1
    return allocation


@dataclass(frozen=True)
class IdentitySplitResult:
    dataset_identifier: str
    dataset_release: str
    protocol_version: str
    seed: int
    allocation_kind: str
    requested_allocation: Mapping[str, int | float]
    partitions: Mapping[str, tuple[str, ...]]
    identity_assignments: Mapping[str, str]
    inventory_hash: str

    def __post_init__(self) -> None:
        _identifier(self.dataset_identifier, "dataset_identifier")
        _identifier(self.dataset_release, "dataset_release")
        _identifier(self.protocol_version, "protocol_version")
        _integer(self.seed, "seed")
        if self.allocation_kind not in ("counts", "proportions"):
            raise ValueError("allocation_kind must be counts or proportions")
        if not isinstance(self.requested_allocation, Mapping) or set(self.requested_allocation) != set(PARTITIONS):
            raise ValueError("requested allocation must specify all four partitions")
        if self.allocation_kind == "counts":
            requested_allocation = {key: _integer(self.requested_allocation[key], f"{key} count") for key in PARTITIONS}
            if sum(requested_allocation.values()) != sum(len(values) for values in self.partitions.values()):
                raise ValueError("requested counts do not match the number of assigned identities")
        elif self.allocation_kind == "proportions":
            requested_allocation = {}
            for key in PARTITIONS:
                value = self.requested_allocation[key]
                if isinstance(value, bool) or not isinstance(value, numbers.Real) or not math.isfinite(float(value)) or value < 0:
                    raise ValueError("requested proportions must be finite and non-negative")
                requested_allocation[key] = float(value)
            if not math.isclose(sum(requested_allocation.values()), 1.0, rel_tol=0.0, abs_tol=1e-9):
                raise ValueError("requested proportions must sum to 1")
        else:
            raise ValueError("allocation_kind must be counts or proportions")
        if not isinstance(self.partitions, Mapping) or set(self.partitions) != set(PARTITIONS):
            raise ValueError("partitions must specify all four protocol partitions")
        normalized_partitions: dict[str, tuple[str, ...]] = {}
        flat: list[str] = []
        for partition in PARTITIONS:
            if isinstance(self.partitions[partition], (str, bytes)):
                raise ValueError("partition identities must be sequences")
            values = tuple(_identifier(value, "identity identifier") for value in self.partitions[partition])
            if len(values) != len(set(values)):
                raise ValueError("partition contains duplicate identities")
            normalized_partitions[partition] = values
            flat.extend(values)
        if len(flat) != len(set(flat)):
            raise ValueError("identity partitions overlap")
        if not isinstance(self.identity_assignments, Mapping) or set(self.identity_assignments) != set(flat):
            raise ValueError("identity assignments do not cover partition identities exactly")
        for partition, identities in normalized_partitions.items():
            if any(self.identity_assignments.get(identity) != partition for identity in identities):
                raise ValueError("identity assignment disagrees with partition membership")
        _validate_hash(self.inventory_hash, "inventory_hash")
        object.__setattr__(self, "requested_allocation", MappingProxyType(requested_allocation))
        object.__setattr__(self, "partitions", MappingProxyType(normalized_partitions))
        object.__setattr__(self, "identity_assignments", MappingProxyType(dict(sorted(self.identity_assignments.items()))))

    @property
    def split_hash(self) -> str:
        return _sha256(self.to_dict(include_hash=False))

    def to_dict(self, *, include_hash: bool = True) -> dict[str, object]:
        result: dict[str, object] = {
            "schema": "identity_split_manifest_v1",
            "dataset_identifier": self.dataset_identifier,
            "dataset_release": self.dataset_release,
            "protocol_version": self.protocol_version,
            "seed": self.seed,
            "allocation_kind": self.allocation_kind,
            "requested_allocation": dict(self.requested_allocation),
            "partitions": {key: list(self.partitions[key]) for key in PARTITIONS},
            "identity_assignments": dict(sorted(self.identity_assignments.items())),
            "inventory_hash": self.inventory_hash,
        }
        if include_hash:
            result["split_hash"] = self.split_hash
        return result

    def to_json(self) -> str:
        return _canonical_json(self.to_dict())

    @classmethod
    def from_json(cls, payload: str) -> "IdentitySplitResult":
        try:
            data = json.loads(payload)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError("invalid identity split JSON") from exc
        fields = {"schema", "dataset_identifier", "dataset_release", "protocol_version", "seed",
                  "allocation_kind", "requested_allocation", "partitions", "identity_assignments",
                  "inventory_hash", "split_hash"}
        if not isinstance(data, dict) or set(data) != fields or data["schema"] != "identity_split_manifest_v1":
            raise ValueError("invalid identity split manifest fields or schema")
        if not isinstance(data["partitions"], dict) or set(data["partitions"]) != set(PARTITIONS):
            raise ValueError("split manifest must include exactly the four protocol partitions")
        assignments = data["identity_assignments"]
        if not isinstance(assignments, dict):
            raise ValueError("identity_assignments must be an object")
        partitions: dict[str, tuple[str, ...]] = {}
        flat: list[str] = []
        for key in PARTITIONS:
            values = data["partitions"][key]
            if not isinstance(values, list):
                raise ValueError("partition identity lists must be arrays")
            partitions[key] = tuple(_identifier(item, "identity identifier") for item in values)
            flat.extend(partitions[key])
        if len(flat) != len(set(flat)) or set(flat) != set(assignments):
            raise ValueError("partitions must assign every identity exactly once")
        if any(assignments.get(identity) != partition for partition, ids in partitions.items() for identity in ids):
            raise ValueError("identity assignments disagree with partitions")
        allocation = data["requested_allocation"]
        if not isinstance(allocation, dict):
            raise ValueError("requested_allocation must be an object")
        result = cls(
            _identifier(data["dataset_identifier"], "dataset_identifier"),
            _identifier(data["dataset_release"], "dataset_release"),
            _identifier(data["protocol_version"], "protocol_version"),
            _integer(data["seed"], "seed"),
            data["allocation_kind"], allocation,
            partitions, dict(assignments), _validate_hash(data["inventory_hash"], "inventory_hash"),
        )
        if data["split_hash"] != result.split_hash:
            raise ValueError("identity split manifest hash mismatch")
        return result

    def validate_inventory(self, identity_to_images: Mapping[str, Sequence[str]]) -> None:
        inventory = validate_identity_inventory(identity_to_images)
        if set(inventory) != set(self.identity_assignments) or inventory_sha256(inventory) != self.inventory_hash:
            raise ValueError("identity inventory does not match the split manifest")


def _validate_hash(value: object, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"{name} must be a lowercase SHA-256 hex digest")
    return value


def split_identities(identity_to_images: Mapping[str, Sequence[str]], *, dataset_identifier: str,
                     dataset_release: str, seed: int,
                     proportions: Mapping[str, float] | None = None,
                     counts: Mapping[str, int] | None = None,
                     protocol_version: str = "identity_split_v1") -> IdentitySplitResult:
    """Assign whole identities to four disjoint partitions deterministically.

    Proportions use largest-remainder apportionment in ``PARTITIONS`` order.
    Identities are ranked by SHA-256(seed, protocol, identity), avoiding any
    dependence on mapping insertion order or dataset image-level splits.
    """
    inventory = validate_identity_inventory(identity_to_images)
    dataset_identifier = _identifier(dataset_identifier, "dataset_identifier")
    dataset_release = _identifier(dataset_release, "dataset_release")
    protocol_version = _identifier(protocol_version, "protocol_version")
    seed = _integer(seed, "seed")
    requested = proportions if proportions is not None else counts
    allocation = _allocate(PARTITIONS, len(inventory), proportions=proportions, counts=counts)
    ranked = sorted(inventory, key=lambda identity: _stable_rank(seed, protocol_version, identity))
    partitions: dict[str, tuple[str, ...]] = {}
    assignments: dict[str, str] = {}
    cursor = 0
    for partition in PARTITIONS:
        selected = tuple(sorted(ranked[cursor:cursor + allocation[partition]]))
        cursor += allocation[partition]
        partitions[partition] = selected
        assignments.update({identity: partition for identity in selected})
    allocation_dict = {key: (int(requested[key]) if counts is not None else float(requested[key])) for key in PARTITIONS}  # type: ignore[index]
    return IdentitySplitResult(dataset_identifier, dataset_release, protocol_version, seed,
                               "counts" if counts is not None else "proportions", allocation_dict,
                               partitions, dict(sorted(assignments.items())), inventory_sha256(inventory))


@dataclass(frozen=True)
class EvaluationIdentitySubsets:
    parent_split_hash: str
    partition: str
    seed: int
    allocation_kind: str
    requested_allocation: Mapping[str, int | float]
    calibration_identities: tuple[str, ...]
    scoring_identities: tuple[str, ...]

    def __post_init__(self) -> None:
        _validate_hash(self.parent_split_hash, "parent_split_hash")
        if self.partition not in PARTITIONS:
            raise ValueError("unknown evaluation partition")
        _integer(self.seed, "seed")
        if self.allocation_kind not in ("counts", "proportions"):
            raise ValueError("allocation_kind must be counts or proportions")
        calibration = tuple(_identifier(value, "calibration identity") for value in self.calibration_identities)
        scoring = tuple(_identifier(value, "scoring identity") for value in self.scoring_identities)
        if len(calibration) != len(set(calibration)) or len(scoring) != len(set(scoring)):
            raise ValueError("calibration and scoring identity lists must not contain duplicates")
        if set(calibration) & set(scoring):
            raise ValueError("calibration and scoring identities must be disjoint")
        if not isinstance(self.requested_allocation, Mapping):
            raise ValueError("requested_allocation must be an object")
        if set(self.requested_allocation) != {"calibration", "scoring"}:
            raise ValueError("requested_allocation must specify calibration and scoring")
        if self.allocation_kind == "counts":
            requested = {key: _integer(self.requested_allocation[key], f"{key} count")
                         for key in ("calibration", "scoring")}
            if requested != {"calibration": len(calibration), "scoring": len(scoring)}:
                raise ValueError("requested counts do not match subset sizes")
        else:
            requested = {}
            for key in ("calibration", "scoring"):
                value = self.requested_allocation[key]
                if isinstance(value, bool) or not isinstance(value, numbers.Real) or not math.isfinite(float(value)) or value < 0:
                    raise ValueError("requested proportions must be finite and non-negative")
                requested[key] = float(value)
            if not math.isclose(sum(requested.values()), 1.0, rel_tol=0.0, abs_tol=1e-9):
                raise ValueError("requested proportions must sum to 1")
        object.__setattr__(self, "calibration_identities", tuple(sorted(calibration)))
        object.__setattr__(self, "scoring_identities", tuple(sorted(scoring)))
        object.__setattr__(self, "requested_allocation", MappingProxyType(requested))

    def to_dict(self) -> dict[str, object]:
        return {"schema": "evaluation_identity_subsets_v1", "parent_split_hash": self.parent_split_hash,
                "partition": self.partition, "seed": self.seed, "allocation_kind": self.allocation_kind,
                "requested_allocation": dict(self.requested_allocation),
                "calibration_identities": list(self.calibration_identities),
                "scoring_identities": list(self.scoring_identities)}

    def to_json(self) -> str:
        return _canonical_json(self.to_dict())


def split_calibration_scoring_identities(split: IdentitySplitResult, partition: str, *, seed: int,
                                         calibration_proportion: float | None = None,
                                         calibration_count: int | None = None) -> EvaluationIdentitySubsets:
    """Split one identity partition into disjoint calibration/scoring identities."""
    if not isinstance(split, IdentitySplitResult):
        raise TypeError("split must be an IdentitySplitResult")
    if partition not in PARTITIONS:
        raise ValueError("partition must be one of the four configured partitions")
    seed = _integer(seed, "seed")
    ids = split.partitions[partition]
    if (calibration_proportion is None) == (calibration_count is None):
        raise ValueError("provide exactly one of calibration_proportion or calibration_count")
    if calibration_count is not None:
        calibration_count = _integer(calibration_count, "calibration_count")
        if calibration_count > len(ids):
            raise ValueError("calibration_count exceeds partition size")
        allocation = {"calibration": calibration_count, "scoring": len(ids) - calibration_count}
        kind = "counts"
    else:
        p = calibration_proportion
        if isinstance(p, bool) or not isinstance(p, numbers.Real) or not math.isfinite(float(p)) or not 0 <= p <= 1:
            raise ValueError("calibration_proportion must be finite and within [0,1]")
        allocation = _allocate(("calibration", "scoring"), len(ids), proportions={"calibration": float(p), "scoring": 1.0 - float(p)})
        kind = "proportions"
    ranked = sorted(ids, key=lambda identity: _stable_rank(seed, f"{split.protocol_version}:{partition}:calibration", identity))
    calibration = tuple(sorted(ranked[:allocation["calibration"]]))
    scoring = tuple(sorted(ranked[allocation["calibration"]:]))
    requested: dict[str, int | float]
    if kind == "counts":
        requested = allocation
    else:
        assert calibration_proportion is not None
        requested = {"calibration": float(calibration_proportion), "scoring": 1.0 - float(calibration_proportion)}
    return EvaluationIdentitySubsets(split.split_hash, partition, seed, kind, requested, calibration, scoring)


@dataclass(frozen=True)
class PairRecord:
    pair_id: str
    first_image_id: str
    second_image_id: str
    first_identity_id: str
    second_identity_id: str
    label: str
    identity_relationship: str
    scenario: str
    first_role: str
    second_role: str

    def to_dict(self) -> dict[str, str]:
        return {"pair_id": self.pair_id, "first_image_id": self.first_image_id,
                "second_image_id": self.second_image_id, "first_identity_id": self.first_identity_id,
                "second_identity_id": self.second_identity_id, "label": self.label,
                "identity_relationship": self.identity_relationship, "scenario": self.scenario,
                "first_role": self.first_role, "second_role": self.second_role}


def _pair_id(payload: Mapping[str, object]) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()[:24]


@dataclass(frozen=True)
class PairManifest:
    dataset_identifier: str
    dataset_release: str
    protocol_version: str
    partition: str
    subset_role: str
    seed: int
    scenario: str
    pair_count_per_class: int
    sampling_strategy: str
    inventory_hash: str
    split_hash: str
    identity_ids: tuple[str, ...]
    calibration_identity_ids: tuple[str, ...]
    scoring_identity_ids: tuple[str, ...]
    excluded_insufficient_images: tuple[str, ...]
    pairs: tuple[PairRecord, ...]

    def to_dict(self, *, include_hash: bool = True) -> dict[str, object]:
        result: dict[str, object] = {
            "schema": "face_pair_manifest_v1", "dataset_identifier": self.dataset_identifier,
            "dataset_release": self.dataset_release, "protocol_version": self.protocol_version,
            "partition": self.partition, "subset_role": self.subset_role, "seed": self.seed,
            "scenario": self.scenario,
            "pair_generation_configuration": {"pair_count_per_class": self.pair_count_per_class,
                                              "sampling_strategy": self.sampling_strategy,
                                              "seed": self.seed},
            "inventory_hash": self.inventory_hash, "split_hash": self.split_hash,
            "identity_ids": list(self.identity_ids),
            "calibration_identity_ids": list(self.calibration_identity_ids),
            "scoring_identity_ids": list(self.scoring_identity_ids),
            "excluded_insufficient_images": list(self.excluded_insufficient_images),
            "pairs": [pair.to_dict() for pair in self.pairs],
        }
        if include_hash:
            result["manifest_hash"] = _sha256(result)
        return result

    @property
    def manifest_hash(self) -> str:
        return _sha256(self.to_dict(include_hash=False))

    def to_json(self) -> str:
        return _canonical_json(self.to_dict())

    @classmethod
    def from_json(cls, payload: str) -> "PairManifest":
        try:
            data = json.loads(payload)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError("invalid pair manifest JSON") from exc
        expected = {"schema", "dataset_identifier", "dataset_release", "protocol_version", "partition",
                    "subset_role", "seed", "scenario", "pair_generation_configuration", "inventory_hash",
                    "split_hash", "identity_ids", "calibration_identity_ids", "scoring_identity_ids",
                    "excluded_insufficient_images", "pairs", "manifest_hash"}
        if not isinstance(data, dict) or set(data) != expected or data["schema"] != "face_pair_manifest_v1":
            raise ValueError("invalid pair manifest fields or schema")
        cfg = data["pair_generation_configuration"]
        if not isinstance(cfg, dict) or set(cfg) != {"pair_count_per_class", "sampling_strategy", "seed"}:
            raise ValueError("invalid pair generation configuration")
        pairs_data = data["pairs"]
        if not isinstance(pairs_data, list):
            raise ValueError("pairs must be an array")
        fields = {"pair_id", "first_image_id", "second_image_id", "first_identity_id", "second_identity_id",
                  "label", "identity_relationship", "scenario", "first_role", "second_role"}
        pairs: list[PairRecord] = []
        for item in pairs_data:
            if not isinstance(item, dict) or set(item) != fields or any(not isinstance(v, str) for v in item.values()):
                raise ValueError("invalid pair record")
            pairs.append(PairRecord(**item))
        result = cls(
            _identifier(data["dataset_identifier"], "dataset_identifier"),
            _identifier(data["dataset_release"], "dataset_release"),
            _identifier(data["protocol_version"], "protocol_version"),
            _identifier(data["partition"], "partition"), _identifier(data["subset_role"], "subset_role"),
            _integer(data["seed"], "seed"), _identifier(data["scenario"], "scenario"),
            _integer(cfg["pair_count_per_class"], "pair_count_per_class", minimum=1),
            _identifier(cfg["sampling_strategy"], "sampling_strategy"),
            _validate_hash(data["inventory_hash"], "inventory_hash"), _validate_hash(data["split_hash"], "split_hash"),
            tuple(_identifier(v, "identity id") for v in data["identity_ids"]),
            tuple(_identifier(v, "calibration identity id") for v in data["calibration_identity_ids"]),
            tuple(_identifier(v, "scoring identity id") for v in data["scoring_identity_ids"]),
            tuple(_identifier(v, "excluded identity id") for v in data["excluded_insufficient_images"]), tuple(pairs),
        )
        if cfg["seed"] != result.seed or data["manifest_hash"] != result.manifest_hash:
            raise ValueError("pair manifest hash or seed mismatch")
        result._validate_structure()
        return result

    def _validate_structure(self) -> None:
        if self.partition not in PARTITIONS or self.scenario not in SCENARIOS:
            raise ValueError("unknown partition or scenario")
        if self.subset_role not in ("partition", "calibration", "scoring"):
            raise ValueError("subset_role must be partition, calibration, or scoring")
        if self.sampling_strategy != SAMPLING_STRATEGY:
            raise ValueError("unsupported sampling strategy")
        if set(self.calibration_identity_ids) & set(self.scoring_identity_ids):
            raise ValueError("calibration and scoring identities overlap")
        if len(self.identity_ids) != len(set(self.identity_ids)):
            raise ValueError("identity_ids contain duplicates")
        if self.subset_role == "calibration" and set(self.identity_ids) != set(self.calibration_identity_ids):
            raise ValueError("manifest identities do not match calibration subset")
        if self.subset_role == "scoring" and set(self.identity_ids) != set(self.scoring_identity_ids):
            raise ValueError("manifest identities do not match scoring subset")
        expected_roles = (("transformed_reference", "transformed_probe") if self.scenario == SCENARIOS[0]
                          else ("original_reference", "transformed_probe"))
        labels = {"genuine": 0, "impostor": 0}
        pair_keys: set[tuple[str, str]] = set()
        pair_ids: set[str] = set()
        for pair in self.pairs:
            if pair.scenario != self.scenario or (pair.first_role, pair.second_role) != expected_roles:
                raise ValueError("pair endpoint roles do not match scenario")
            if pair.label not in labels or pair.identity_relationship not in ("same", "different"):
                raise ValueError("invalid pair label or identity relationship")
            same = pair.first_identity_id == pair.second_identity_id
            if (pair.label, pair.identity_relationship) != (("genuine", "same") if same else ("impostor", "different")):
                raise ValueError("pair label is inconsistent with identity relationship")
            key = tuple(sorted((pair.first_image_id, pair.second_image_id)))
            if key[0] == key[1] or key in pair_keys:
                raise ValueError("duplicate or reversed duplicate pair")
            pair_keys.add(key)
            if pair.pair_id in pair_ids:
                raise ValueError("duplicate pair identifier")
            pair_ids.add(pair.pair_id)
            payload = {k: v for k, v in pair.to_dict().items() if k != "pair_id"}
            if pair.pair_id != _pair_id(payload):
                raise ValueError("pair identifier does not match pair contents")
            labels[pair.label] += 1
            if pair.first_identity_id not in self.identity_ids or pair.second_identity_id not in self.identity_ids:
                raise ValueError("pair endpoint identity is outside the declared subset")
        if labels["genuine"] != self.pair_count_per_class or labels["impostor"] != self.pair_count_per_class:
            raise ValueError("manifest genuine and impostor counts must be balanced and match configuration")

    def validate(self, identity_to_images: Mapping[str, Sequence[str]], split: IdentitySplitResult,
                 subsets: EvaluationIdentitySubsets | None = None) -> None:
        """Validate identifiers, roles, inventory, partition membership and labels."""
        self._validate_structure()
        inventory = validate_identity_inventory(identity_to_images)
        split.validate_inventory(inventory)
        if inventory_sha256(inventory) != self.inventory_hash or split.split_hash != self.split_hash:
            raise ValueError("manifest inventory or split does not match supplied inputs")
        if self.dataset_identifier != split.dataset_identifier or self.dataset_release != split.dataset_release:
            raise ValueError("manifest dataset does not match identity split")
        partition_ids = set(split.partitions[self.partition])
        if not set(self.identity_ids).issubset(partition_ids):
            raise ValueError("manifest contains identities from outside its declared partition")
        if self.subset_role == "partition" and set(self.identity_ids) != partition_ids:
            raise ValueError("partition manifest must declare every identity in that partition")
        if self.subset_role in ("calibration", "scoring"):
            if subsets is None:
                raise ValueError("calibration/scoring manifests require their identity subset definition")
            if subsets.parent_split_hash != split.split_hash or subsets.partition != self.partition:
                raise ValueError("evaluation subsets do not belong to the supplied split partition")
            if (set(subsets.calibration_identities) & set(subsets.scoring_identities)
                    or set(subsets.calibration_identities) | set(subsets.scoring_identities) != partition_ids):
                raise ValueError("calibration/scoring identities must be disjoint and cover the evaluation partition")
            if tuple(sorted(subsets.calibration_identities)) != self.calibration_identity_ids or tuple(sorted(subsets.scoring_identities)) != self.scoring_identity_ids:
                raise ValueError("manifest calibration/scoring identities mismatch")
        elif subsets is not None and (subsets.parent_split_hash != split.split_hash or subsets.partition != self.partition):
            raise ValueError("evaluation subsets do not belong to the supplied split partition")
        for pair in self.pairs:
            if pair.first_image_id not in inventory.get(pair.first_identity_id, ()):
                raise ValueError("first image reference does not belong to its declared identity")
            if pair.second_image_id not in inventory.get(pair.second_identity_id, ()):
                raise ValueError("second image reference does not belong to its declared identity")
        expected_excluded = tuple(sorted(identity for identity in self.identity_ids if len(inventory[identity]) < 2))
        if self.excluded_insufficient_images != expected_excluded:
            raise ValueError("insufficient-image exclusion list is incorrect")


def generate_pair_manifest(identity_to_images: Mapping[str, Sequence[str]], split: IdentitySplitResult, *,
                           partition: str, subset_role: str, pair_count_per_class: int, seed: int,
                           scenario: str, subsets: EvaluationIdentitySubsets | None = None,
                           protocol_version: str = "face_pair_protocol_v1",
                           max_attempts_per_pair: int = 200) -> PairManifest:
    """Generate balanced, bounded genuine/impostor samples without enumerating all pairs.

    Identities are selected round-robin for genuine pairs and by minimum current
    endpoint use for impostor pairs. Thus prolific identities cannot dominate
    merely because they contain more images. Pair endpoints retain explicit
    reference/probe roles in both scenarios.
    """
    if not isinstance(split, IdentitySplitResult):
        raise TypeError("split must be an IdentitySplitResult")
    if partition not in PARTITIONS or scenario not in SCENARIOS:
        raise ValueError("unknown partition or scenario")
    if subset_role not in ("partition", "calibration", "scoring"):
        raise ValueError("subset_role must be partition, calibration, or scoring")
    pair_count_per_class = _integer(pair_count_per_class, "pair_count_per_class", minimum=1)
    seed = _integer(seed, "seed")
    max_attempts_per_pair = _integer(max_attempts_per_pair, "max_attempts_per_pair", minimum=1)
    protocol_version = _identifier(protocol_version, "protocol_version")
    inventory = validate_identity_inventory(identity_to_images)
    split.validate_inventory(inventory)
    if subset_role == "partition":
        if subsets is not None:
            raise ValueError("subsets must be omitted when subset_role is partition")
        selected_ids = tuple(split.partitions[partition])
        calibration_ids: tuple[str, ...] = ()
        scoring_ids: tuple[str, ...] = ()
    else:
        if subsets is None or subsets.parent_split_hash != split.split_hash or subsets.partition != partition:
            raise ValueError("calibration/scoring generation requires matching evaluation identity subsets")
        calibration_ids = tuple(sorted(subsets.calibration_identities))
        scoring_ids = tuple(sorted(subsets.scoring_identities))
        selected_ids = calibration_ids if subset_role == "calibration" else scoring_ids
    if not selected_ids:
        raise ValueError("selected identity subset is empty")
    if not set(selected_ids).issubset(split.partitions[partition]):
        raise ValueError("selected identities are outside the requested partition")
    selected = {identity: inventory[identity] for identity in selected_ids}
    eligible = [identity for identity in selected_ids if len(selected[identity]) >= 2]
    excluded = tuple(sorted(set(selected_ids) - set(eligible)))
    if len(eligible) < 2:
        raise ValueError("at least two identities with two or more images are required for balanced pair classes")
    genuine_capacity = sum(len(selected[identity]) * (len(selected[identity]) - 1) // 2 for identity in eligible)
    impostor_capacity = sum(len(selected[a]) * len(selected[b]) for i, a in enumerate(eligible) for b in eligible[i + 1:])
    if pair_count_per_class > genuine_capacity or pair_count_per_class > impostor_capacity:
        raise ValueError("requested pair count exceeds the available unique genuine or impostor pairs")

    rng = random.Random(seed)
    roles = (("transformed_reference", "transformed_probe") if scenario == SCENARIOS[0]
             else ("original_reference", "transformed_probe"))
    unique_keys: set[tuple[str, str]] = set()
    genuine_raw: list[tuple[str, str, str, str]] = []
    attempts = pair_count_per_class * max_attempts_per_pair
    identity_order = sorted(eligible)
    rng.shuffle(identity_order)
    cursor = 0
    while len(genuine_raw) < pair_count_per_class and attempts > 0:
        identity = identity_order[cursor % len(identity_order)]
        cursor += 1
        attempts -= 1
        first, second = rng.sample(selected[identity], 2)
        key = tuple(sorted((first, second)))
        if key not in unique_keys:
            unique_keys.add(key)
            genuine_raw.append((identity, identity, first, second))
    if len(genuine_raw) != pair_count_per_class:
        raise ValueError("bounded sampling could not produce the requested unique genuine pairs")

    impostor_raw: list[tuple[str, str, str, str]] = []
    endpoint_uses = {identity: 0 for identity in eligible}
    attempts = pair_count_per_class * max_attempts_per_pair
    while len(impostor_raw) < pair_count_per_class and attempts > 0:
        minimum = min(endpoint_uses.values())
        least_used = [identity for identity in eligible if endpoint_uses[identity] == minimum]
        if len(least_used) >= 2:
            first_identity, second_identity = rng.sample(least_used, 2)
        else:
            first_identity = least_used[0]
            next_use = min(value for identity, value in endpoint_uses.items() if identity != first_identity)
            candidates = [identity for identity in eligible if identity != first_identity and endpoint_uses[identity] == next_use]
            second_identity = rng.choice(candidates)
        first_image = rng.choice(selected[first_identity])
        second_image = rng.choice(selected[second_identity])
        attempts -= 1
        key = tuple(sorted((first_image, second_image)))
        if key in unique_keys:
            continue
        unique_keys.add(key)
        endpoint_uses[first_identity] += 1
        endpoint_uses[second_identity] += 1
        impostor_raw.append((first_identity, second_identity, first_image, second_image))
    if len(impostor_raw) != pair_count_per_class:
        raise ValueError("bounded sampling could not produce the requested unique impostor pairs")

    records: list[PairRecord] = []
    common = {"scenario": scenario, "first_role": roles[0], "second_role": roles[1]}
    for label, raw_pairs in (("genuine", genuine_raw), ("impostor", impostor_raw)):
        for first_identity, second_identity, first_image, second_image in raw_pairs:
            record_payload: dict[str, object] = {
                "first_image_id": first_image, "second_image_id": second_image,
                "first_identity_id": first_identity, "second_identity_id": second_identity,
                "label": label, "identity_relationship": "same" if label == "genuine" else "different",
                **common,
            }
            records.append(PairRecord(_pair_id(record_payload), **record_payload))
    # Stable interleaving keeps both labels spread throughout the manifest.
    records.sort(key=lambda pair: (pair.label != "genuine", pair.pair_id))
    manifest = PairManifest(split.dataset_identifier, split.dataset_release, protocol_version, partition,
                            subset_role, seed, scenario, pair_count_per_class, SAMPLING_STRATEGY,
                            inventory_sha256(inventory), split.split_hash, tuple(sorted(selected_ids)),
                            calibration_ids, scoring_ids, excluded, tuple(records))
    manifest._validate_structure()
    manifest.validate(inventory, split, subsets)
    return manifest
