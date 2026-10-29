"""Pydantic v2 validation, unit normalization, and GD&T hierarchy resolution for assembly.yaml."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator


UNIT_TO_MM: dict[str, float] = {
    "mm": 1.0,
    "in": 25.4,
    "um": 1e-3,
}


class AssemblyHeader(BaseModel):
    name: str
    units: Literal["mm", "in", "um"] = "mm"
    default_process_sigma: float = Field(default=3.0, gt=0.0)
    mmc_statistical_mode: Literal["acceptance_truncated", "fully_consumed"] = (
        "acceptance_truncated"
    )


class PoseSpec(BaseModel):
    pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rpy_deg: tuple[float, float, float] = (0.0, 0.0, 0.0)


class BoundsSpec(BaseModel):
    lx: float = Field(default=10.0, gt=0.0)
    ly: float = Field(default=10.0, gt=0.0)
    lz: float = Field(default=10.0, gt=0.0)


class SizeSpec(BaseModel):
    nom: float = Field(gt=0.0)
    upper_dev: float = 0.0
    lower_dev: float = 0.0

    @model_validator(mode="after")
    def _check_devs(self) -> SizeSpec:
        if self.upper_dev < self.lower_dev:
            raise ValueError(
                f"upper_dev ({self.upper_dev}) must be >= lower_dev ({self.lower_dev})"
            )
        return self

    @property
    def d_max(self) -> float:
        return self.nom + self.upper_dev

    @property
    def d_min(self) -> float:
        return self.nom + self.lower_dev

    @property
    def r_mid(self) -> float:
        return 0.25 * (self.d_max + self.d_min)

    @property
    def t_size(self) -> float:
        return self.d_max - self.d_min


class GDTSpec(BaseModel):
    position: float | None = Field(default=None, ge=0.0)
    profile: float | None = Field(default=None, ge=0.0)
    flatness: float | None = Field(default=None, ge=0.0)
    parallelism: float | None = Field(default=None, ge=0.0)
    perpendicularity: float | None = Field(default=None, ge=0.0)
    angularity: float | None = Field(default=None, ge=0.0)
    modifier: Literal["RFS", "MMC", "LMC"] = "RFS"
    datum_refs: list[str] = Field(default_factory=list)


class CostModelSpec(BaseModel):
    base_a: float = Field(default=1.0, ge=0.0)
    scale_b: float = Field(default=0.1, ge=0.0)
    exponent: float = Field(default=2.0, gt=0.0)
    bounds: tuple[float, float] = (0.01, 0.50)

    @model_validator(mode="after")
    def _check_bounds(self) -> CostModelSpec:
        if self.bounds[0] <= 0.0 or self.bounds[1] <= self.bounds[0]:
            raise ValueError(
                f"Cost model bounds {self.bounds} must satisfy 0 < t_min < t_max"
            )
        return self


class FeatureSpec(BaseModel):
    id: str
    type: Literal[
        "planar", "cylinder_pin", "cylinder_hole", "slot_width", "sphere", "point"
    ]
    step_face_tag: str | None = None
    nominal_frame: PoseSpec = Field(default_factory=PoseSpec)
    bounds: BoundsSpec = Field(default_factory=BoundsSpec)
    size: SizeSpec | None = None
    length: float = Field(default=10.0, gt=0.0)
    gdt: GDTSpec = Field(default_factory=GDTSpec)
    cost_model: CostModelSpec | None = None

    def resolve_tolerances(self, is_primary_datum: bool = False) -> tuple[float, float, float]:
        """Resolve (t_pos, t_orient, t_size) in mm according to ASME Y14.5 hierarchy rules."""
        g = self.gdt
        t_size = self.size.t_size if self.size is not None else 0.0

        t_pos = next(
            (v for v in (g.position, g.profile) if v is not None),
            0.0,
        )
        t_orient_explicit = next(
            (
                v
                for v in (g.perpendicularity, g.parallelism, g.angularity)
                if v is not None
            ),
            None,
        )
        if t_orient_explicit is not None:
            t_orient = min(t_orient_explicit, t_pos) if t_pos > 0.0 else t_orient_explicit
        elif t_pos > 0.0:
            t_orient = t_pos
        elif is_primary_datum and g.flatness is not None:
            # Form tolerance on a primary datum contact acts as seating rock envelope
            t_orient = g.flatness
        else:
            t_orient = 0.0

        return t_pos, t_orient, t_size


class PartSpec(BaseModel):
    id: str
    fixed: bool = False
    step_file: str | None = None
    nominal_pose: PoseSpec = Field(default_factory=PoseSpec)
    features: list[FeatureSpec]

    @model_validator(mode="after")
    def _validate_part_features(self) -> PartSpec:
        seen: set[str] = set()
        for f in self.features:
            if f.id in seen:
                raise ValueError(f"Duplicate feature id '{f.id}' on part '{self.id}'")
            seen.add(f.id)
        for f in self.features:
            for dref in f.gdt.datum_refs:
                if dref not in seen:
                    raise ValueError(
                        f"Feature '{self.id}.{f.id}' references unknown datum '{dref}' on part '{self.id}'"
                    )
        return self


class ContactOffsetSpec(BaseModel):
    source: float = 0.0
    target: float = 0.0


class MateSpec(BaseModel):
    id: str
    type: Literal[
        "planar_coincident",
        "parallel_pin_hole",
        "clocking_pin_slot",
        "line_contact",
        "point_contact",
        "coaxial_rigid",
        "cylindrical_clearance",
        "spherical_joint",
        "rigid_weld",
    ]
    source: str
    target: str
    contact_offset_z: ContactOffsetSpec = Field(default_factory=ContactOffsetSpec)
    kinematic_mode: Literal["centered", "float_random", "gravity_bias"] = "centered"
    bias_vector: tuple[float, float] = (1.0, 0.0)


class JamCheckSpec(BaseModel):
    id: str
    source: str
    target: str
    axis: Literal["xy", "x", "y"] = "xy"


class LimitsSpec(BaseModel):
    lsl: float
    usl: float

    @model_validator(mode="after")
    def _check_limits(self) -> LimitsSpec:
        if self.usl <= self.lsl:
            raise ValueError(f"USL ({self.usl}) must be strictly greater than LSL ({self.lsl})")
        return self


class MeasurementSpec(BaseModel):
    id: str
    description: str = ""
    type: Literal[
        "distance_xyz", "point_to_plane_distance", "coaxiality", "angular_tilt"
    ]
    axis: tuple[float, float, float] = (0.0, 0.0, 1.0)
    source: str
    target: str
    limits: LimitsSpec


class AssemblySpec(BaseModel):
    schema_version: str = "1.0"
    assembly: AssemblyHeader
    parts: list[PartSpec]
    mates: list[MateSpec]
    jam_checks: list[JamCheckSpec] = Field(default_factory=list)
    measurements: list[MeasurementSpec]

    @model_validator(mode="after")
    def _validate_references_and_ground(self) -> AssemblySpec:
        if not self.parts:
            raise ValueError("Assembly must contain at least one part")
        if not any(p.fixed for p in self.parts):
            raise ValueError("Assembly must have at least one fixed ground part (fixed: true)")

        valid_refs: dict[str, FeatureSpec] = {}
        for p in self.parts:
            for f in p.features:
                valid_refs[f"{p.id}.{f.id}"] = f

        for m in self.mates:
            if m.source not in valid_refs:
                raise ValueError(f"Mate '{m.id}' references unknown source feature '{m.source}'")
            if m.target not in valid_refs:
                raise ValueError(f"Mate '{m.id}' references unknown target feature '{m.target}'")

        for jc in self.jam_checks:
            if jc.source not in valid_refs:
                raise ValueError(f"Jam check '{jc.id}' references unknown source '{jc.source}'")
            if jc.target not in valid_refs:
                raise ValueError(f"Jam check '{jc.id}' references unknown target '{jc.target}'")

        for meas in self.measurements:
            if meas.source not in valid_refs:
                raise ValueError(
                    f"Measurement '{meas.id}' references unknown source '{meas.source}'"
                )
            if meas.target not in valid_refs:
                raise ValueError(
                    f"Measurement '{meas.id}' references unknown target '{meas.target}'"
                )

        return self


def load_assembly_spec(path: str | Path) -> AssemblySpec:
    """Load and validate an AssemblySpec from a YAML or JSON file."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return AssemblySpec.model_validate(raw)


def save_assembly_spec(spec: AssemblySpec, path: str | Path) -> None:
    """Serialize an AssemblySpec back to YAML."""
    data = spec.model_dump(exclude_none=True)
    Path(path).write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
