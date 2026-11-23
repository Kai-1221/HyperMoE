from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class Config:
    wsi_dir: Path = Path("data/wsi")
    mri_dir: Path = Path("data/mri")
    split_dir: Path = Path("data/splits")
    out_dir: Path = Path("runs")

    feature_dim: int = 512
    num_levels: int = 5
    level_names: tuple[str, ...] = ("patch", "region", "slide", "roi", "fs")
    wsi_grid: int = 4
    num_classes: int = 2
    class_weights: tuple[float, float] = (1.0, 2.5)

    temporal: str = "t2"
    temporal_dim: int = 512
    temporal_heads: int = 4
    temporal_dropout: float = 0.1
    temporal_share: bool = False
    temporal_delta: bool = False

    geometry: str = "hyper"
    hierarchy_mode: str = "full"
    level_adaptor: bool = False
    level_adaptor_hidden: int = 64
    level_radii: list[float] = field(default_factory=list)
    level_radius_band: float = 0.05
    cone_level_edges: bool = False
    head_radius: float = 0.0
    num_experts: int = 2
    moe_hidden: int = 512
    moe_dropout: float = 0.2
    router_temperature_init: float = 1.0
    router_mode: str = "proto"
    router_tau_init: float = 1.0
    expert_indep_init: bool = False
    expert_split: str = "mixed"
    fusion_weight_mode: str = "gate"
    fusion_w_init: float = 0.5
    expert_aux: float = 0.0
    patch_aux: float = 0.0
    patch_topk_frac: float = 0.05
    exp_map_max_norm: float = 3.0
    class_cone_k: float = 0.5
    cone_aperture_deg: float = 0.0
    cone_normalize: bool = False
    cone_tie_class: bool = False

    proto_init_radius: float = 1.0
    proto_init_sep: float = 1.2

    lambda_bal: float = 0.05
    lambda_hyp: float = 0.1
    lambda_ent: float = 0.01
    lambda_cone: float = 0.01
    lambda_proto_sep: float = 0.01
    hyp_margin: float = 0.5
    hyp_cap: float = 2.0
    ent_margin: float = 0.1
    proto_sep_margin: float = 2.0
    warmup_epochs: int = 10

    folds: int = 5
    seed: int = 42
    epochs: int = 150
    batch_size: int = 1
    grad_accum: int = 16
    lr: float = 2e-5
    weight_decay: float = 5e-3
    grad_clip: float = 1.0
    patience: int = 12
    lr_warmup_epochs: int = 3
    smooth_window: int = 3
    min_select_epoch: int = 0
    lr_patience: int = 6
    lr_factor: float = 0.5
    num_workers: int = 0
    modality_dropout: float = 0.2
    modality_mode: str = "both"

    bootstrap_n: int = 2000


    def to_dict(self) -> dict:
        d = asdict(self)
        for k, v in list(d.items()):
            if isinstance(v, Path):
                d[k] = str(v)
            elif isinstance(v, tuple):
                d[k] = list(v)
        return d

    def validate(self) -> None:
        assert self.num_levels == len(self.level_names) == 5
        assert self.level_names == ("patch", "region", "slide", "roi", "fs")
        assert self.temporal in {"t0", "t1", "t2"}
        assert self.geometry in {"hyper", "euclidean"}
        assert self.hierarchy_mode in {"full", "no_topo", "no_levels"}
        assert self.expert_split in {"mixed", "modality"}
        assert self.fusion_weight_mode in {"gate", "scalar", "fixed"}
        if self.level_radii:
            assert len(self.level_radii) in {3, 5}, (
                "level_radii must give 3 (patch, region, slide) or 5 "
                "(patch, region, slide, roi, fs) positive radii"
            )
            assert all(float(r) > 0.0 for r in self.level_radii)
            assert float(self.level_radius_band) >= 0.0
        assert self.num_classes == 2
        assert self.feature_dim == 512
        assert self.batch_size == 1, "this pipeline currently expects batch_size=1"
        assert self.folds == 5

DEFAULT = Config()
