from lightx2v_train.utils.registry import build_trainer

from .dmd import DmdTrainer, VideoArDmdTrainer, VideoDmdTrainer
from .dopsd import DopsdTrainer
from .flow import FlowMatchingTrainer
from .tf import TFTrainer
from .s2v_dmd import S2VDmdLoraTrainer

ARDmdTrainer = VideoArDmdTrainer

__all__ = ["build_trainer", "ARDmdTrainer", "DmdTrainer", "FlowMatchingTrainer", "TFTrainer", "VideoArDmdTrainer", "VideoDmdTrainer", "DopsdTrainer", "S2VDmdLoraTrainer"]
