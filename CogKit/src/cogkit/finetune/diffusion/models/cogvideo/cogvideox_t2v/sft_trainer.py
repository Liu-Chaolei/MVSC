# -*- coding: utf-8 -*-


from cogkit.finetune import register

from ..cogvideox_t2v.lora_trainer import CogVideoXT2VLoraTrainer


class CogVideoXT2VSftTrainer(CogVideoXT2VLoraTrainer):
    pass


register("cogvideox-t2v", "sft", CogVideoXT2VSftTrainer)
