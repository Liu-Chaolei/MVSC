# -*- coding: utf-8 -*-


from cogkit.finetune import register

from ..cogvideox_i2v.lora_trainer import CogVideoXI2VLoraTrainer


class CogVideoXI2VSftTrainer(CogVideoXI2VLoraTrainer):
    pass


register("cogvideox-i2v", "sft", CogVideoXI2VSftTrainer)
