# -*- coding: utf-8 -*-


from cogkit.finetune import register

from ..cogvideox_i2v.sft_trainer import CogVideoXI2VSftTrainer


class CogVideoX1_5I2VSftTrainer(CogVideoXI2VSftTrainer):
    pass


register("cogvideox1.5-i2v", "sft", CogVideoX1_5I2VSftTrainer)
