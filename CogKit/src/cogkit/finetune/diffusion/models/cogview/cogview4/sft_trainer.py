# -*- coding: utf-8 -*-


from cogkit.finetune import register

from .lora_trainer import Cogview4Trainer


class Cogview4SFTTrainer(Cogview4Trainer):
    pass


register("cogview4-6b", "sft", Cogview4SFTTrainer)
