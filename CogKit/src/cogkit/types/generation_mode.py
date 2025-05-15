# -*- coding: utf-8 -*-


import enum


class GenerationMode(enum.Enum):
    TextToImage = "t2i"
    CtrlTextToImage = "ct2i"

    TextToVideo = "t2v"
    ImageToVideo = "i2v"
