# -*- coding: utf-8 -*-
"""
SoulX-Podcast WebUI Package

This package provides a modular Gradio-based web interface for SoulX-Podcast.
"""

from .interface import render_interface
from .synthesis import initiate_model, get_model, get_dataset

__all__ = [
    "render_interface",
    "initiate_model",
    "get_model",
    "get_dataset",
]
