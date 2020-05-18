"""BMI Live clinic"""
import os
from .diffusion import Diffusion
from .bmi_diffusion import BmiDiffusion


__all__ = ['Diffusion', 'BmiDiffusion']

pkg_directory = os.path.dirname(__file__)
data_directory = os.path.join(pkg_directory, 'data')
