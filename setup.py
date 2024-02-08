from setuptools import setup, find_packages

setup(
    name='torq',
    version='0.0.1',
    description='Quantization simulation of neural networks with PyTorch',
    url='https://github.com/insuofficial/pytorch-quantization.git',
    author='Insu Choi',
    author_email='insuoffical@yonsei.ac.kr',
    license='Apache License 2.0',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'torch',
        'timm'
    ]
)
