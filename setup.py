from setuptools import setup, find_packages

with open('requirements.txt', 'r', encoding='utf-8') as f:
    required = f.read().splitlines()

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='pidtransformer',
    version='1.0.4',
    author='Architect & Gem',
    description='A Transformer architecture with internal PID control for stabilizing learning dynamics.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/Kangmin22/PID-Transformer-PROJECT',
    packages=find_packages(),
    install_requires=required,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires='>=3.9',
)
