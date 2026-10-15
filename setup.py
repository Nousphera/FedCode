"""Install FedCode's source and native Flower application."""
from pathlib import Path
from setuptools import find_packages, setup

source_root = 'src' if Path('src').is_dir() else '.'

setup(
    name='fedcode', version='0.2.0',
    description='Communication-Efficient Federated Learning via Transferring Codebooks',
    long_description=Path('README.md').read_text() if Path('README.md').exists() else 'FedCode federated learning.',
    long_description_content_type='text/markdown', python_requires='>=3.11,<3.14',
    install_requires=['torch==2.8.0', 'torchvision==0.23.0', 'flwr[simulation]==1.40.0',
                      'numpy>=2,<3', 'scikit-learn>=1.5,<2', 'tomli-w>=1,<2'],
    package_dir={'': source_root}, packages=find_packages(source_root, include=['fedcode', 'fedcode.*']),
    data_files=[('share/fedcode', ['setup.py', 'LICENSE'])],
    entry_points={'console_scripts': ['fedcode-experiment=fedcode.experiment:main']},
    license_files=['LICENSE'],
)
