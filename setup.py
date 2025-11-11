from setuptools import setup, find_packages

setup(
    name='btc_quant_prob',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'pandas',
        'numpy',
        'scipy',
        'scikit-learn',
        'lightgbm',
        'torch',
        'optuna',
        'ta',
        'joblib',
        'fastapi',
        'uvicorn',
        'matplotlib',
        'seaborn',
        'tqdm',
        'pytest',
    ],
    author='Jordan Theovandy',
    author_email='dev@example.com',
    description='A production-grade system for probabilistic Bitcoin price forecasting.',
    url='https://github.com/user/btc-quant-prob',
)