from setuptools import setup, find_packages

setup(
    name="voyage-analytics",
    version="1.0.0",
    description="Corporate Travel Intelligence Platform — Flight Data ML Pipeline",
    author="Voyage Analytics Team",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "pandas>=2.0",
        "numpy>=1.24",
        "scikit-learn>=1.3",
        "xgboost>=2.0",
        "lightgbm>=4.0",
        "flask>=3.0",
        "flask-cors>=4.0",
        "gunicorn>=21.0",
        "streamlit>=1.30",
        "plotly>=5.18",
        "mlflow>=2.10",
        "joblib>=1.3",
        "scipy>=1.11",
        "pyyaml>=6.0",
    ],
)
