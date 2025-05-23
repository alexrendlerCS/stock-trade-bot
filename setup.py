from setuptools import setup, find_packages

setup(
    name="ai-stock-backend",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "alpaca-trade-api==3.0.2",
        "fastapi==0.104.1",
        "uvicorn==0.24.0",
        "sqlalchemy==2.0.23",
        "python-dotenv==1.0.0",
        "pandas==2.1.3",
        "numpy==1.26.2",
        "scikit-learn==1.3.2",
        "ta==0.10.2",
        "apscheduler==3.10.4",
        "pytest==7.4.3",
        "python-jose==3.3.0",
        "passlib==1.7.4",
        "python-multipart==0.0.6",
        "requests==2.31.0",
        "pytz==2023.3",
    ],
) 