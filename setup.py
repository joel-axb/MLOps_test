# MLOps_test/setup.py
from setuptools import setup, find_packages

setup(
    name="mlops_test",
    version="0.1",
    packages=find_packages(),  # modeling, modeling.commons 모두 자동 탐색
    install_requires=["pandas"],  # 필요한 패키지 추가 가능
    python_requires=">=3.8"
)