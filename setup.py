# setup.py
from setuptools import setup, find_packages
import os

def read_file(fname: str) -> str:
    base_dir = os.path.abspath(os.path.dirname(__file__))
    path = os.path.join(base_dir, fname)
    if not os.path.exists(path):
        return ""
    with open(path, "r", encoding="utf-8") as fh:
        return fh.read()

def read_requirements(fname: str = "requirements.txt"):
    base_dir = os.path.abspath(os.path.dirname(__file__))
    path = os.path.join(base_dir, fname)
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as fh:
        lines = []
        for line in fh:
            ln = line.strip()
            if not ln or ln.startswith("#"):
                continue
            lines.append(ln)
        return lines

if __name__ == "__main__":
    long_description = read_file("README.md")
    requirements = read_requirements("requirements.txt")

    setup(
        name="llm-apm",
        version="1.1.2",
        author="Suhas O",
        author_email="suhaso2002@gmail.com",
        description="LLM Application Performance Monitoring - Real-time monitoring for LLM-powered applications",
        long_description=long_description,
        long_description_content_type="text/markdown",
        url="https://github.com/yourusername/llm-apm",
        packages=find_packages(),
        classifiers=[
            "Development Status :: 4 - Beta",
            "Intended Audience :: Developers",
            "Topic :: Software Development :: Libraries :: Python Modules",
            "Topic :: System :: Monitoring",
            "License :: OSI Approved :: MIT License",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3.10",
            "Programming Language :: Python :: 3.11",
        ],
        python_requires=">=3.8",
        install_requires=requirements,
        extras_require={
            "postgresql": ["psycopg2-binary", "SQLAlchemy[asyncio]>=1.4.0", "asyncpg"],
            "dev": ["pytest", "pytest-asyncio", "black", "flake8"],
        },
        include_package_data=True,
        package_data={
            "llm_apm": ["*.json", "*.yaml"],
        },
    )
