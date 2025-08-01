#!/usr/bin/env python3
"""
Setup script for the Optimized Image Classifier
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="optimized-image-classifier",
    version="1.0.0",
    author="Performance Optimization Team",
    description="A highly optimized TensorFlow-based image classifier for emotion detection",
    long_description=long_description,
    long_description_content_type="text/markdown",
    py_modules=["optimized_image_classifier"],
    python_requires=">=3.8",
    install_requires=requirements,
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    entry_points={
        "console_scripts": [
            "optimized-classifier=optimized_image_classifier:main",
        ],
    },
    keywords="tensorflow, image-classification, optimization, performance, cnn, deep-learning",
)