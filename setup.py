from setuptools import setup, find_packages

setup(
    name="steganography",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
        "Pillow>=9.0.0",
        "scipy>=1.7.0",
        "scikit-image>=0.19.0",
        "opencv-python>=4.5.0",
        "cryptography>=3.4.0",
        "questionary>=2.0.1",
        "rich>=10.0.0",
        "reedsolo>=1.5.0"
    ],
    extras_require={
        'dev': [
            'pytest>=6.0',
            'black>=21.0',
            'flake8>=3.9',
            'pytest-cov>=2.12',
        ],
    },
    entry_points={
        'console_scripts': [
            'stego=steganography.cli:main',
            'stego-detect=stego_detector.cli:main',
        ],
    },
    author="eboygarcia",
    author_email="eboygarcia@proton.me",
    description="Food-grade steganography detector",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/reneboygarcia/steganography-project",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
) 