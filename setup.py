from setuptools import find_packages, setup
from hcrystalball import __version__

with open("README.md") as f:
    long_description = f.read()

setup(
    name="hcrystalball",
    setup_requires=["setuptools"],
    version=__version__,
    description="A library that unifies the API for most commonly "
    "used techniques for time-series forecasting in the Python ecosystem.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://hcrystalball.readthedocs.io/",
    author="Data Science Team @ HeidelbergCement",
    author_email="datascience@heidelbergcement.com",
    license="MIT",
    classifiers=[  # Optional
        "Development Status :: 4 - Beta",
        "Programming Language :: Python",
        "Intended Audience :: Developers",
        "Intended Audience :: Information Technology",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries",
    ],
    packages=find_packages(include=["hcrystalball", "hcrystalball.*"]),
    package_data={"hcrystalball": ["data/*"]},
    install_requires=[
        "numpy>=1.18",
        "pandas>=1.0",
        "scipy>=1.4",
        "workalendar>=10.1",
        "scikit-learn>=0.23",
        "matplotlib",
    ],
    python_requires=">=3.7",
)
