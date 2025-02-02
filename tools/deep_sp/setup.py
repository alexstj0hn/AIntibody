from setuptools import setup

setup(
    name="deepsp",
    version="0.1",
    py_modules=["deepsp"],
    install_requires=[],
    entry_points={
        "console_scripts": [
            "deepsp=deepsp:main",
        ],
    },
)
