from setuptools import setup

setup(
    name="tap",
    version="0.1",
    py_modules=["tap"],
    install_requires=[],
    entry_points={
        "console_scripts": [
            "tap=tap:main",
        ],
    },
)
