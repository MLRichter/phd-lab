from setuptools import setup, find_packages
setup(
    name="phd_lab",
    version="0.1",
    packages=find_packages(),
    author="Mats L. Richter",
    author_email="matsrichter@gmail.com",
    description="Experimental Repository for conducting saturation related experiments.",
    scripts=["compute_receptive_field.py",
             "infer_with_altering_delta.py",
             "train_model.py",
             "train_probes.py"],
    install_requires=[
        "delve>=0.1.41",
        "torch",
        "torchvision",
        "numpy",
        "scipy",
        "pandas",
        "matplotlib"
    ]
)