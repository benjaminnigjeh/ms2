from setuptools import find_packages, setup

setup(
    name="ms2",
    version="0.0.3",
    author="Ben Nigjeh",
    author_email="benjamin.nigjeh@gmail.com",
    install_requires=["pyteomics", "matplotlib", "pandas", "numpy", "wget", "h5py", "tensorflow", "keras"],
    packages=find_packages()
)



