from setuptools import setup

setup(
    name="cylinder_t_matrix",
    version="1.0",
    author="Alan Zhan",
#    author_email="",
#    url="",
    description="Compute T-matrix and derivative of T-matrix",
    long_description="This is a routine for computing the T-matrix and T-matrix derivatives (with respect to radius and height) of a circular right cylinder using the Null Field Method. It is valid for moderate aspect ratios.",
    install_requires=["smuthi>=0.9.1", "numpy", "scipy"],
    license='MIT',
)
