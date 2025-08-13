from setuptools import setup, find_packages

setup(
    name="little_learner",
    version="0.1.0",
    description="A package for curriculum learning and neural module training",
    author="Samuel Lozano Iglesias",
    author_email="samuel.lozano@ucm.es",
    packages=find_packages(),
    install_requires=[
        "jax",
        "flax",
        "optax",
        "pandas",
        "scikit-learn"
    ],
    python_requires=">=3.8",
    include_package_data=True,
    zip_safe=False,
)
