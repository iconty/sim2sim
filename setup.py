from setuptools import setup, find_packages

setup(
    name='legged_mujoco',
    packages=find_packages(),
    install_requires=[
        'mujoco>=3.0.0',
        'pygame',
        'scipy',
        'zmq'
    ],
)
