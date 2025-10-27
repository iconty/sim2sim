from setuptools import setup, find_packages

setup(
    name='legged_mujoco',
    packages=find_packages(),
    install_requires=[
        'mujoco',
        'pygame',
        'scipy',
        'zmq'
    ],
)
