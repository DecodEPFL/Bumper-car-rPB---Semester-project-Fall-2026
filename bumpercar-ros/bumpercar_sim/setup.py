from glob import glob
import os

from setuptools import find_packages, setup

package_name = 'bumpercar_sim'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (
            os.path.join('share', package_name, 'mlp_sim_onnx_models'),
            glob('mlp_sim_onnx_models/*.onnx'),
        ),
        (
            os.path.join('share', package_name, 'mlp_simulation_models'),
            glob('mlp_simulation_models/*.pth'),
        ),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ubuntubirger',
    maintainer_email='birger.morud@gmail.com',
    description='Simulation and visualization nodes for the bumpercar project.',
    license='Apache-2.0',
    extras_require={'test': ['pytest']},
    entry_points={
        'console_scripts': [
            'simulator = bumpercar_sim.simulator_node:main',
            'visualizer = bumpercar_sim.visualization_node:main',
            'arena_simulator = bumpercar_sim.arena_simulator_node:main',
        ],
    },
)
