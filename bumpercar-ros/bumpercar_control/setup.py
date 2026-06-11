from glob import glob
import os

from setuptools import find_packages, setup

package_name = 'bumpercar_control'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (
            os.path.join('share', package_name, 'onnx_models'),
            glob('onnx_models/*.onnx'),
        ),
        (
            os.path.join('share', package_name, 'mlp_control_models'),
            glob('mlp_control_models/*.pth'),
        ),
        (
            os.path.join('share', package_name, 'rPB_control_models'),
            glob('rPB_control_models/*.pt') + glob('rPB_control_models/*.pth'),
        ),
        (
            os.path.join('share', package_name, 'GRU_control_models', 'onnx'),
            glob('GRU_control_models/onnx/*.onnx'),
        ),
        (
            os.path.join('share', package_name, 'GRU_control_models', 'pth'),
            glob('GRU_control_models/pth/*.pth'),
        ),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ubuntubirger',
    maintainer_email='birger.morud@gmail.com',
    description='Controller nodes and learned controller models for the bumpercar project.',
    license='Apache-2.0',
    extras_require={'test': ['pytest']},
    entry_points={
        'console_scripts': [
            'mlp_controller = bumpercar_control.controller_node:main',
            'rpb_controller = bumpercar_control.rPB_controller_node:main',
        ],
    },
)
