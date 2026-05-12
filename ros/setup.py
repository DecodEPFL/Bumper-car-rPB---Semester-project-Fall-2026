from setuptools import find_packages, setup

package_name = 'my_package'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ubuntubirger',
    maintainer_email='birger.morud@gmail.com',
    description='Beginner client libraries tutorials practice package',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'controller = my_package.controller_node:main',
            'simulator = my_package.simulator_node:main',
            'visualizer = my_package.visualization_node:main'

        ],
    },
)
