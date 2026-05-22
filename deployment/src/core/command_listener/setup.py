from setuptools import setup, find_packages
import os
from glob import glob

package_name = 'command_listener'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    # If your source code is under 'src/', uncomment the next line
    # package_dir={'': 'src'},
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        # (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        # (os.path.join('share', package_name, 'model_weight'), glob('model_weight/*')),
    ],
    install_requires=['setuptools', 'numpy'],
    zip_safe=True,
    maintainer='jyao97',
    maintainer_email='jyao073@ucr.edu',
    description='Package for detecting pedestrians using DROW3 or DR-SPAAM in ROS 2.',
    license='BSD-3-Clause',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'command_listener = command_listener.main:main',
        ],
    },
)
