from setuptools import setup, find_packages
import os
from glob import glob

package_name = 'target_tracker'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools', 'numpy', 'torch', 'torchvision', 'opencv-python'],
    zip_safe=True,
    maintainer='jyao97',
    maintainer_email='jyao073@ucr.edu',
    description='Camera-based target person tracking with Re-ID for robot person following.',
    license='BSD-3-Clause',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'target_tracker = target_tracker.main:main',
        ],
    },
)
