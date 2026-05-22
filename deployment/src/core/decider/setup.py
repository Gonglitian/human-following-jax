from setuptools import setup, find_packages
import os
from glob import glob

package_name = 'decider'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        # Top-level .pt files (Ours / Meta-NoMap / RL-PC) are always installed.
        (os.path.join('share', package_name, 'model_weight'), glob('model_weight/*.pt')),
        # baselines/ ckpts (CRL guiding × 5, fixed-d pure-RL × 3, Ours alts × 3)
        # — needed at runtime for `decider_crl` / model_weight_file:=baselines/...
        (os.path.join('share', package_name, 'model_weight', 'baselines'),
            glob('model_weight/baselines/*.pt')),
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
            'decider = decider.main:main',
            'decider_orca = decider.main_orca:main',
            'decider_mpc = decider.main_mpc_adc:main',
            'decider_rlpc = decider.main_rlpc:main',
            'decider_crl = decider.main_crl:main',
        ],
    },
)
