from glob import glob
import os

from setuptools import find_packages, setup

package_name = 'following_sim'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        (os.path.join('share', package_name), ['package.xml']),
        (os.path.join('share', package_name, 'urdf'),
         glob('urdf/*.xacro')),
        (os.path.join('share', package_name, 'meshes'),
         [p for p in glob('meshes/*') if os.path.isfile(p)]),
        (os.path.join('share', package_name, 'meshes', 'x3'),
         glob('meshes/x3/*')),
        (os.path.join('share', package_name, 'worlds'),
         glob('worlds/*.world')),
        (os.path.join('share', package_name, 'config'),
         glob('config/*.yaml')),
        (os.path.join('share', package_name, 'launch'),
         glob('launch/*.py')),
        (os.path.join('share', package_name, 'rviz'),
         glob('rviz/*.rviz')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='glt',
    maintainer_email='gonglitian2002@gmail.com',
    description='Gazebo + HuNavSim evaluation harness for the human-following policy.',
    license='BSD-3-Clause',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'target_to_uwb_bridge = following_sim.target_to_uwb_bridge:main',
            'metrics_recorder = following_sim.metrics_recorder:main',
            'human_states_viz = following_sim.human_states_viz:main',
            'detections_merger = following_sim.detections_merger:main',
            'tf_health_monitor = following_sim.tf_health_monitor:main',
            'cmd_vel_watchdog = following_sim.cmd_vel_watchdog:main',
        ],
    },
)
