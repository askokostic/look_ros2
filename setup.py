from setuptools import find_packages, setup
import os

package_name = 'look_ros2'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/looking/models/predictor', ['looking/models/predictor/' + f for f in os.listdir('looking/models/predictor')]),
        ('share/' + package_name + '/looking/utils', ['looking/utils/' + f for f in os.listdir('looking/utils')])
    ],
    install_requires=[
        'setuptools',
        'opencv-python',
        'numpy<2',
        'openpifpaf==0.13.10',
        'Pillow',
    ],
    zip_safe=True,
    maintainer='Aleksa Kostic',
    maintainer_email='aleksa.kostic.fl@ait.ac.at',
    description='ROS2 wrapper for LOOK - a gaze detector',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'predictor = look_ros2.predictor:main'
        ],
    },
)
