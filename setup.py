from setuptools import setup
import os
from glob import glob

package_name = 'data_collection'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Install json files from the source 'share' directory to the package share directory
        (os.path.join('share', package_name), glob('share/*.json')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='user',
    maintainer_email='user@todo.todo',
    description='Data collection tools including video recording and image extraction',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'data_recorder = data_collection.data_recorder:main',
            'extract_frames = data_collection.extract_frames:main',
        ],
    },
)
