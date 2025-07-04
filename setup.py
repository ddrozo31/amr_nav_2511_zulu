from setuptools import find_packages, setup

package_name = 'amr_nav_2511_zulu'

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
    maintainer='davidrozoosorio',
    maintainer_email='david.rozo31@eia.edu.co',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'amr_nav_2511_zulu_astar = amr_nav_2511_zulu.amr_nav_2511_zulu_astar:main',
            'amr_nav_2511_zulu_ppa= amr_nav_2511_zulu.amr_nav_2511_zulu_ppa:main',
            'amr_nav_2511_zulu_astar_bspline = amr_nav_2511_zulu.amr_nav_2511_zulu_astar_bspline:main',
        ],
    },
)
