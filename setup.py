from setuptools import find_packages, setup
import os

with open('requirements.txt') as f:
    required = f.read().splitlines()

# Check if the DISPLAY environment variable is set and install the corresponding OpenCV version
if 'DISPLAY' in os.environ:
    opencv_pack = "opencv-python"
else:
    opencv_pack = "opencv-python-headless"
required.append(opencv_pack)

setup(
    name='pymarlzooplus',
    version="1.1.0",
    author='AILabDsUnipi',
    author_email='gp.papadopoulos.george@gmail.com',
    description='An extended benchmarking of multi-agent reinforcement learning algorithms',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/AILabDsUnipi/pymarlzooplus',
    packages=find_packages(),
    include_package_data=True,
    license='Apache License 2.0',
    install_requires=required,
    python_requires='>=3.8',
)


