from setuptools import find_packages, setup


with open('README.md') as file:
    long_description = file.read()


setup(
    name='penn',
    description='Pitch Estimating Neural Networks (PENN)',
    version='1.0.0',
    author='Max Morrison, Caedon Hsieh, Nathan Pruyne, and Bryan Pardo',
    author_email='interactiveaudiolab@gmail.com',
    url='https://github.com/interactiveaudiolab/penn',
    python_requires='>=3.12',
    extras_require={
        'train': [
            'librosa>=1.0',
            'matplotlib>=3.11',
            'pyworld>=0.3.6',
            'scipy>=1.18',
            'torchcrepe>=0.0.24'
        ],
        'test': [
            'librosa>=1.0',
            'pytest>=9.2'
        ],
        'triton': [
            'triton>=3.9'
        ]
    },
    install_requires=[
        'huggingface_hub>=2.4',
        'numpy>=2.5',
        'torch>=2.15',
        'torchcodec>=0.17',  # Requires FFmpeg
        'torchutil>=0.1.2',
        'yapecs>=0.5'
    ],
    packages=find_packages(),
    package_data={'penn': ['assets/*', 'assets/*/*']},
    long_description=long_description,
    long_description_content_type='text/markdown',
    keywords=['audio', 'frequency', 'music', 'periodicity', 'pitch', 'speech'],
    classifiers=['License :: OSI Approved :: MIT License'],
    license='MIT')
