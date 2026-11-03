import setuptools

with open('README.md', 'r', encoding='utf-8') as fh:
    long_description = fh.read()

setuptools.setup(
    name='BlurGenerator',
    author='Nat Lee',
    author_email='natlee.work@gmail.com',
    description='Fast generate blur image.',
    keywords='blur, generator, blur-image, cli',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/NatLee/blur-generator',
    project_urls={
        'Documentation': 'https://github.com/NatLee/blur-generator',
        'Bug Reports': 'https://github.com/NatLee/blur-generator/issues',
        'Source Code': 'https://github.com/NatLee/blur-generator',
        # 'Funding': '',
        # 'Say Thanks!': '',
    },
    package_dir={'': 'src'},
    packages=setuptools.find_packages(where='src'),
    classifiers=[
        # see https://pypi.org/classifiers/
        'Development Status :: 5 - Production/Stable',

        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',

        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Programming Language :: Python :: 3.13',
        'Programming Language :: Python :: 3.14',
        'Programming Language :: Python :: 3 :: Only',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.7',
    install_requires=['opencv-python', 'numpy'],
    extras_require={
        'dev': ['check-manifest'],
        # 'test': ['coverage'],
    },
    entry_points={
        'console_scripts': [
            'blurgenerator=blurgenerator.cli:main'
        ]
    }
)
