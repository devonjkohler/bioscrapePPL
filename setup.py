from distutils.core import setup

setup(
    name = 'bioscrapePPL',
    version = '0.0.1',
    author='Anandh Swaminathan, William Poole, Ayush Pandey, Devon Kohler',
    url='https://github.com/biocircuits/bioscrape/',
    description='Biological Stochastic Simulation of Single Cell Reactions and Parameter Estimation.',
    packages = ['bioscrapePPL'],
    zip_safe=False,
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python :: 3.6',
        'Topic :: Software Development',
        'Topic :: Scientific/Engineering',
        'Operating System :: OS Independent',
    ],
    setup_requires = [
        "numpy",
        "cython",
        ],
    install_requires=[
        "numpy",
        "matplotlib",
        "pytest",
        "scipy",
        "cython",
        "python-libsbml",
        "beautifulsoup4",
        "sympy",
        "emcee",
        "pandas",
        "pyprob"
    ],
    python_requires='>=3.6',
    keywords="SBML synthetic biology modeling Chemical Reaction Network CRN simulator stochastic parameter inference",
    tests_require=["pytest"],
    project_urls={
    'Documentation': 'https://readthedocs.org/projects/biocrnpyler/',
    'Funding': 'http://www.cds.caltech.edu/~murray/wiki/DARPA_BioCon',
    'Source': 'https://github.com/biocircuits/bioscrape/',
    'Tracker': 'https://github.com/biocircuits/bioscrape/issues',
    }
)
