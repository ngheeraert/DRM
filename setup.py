from setuptools import setup

def readme():
    with open('README.rst') as f:
        return f.read()

setup(name='medrm',
      version='0.1',
      description='This program simulates the driven rabi model using master equation integration.',
      long_description=readme(),
      author='Nicolas Gheeraert',
      author_email='n.gheeraert@physics.iitm.ac.in',
      license='',
      packages=['medrm'],
      install_requires=[],
      include_package_data=True,
      zip_safe=False,
      test_suite='nose.collector',
      tests_require=['nose'],)
