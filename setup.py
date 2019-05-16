from setuptools import setup, find_packages


setup(name='bmi-live',
      version='0.1',
      author='Mark Piper',
      author_email='mark.piper@colorado.edu',
      license='MIT',
      description='Code, documentation, and Notebooks for the BMI Live clinic',
      long_description=open('README.md').read(),
      install_requires=('numpy', 'pyyaml', 'basic-modeling-interface'),
      packages=find_packages(exclude=['*.tests']),
      package_data={'bmi_live':['data/*']},
)
