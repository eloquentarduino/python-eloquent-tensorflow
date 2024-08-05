from distutils.core import setup


setup(
  name='eloquent_tensorflow',
  packages=['eloquent_tensorflow'],
  version='1.0.2',
  license='MIT',
  description='A utility to convert TensorFlow models to Arduino code',
  author='Simone Salerno',
  author_email='support@eloquentarduino.com',
  url='https://github.com/eloquentarduino/python-eloquent-tensorflow',
  keywords=[
    'ML',
    'Edge AI'
  ],
  install_requires=[
    'numpy',
    'tensorflow',
    'hexdump',
    'Jinja2'
  ]
)