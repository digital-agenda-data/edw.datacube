""" EEA Sparql Installer
"""
import os
from setuptools import setup, find_packages

NAME = 'edw.datacube'
PATH = NAME.split('.') + ['version.txt']
VERSION = open(os.path.join(*PATH)).read().strip()

setup(name=NAME,
      version=VERSION,
      description="Daviz visualization data source",
      long_description=open("README.rst").read() + "\n" +
                       open(os.path.join("docs", "HISTORY.txt")).read(),
      classifiers=[
          "Framework :: Zope2",
          "Framework :: Zope3",
          "Framework :: Plone",
          "Framework :: Plone :: 4.0",
          "Framework :: Plone :: 4.1",
          "Framework :: Plone :: 4.2",
          "Programming Language :: Zope",
          "Programming Language :: Python",
          "Topic :: Software Development :: Libraries :: Python Modules",
          "License :: OSI Approved :: GNU General Public License (GPL)",
          "License :: OSI Approved :: Mozilla Public License 1.0 (MPL)",
        ],
      keywords='edw eea daviz visualization zope plone cube sparql',
      author='Eau de Web',
      author_email='contact@eaudeweb.ro',
      maintainer='David Batranu (Eau de Web)',
      maintainer_email='david.batranu@eaudeweb.ro',
      download_url="http://pypi.python.org/pypi/edw.datacube",
      url='http://github.com/eaudeweb/edw.datacube',
      license='GPL',
      packages=find_packages(exclude=['ez_setup']),
      namespace_packages=['edw'],
      include_package_data=True,
      zip_safe=False,
      install_requires=[
          'setuptools',
          'xlutils',
          'Products.Ploneboard',
          'openpyxl',
          'collective.recaptcha',
          'Products.PloneboardNotify',
      ],
      extras_require={
          'test': [
              'plone.app.testing',
          ]
      },
      entry_points="""
      # -*- Entry points: -*-
      """,
      )
