#!/usr/bin/env python

# Copyright 2014-2017 OpenMarket Ltd
# Copyright 2017 Vector Creations Ltd
# Copyright 2017-2018 New Vector Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
from typing import Any, Dict

from setuptools import Command, find_packages, setup

here = os.path.abspath(os.path.dirname(__file__))


# Some notes on `setup.py test`:
#
# Once upon a time we used to try to make `setup.py test` run `tox` to run the
# tests. That's a bad idea for three reasons:
#
# 1: `setup.py test` is supposed to find out whether the tests work in the
#    *current* environmentt, not whatever tox sets up.
# 2: Empirically, trying to install tox during the test run wasn't working ("No
#    module named virtualenv").
# 3: The tox documentation advises against it[1].
#
# Even further back in time, we used to use setuptools_trial [2]. That has its
# own set of issues: for instance, it requires installation of Twisted to build
# an sdist (because the recommended mode of usage is to add it to
# `setup_requires`). That in turn means that in order to successfully run tox
# you have to have the python header files installed for whichever version of
# python tox uses (which is python3 on recent ubuntus, for example).
#
# So, for now at least, we stick with what appears to be the convention among
# Twisted projects, and don't attempt to do anything when someone runs
# `setup.py test`; instead we direct people to run `trial` directly if they
# care.
#
# [1]: http://tox.readthedocs.io/en/2.5.0/example/basic.html#integration-with-setup-py-test-command
# [2]: https://pypi.python.org/pypi/setuptools_trial
class TestCommand(Command):
    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        print(
            """Synapse's tests cannot be run via setup.py. To run them, try:
     PYTHONPATH="." trial tests
"""
        )


def read_file(path_segments):
    """Read a file from the package. Takes a list of strings to join to
    make the path"""
    file_path = os.path.join(here, *path_segments)
    with open(file_path) as f:
        return f.read()


def exec_file(path_segments):
    """Execute a single python file to get the variables defined in it"""
    result: Dict[str, Any] = {}
    code = read_file(path_segments)
    exec(code, result)
    return result


version = exec_file(("synapse", "__init__.py"))["__version__"]
dependencies = exec_file(("synapse", "python_dependencies.py"))
long_description = read_file(("README.rst",))

REQUIREMENTS = dependencies["REQUIREMENTS"]
CONDITIONAL_REQUIREMENTS = dependencies["CONDITIONAL_REQUIREMENTS"]
ALL_OPTIONAL_REQUIREMENTS = dependencies["ALL_OPTIONAL_REQUIREMENTS"]

# Make `pip install matrix-synapse[all]` install all the optional dependencies.
CONDITIONAL_REQUIREMENTS["all"] = list(ALL_OPTIONAL_REQUIREMENTS)

# Developer dependencies should not get included in "all".
#
# We pin black so that our tests don't start failing on new releases.
CONDITIONAL_REQUIREMENTS["lint"] = [
    "isort==5.7.0",
    "black==22.3.0",
    "flake8-comprehensions",
    "flake8-bugbear==21.3.2",
    "flake8",
]

CONDITIONAL_REQUIREMENTS["mypy"] = [
    "mypy==0.931",
    "mypy-zope==0.3.5",
    "types-bleach>=4.1.0",
    "types-jsonschema>=3.2.0",
    "types-opentracing>=2.4.2",
    "types-Pillow>=8.3.4",
    "types-psycopg2>=2.9.9",
    "types-pyOpenSSL>=20.0.7",
    "types-PyYAML>=5.4.10",
    "types-requests>=2.26.0",
    "types-setuptools>=57.4.0",
]

# Dependencies which are exclusively required by unit test code. This is
# NOT a list of all modules that are necessary to run the unit tests.
# Tests assume that all optional dependencies are installed.
#
# parameterized_class decorator was introduced in parameterized 0.7.0
CONDITIONAL_REQUIREMENTS["test"] = ["parameterized>=0.7.0"]

CONDITIONAL_REQUIREMENTS["dev"] = (
    CONDITIONAL_REQUIREMENTS["lint"]
    + CONDITIONAL_REQUIREMENTS["mypy"]
    + CONDITIONAL_REQUIREMENTS["test"]
    + [
        # The following are used by the release script
        "click==8.1.0",
        "redbaron==0.9.2",
        "GitPython==3.1.14",
        "commonmark==0.9.1",
        "pygithub==1.55",
        # The following are executed as commands by the release script.
        "twine",
        "towncrier",
    ]
)

setup(
    name="matrix-synapse",
    version=version,
    packages=find_packages(exclude=["tests", "tests.*"]),
    description="Reference homeserver for the Matrix decentralised comms protocol",
    install_requires=REQUIREMENTS,
    extras_require=CONDITIONAL_REQUIREMENTS,
    include_package_data=True,
    zip_safe=False,
    long_description=long_description,
    long_description_content_type="text/x-rst",
    python_requires="~=3.7",
    entry_points={
        "console_scripts": [
            # Application
            "synapse_homeserver = synapse.app.homeserver:main",
            "synapse_worker = synapse.app.generic_worker:main",
            "synctl = synapse._scripts.synctl:main",
            # Scripts
            "export_signing_key = synapse._scripts.export_signing_key:main",
            "generate_config = synapse._scripts.generate_config:main",
            "generate_log_config = synapse._scripts.generate_log_config:main",
            "generate_signing_key = synapse._scripts.generate_signing_key:main",
            "hash_password = synapse._scripts.hash_password:main",
            "register_new_matrix_user = synapse._scripts.register_new_matrix_user:main",
            "synapse_port_db = synapse._scripts.synapse_port_db:main",
            "synapse_review_recent_signups = synapse._scripts.review_recent_signups:main",
            "update_synapse_database = synapse._scripts.update_synapse_database:main",
        ]
    },
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Topic :: Communications :: Chat",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    cmdclass={"test": TestCommand},
)
