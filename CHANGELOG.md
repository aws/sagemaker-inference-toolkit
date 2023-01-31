# Changelog

## v1.8.1 (2023-01-31)

### Bug Fixes and Other Changes

 * Add environment variable VMARGS

## v1.8.0 (2022-10-14)

### Features

 * Write handler to config file and serve model directly from o…

## v1.7.1 (2022-09-08)

### Bug Fixes and Other Changes

 * fix for batch inference

## v1.7.0 (2022-08-22)

### Features

 * pass context to handler functions

## v1.6.1 (2022-05-12)

### Bug Fixes and Other Changes

 * inference handler issue

## v1.6.0 (2022-05-10)

### Features

 * preModel and warmup function support

## v1.5.11 (2022-02-02)

### Bug Fixes and Other Changes

 * Add configurable startup timeout

## v1.5.10 (2022-02-01)

### Bug Fixes and Other Changes

 * Add NOTICE

## v1.5.9 (2022-01-08)

### Bug Fixes and Other Changes

 * log4j migration from 1 to 2. Moving properties file to xml

## v1.5.8 (2021-12-23)

### Bug Fixes and Other Changes

 * Add formatter to logger with a timestamp.

## v1.5.7 (2021-12-10)

### Bug Fixes and Other Changes

 * Re-enable output capturing

## v1.5.6 (2021-12-09)

### Bug Fixes and Other Changes

 * Increase timeout for starting model server to 10 minutes
 * Fixing issue #82

## v1.5.5 (2021-01-30)

### Bug Fixes and Other Changes

 * remove conflicting dependencies

## v1.5.4 (2021-01-18)

### Bug Fixes and Other Changes

 * add SIGCHILD Handler for MMS
 * add model-archiver and multi-model-server to required packages

## v1.5.3 (2020-10-15)

### Bug Fixes and Other Changes

 * upgrade MMS version and update command

### Documentation Changes

 * update link in README.md

## v1.5.2 (2020-08-04)

### Bug Fixes and Other Changes

 * sanitize only the response phrase

## v1.5.1 (2020-07-31)

### Bug Fixes and Other Changes

 * remove prohibited characters from error response

## v1.5.0 (2020-07-30)

### Features

 * Support multiple accept types

## v1.4.0 (2020-07-27)

### Features

 * Decode application/x-npz content type

## v1.3.2.post1 (2020-06-29)

### Testing and Release Infrastructure

 * clarify feature request issue template

## v1.3.2.post0 (2020-06-16)

### Documentation Changes

 * fix package name

## v1.3.2 (2020-05-25)

### Bug Fixes and Other Changes

 * include stacktrace when handling an error

## v1.3.1 (2020-05-11)

### Bug Fixes and Other Changes

 * Remove typing

## v1.3.0 (2020-05-07)

### Features

 * Add Python 3.7 support

### Testing and Release Infrastructure

 * Do not parallelize unit tests in buildspecs

## v1.2.2 (2020-04-01)

### Bug Fixes and Other Changes

 * add model_dir to python path at service initialization

## v1.2.1 (2020-03-23)

### Bug Fixes and Other Changes

 * Remove duplicated call to validate_and_initialize when handling requests

## v1.2.0 (2020-03-04)

### Features

 * MME support

## v1.1.5.post1 (2020-02-17)

### Documentation Changes

 * convert README to Markdown and add emoji

## v1.1.5.post0 (2020-02-06)

### Documentation Changes

 * remove labels from issue templates

## v1.1.5 (2020-01-31)

### Bug Fixes and Other Changes

 * add DLC local integration tests

### Documentation Changes

 * add missing docstrings and update README

### Testing and Release Infrastructure

 * fix buildspec-release.yml

## v1.1.4.post1 (2020-01-16)

### Documentation changes

 * Update link for questions in issue template config and minor issue template edits
 * Edit developer forum link for consistency

## v1.1.4.post0 (2020-01-14)

### Documentation changes

 * Update CONTRIBUTING.md
 * Add issue templates
 * Add pull request template and issue template config

## v1.1.4 (2020-01-13)

### Bug fixes and other changes

 * update linting
 * Add linting with Pylint
 * Add code formatting with Black

## v1.1.3 (2020-01-08)

### Bug fixes and other changes

 * update copyright year in license header

## v1.1.2 (2019-11-14)

### Bug fixes and other changes

 * management and inference address use same port by default

## v1.1.1 (2019-11-06)

### Bug fixes and other changes

 * mme support and local integration test

## v1.1.0 (2019-10-23)

### Features

 * support requirements.txt

### Bug fixes and other changes

 * combine import statements
 * Error handling 20191015
 * wait for mms server process to finish instead of tailing dev null

## v1.0.4 (2019-10-07)

### Bug fixes and other changes

 * Update README.rst

## v1.0.3 (2019-09-06)

### Bug fixes and other changes

 * discard build output artifact path

## v1.0.2 (2019-09-06)

### Bug fixes and other changes

 * enable release publish
 * enable sagemaker-inference-toolkit codepipeline

## v1.0.1

* fix: update logic to handle changes in [MMS v1.0.5](https://github.com/awslabs/mxnet-model-server/releases/tag/v1.0.5)

## v1.0.0

* Initial commit
