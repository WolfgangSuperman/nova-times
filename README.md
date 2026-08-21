# Nova Times

Measuring Nova decay times.


# Installation

## Requirements

You'll need python >= 3.11

We recommend using a version manager (e.g. [pyenv](https://github.com/pyenv/pyenv)) if you
aren't already using an environment manager (e.g. [Conda](https://docs.conda.io/projects/conda/en/stable/)),

## Simple CLI Installation via `pipx`

### Requirements

> pipx is a tool to help you install and run end-user applications written in Python. It's roughly similar to macOS's brew, JavaScript's npx, and Linux's apt.

Install `pipx` for your platform using the [instructions here](https://pipx.pypa.io/stable/).

### Installation

To install the most current development version of this CLI application:

```
pipx install git+https://github.com/WolfgangSuperman/nova-times.git
```

If you want to force a reinstallation to bring in updates before a new version is published:

```
pipx install --force git+https://github.com/WolfgangSuperman/nova-times.git
```

Test that the installation was successful with this:
```
nova-times --help
```

You should see the help text for the `nova-times` CLI.


## Development via `poetry`

This project was developed using [poetry](https://python-poetry.org/) follow their instructions to install `poetry` on your system.

### Quickstart

Clone this repo:

```
git clone https://github.com/WolfgangSuperman/nova-times.git
cd nova-times
```

Install the project:

```
poetry install --with test
```

Run tests:

```
poetry run pytest
```

Run the CLI:

```
poetry run nova-times --help
```

