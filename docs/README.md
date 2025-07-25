
# Docs

## View documentation

```sh
docs/serve.sh
```

will serve the documentation files on [localhost:8080](http://localhost:8080). It will also build the documentation automatically if no html files are found (you will need to manually rebuild on changes).

## Build

```sh
docs/build.sh
```

will build the documentation.

## Requirements

Requirements needed to build the docs are available in `requirements-dev.txt`, install them with `pip install -r requirements.txt`

**PSA**: The `pandoc` library in python is only a wrapper for the program. You need to manually install the program on your computer:

- Linux: `sudo apt-get install pandoc`
- Mac: `brew install pandoc`
- Windows:
  - `choco install pandoc`
  - `winget install --source winget --exact --id JohnMacFarlane.Pandoc`
  - or use a windows installer (have fun :))
