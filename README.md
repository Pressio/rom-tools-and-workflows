# rom-tools-and-workflows
The ROM tools and workflows Python library comprises a set of algorithms for constructing and exploiting ROMs that rely on *protocol classes* that encapsulate all the information needed to run a given algorithm. The philosophy is that, for any given application, the user simply needs to "fill out" a class that meets the required API of the protocol. Once this class is complete, the user gains access to all of our existing algorithms.

## Documentation

https://pressio.github.io/rom-tools-and-workflows/

## Demos

https://pressio.github.io/rom-tools-and-workflows/demos/

## Installation

```bash
cd my-path/rom-tools-and-workflows
pip install .
```

Optional: minimal `setup.py` install (drops docs/testing dependencies)

```bash
cd my-path/rom-tools-and-workflows
python setup.py install
```

### Verify installation by running the tests

Note: you need `pytest` installed

```bash
cd my-path/rom-tools-and-workflows
pytest
```
Note: some tests actually generate some auxiliary/temporary files which
are handled via the `tmp_path` as suggested https://docs.pytest.org/en/7.1.x/how-to/tmp_path.html.


## Building the documentation

```
cd <repo-dir>
make -C docs html
```

Open `docs/build/html/index.html` in a browser to view the docs.
