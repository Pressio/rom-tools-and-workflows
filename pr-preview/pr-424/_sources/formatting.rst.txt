Formatting Utility
==================

The formatting utility replaces placeholders of the form ``{RTVAR:name}`` in
text files with values provided from a ``params.in`` file and/or command line
overrides.

Command line usage
------------------

Single file with params file and explicit output:

.. code-block:: bash

   romtools-format --params params.in --i input.txt --o output.txt

If ``params.in`` exists in the current directory, it is loaded automatically.
You can also pass ``--params`` explicitly. Command line ``--var`` values
override any values found in the params file.

Multiple files:

.. code-block:: bash

   romtools-format a.txt b.txt --params params.in --output-dir out

Optional overrides:

.. code-block:: bash

   romtools-format --params params.in --i input.txt --o output.txt --var alpha=2.0

``params.in`` format
--------------------

One ``NAME=VALUE`` pair per line. Blank lines and lines starting with ``#`` are
ignored.

.. code-block:: text

   # comment
   alpha=1.5
   beta=two

Python API usage
----------------

.. code-block:: python

   from romtools.workflows.formatting import format_text, format_file, format_files

   format_text("alpha={RTVAR:alpha}", {"alpha": 1.5})
   format_file("input.txt", {"alpha": 1.5}, output_path="output.txt")
   format_files(["a.txt", "b.txt"], {"x": 7}, output_dir="out")
