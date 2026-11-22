"""``python -m findex`` runs the same CLI as the ``findex`` command."""

from findex.cli import app

# The guard matters: under the "spawn" start method each worker process imports
# the parent's main module again (as __mp_main__). Without it, every worker of
# `python -m findex index --executor processes` would start the CLI over again.
if __name__ == "__main__":
    app(prog_name="findex")
