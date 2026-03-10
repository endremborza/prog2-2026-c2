.SILENT:
setup:
	uv pip install pandas numpy pykdtree

preproc:
	uv run python preproc.py

compute:
	uv run python src.py

cleanup:
	rm -f clean.npy
