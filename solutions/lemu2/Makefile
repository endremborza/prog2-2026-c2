setup:
	uv pip install numpy

preproc:
	python preproc.py

compute:
	python src.py

cleanup:
	-del /f genre_buckets.pkl
# preventing data contamination