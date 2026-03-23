.SILENT:

setup:
	uv pip install flask pandas numpy pykdtree

preproc:
	uv run python preproc.py > preproc.log 2>&1 &
	sleep 1

compute:
	curl --retry 10 --retry-connrefused --retry-delay 0 -s http://127.0.0.1:5678/ping

cleanup:
	-curl -s http://127.0.0.1:5678/shutdown >/dev/null 2>&1