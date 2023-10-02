
mkdir ./outputs 2>/dev/null

export PYTHONBREAKPOINT="pudb.set_trace"

python -u test/CaETest.py
