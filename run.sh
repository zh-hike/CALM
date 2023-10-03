
mkdir ./outputs 2>/dev/null

# export PYTHONBREAKPOINT="pudb.set_trace"

export PYTHONBREAKPOINT="0"
dataset=scene15

echo train $dataset
# python -u test/CaETest.py
# python -m pudb train.py --dataset ${dataset}
python  train.py --dataset ${dataset}

