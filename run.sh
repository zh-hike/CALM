
mkdir ./outputs 2>/dev/null

# export PYTHONBREAKPOINT="pudb.set_trace"

export PYTHONBREAKPOINT="0"

# 数据集，可选 handwritten, scene15, landUse21
dataset=scene15

echo train $dataset

# python -m pudb train.py --dataset ${dataset}
python train.py --dataset ${dataset}

