pip uninstall timm detectron2 transformers
pip install transformers
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git@v0.5'

python inference.py  --device cuda --precision float16 --jit --nv_fuser --profile
