# totagarpad
MIHVG.hu 
Címsor megadásával cikket generál egy adott szerző stílusában és képes átírni a meglévő cikkeket adott sítlusra.
Jelenleg Tóta Gép Árpad elérhető :)

Dataset
```
https://huggingface.co/datasets/L4IO/tota_gep_arpad
```

Model
```
https://huggingface.co/NYTK/PULI-GPT-3SX
```

Engine
```
https://studiolab.sagemaker.aws/
```

Set-up
```
git clone https://github.com/huggingface/transformers
cd transformers
pip install .
pip install datasets
pip install evaluate
pip install torch
pip install sklearn
pip install scikit-learn
```

Train
```
python examples/pytorch/language-modeling/run_clm.py --model_name_or_path NYTK/PULI-GPT-2 --dataset_name tota_gep_arpad --do_train --do_eval --output_dir totageparpad --per_device_train_batch_size 1 --per_device_eval_batch_size 2 --block_size 512
```

Run
```
gen.py
```
```
from transformers import pipeline
textgen_totagepartpad = pipeline("text-generation", "./totageparpad")
print(textgen_totagepartpad(" Szépségük lett a vesztük a Trónok harca sorozatból elhíresült állatoknak ", min_length=50, max_length=400)[0]['generated_text'])
```
