# Neural Polar Decoders for DNA Data Storage

This repository contains the code for training and evaluating neural polar decoders (NPDs) on communication channels. 
It includes a code that optimizes the code rate of the polar code by maximizing the mutual information (MI) of the channel's inputs and outputs.

TODO: add link
The code is based on the paper "Neural Polar Decoders for DNA Data Storage" [[1] add link](link).

---

## Setup

For conda:

```bash
  conda create -n npd-env python=3.9 -y
  conda activate npd-env
```
Clone the repository:
```bash
    git clone https://github.com/zivaharoni/neural-polar-decoders-dna-storage.git
    cd  neural-polar-decoders-dna-storage
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Experiments 
---
### Uniform and i.i.d. input distribution

##### Deletion channels

###### CNN embedding
Training and decoding:
```bash
python experiments/train_iid.py --channel deletion --d 0.1 --N 128 --batch 256 --epochs 1000 --steps_per_epoch 5000 --embedding_config_path configs/cnn_small_embedding.json --npd_config_path configs/npd_small_config.json --save_dir_path results/train-iid-deletion-128-cnn
python experiments/decode_iid.py --channel deletion --d 0.1 --N 128 --batch 256 --mc_length_design 1000 --mc_length 1000 --list_num 1 --code_rate 0.329  --embedding_config_path configs/cnn_small_embedding.json --npd_config_path configs/npd_small_config.json --save_dir_path results/train-iid-deletion/30015979 --load_path results/train-iid-deletion/30015979/model/model.weights.h5
```

###### Attention embedding
Training and decoding:
```bash
python experiments/train_iid.py --channel deletion --d 0.1 --N 128 --batch 256 --epochs 1000 --steps_per_epoch 5000 --embedding_config_path configs/attention_small_embedding.json --npd_config_path configs/npd_small_config.json --save_dir_path results/train-iid-deletion-128-cnn
python experiments/decode_iid.py --channel deletion --d 0.1 --N 128 --batch 256 --mc_length_design 1000 --mc_length 1000 --list_num 1 --code_rate 0.329  --embedding_config_path configs/attention_small_embedding.json --npd_config_path configs/npd_small_config.json --save_dir_path results/train-iid-deletion/30015979 --load_path results/train-iid-deletion/30015979/model/model.weights.h5
```

##### IDS channels
###### Attention embedding
Training and decoding:
```bash
python experiments/train_iid.py --channel ids --i 0.01 --d 0.01 --s 0.01 --batch 256 --N 128 --epochs 1000 --steps_per_epoch 5000 --embedding_config_path configs/attention_small_embedding.json --npd_config_path configs/npd_small_config.json --save_dir_path results/train-iid-ids-128
python experiments/decode_iid.py --channel ids --i 0.01 --d 0.01 --s 0.01 --batch 256 --N 128 -mc_length_design 1000 --mc_length 1000 --list_num 1 --code_rate 0.5 --embedding_config_path configs/attention_small_embedding.json --npd_config_path configs/npd_small_config.json --save_dir_path results/train-iid-ids-128 --load_path results/train-iid-ids-128/model/model.weights.h5
```

Decoding:
```bash
python experiments/decode_iid.py --channel ids --i 0.01 --d 0.01 --s 0.01 --batch 256 --N 128 --mc_length_design 1000 --mc_length 1000 --list_num 1 --code_rate 0.5 --embedding_config_path configs/attention_small_embedding.json  --npd_config_path configs/npd_medium_config.json --save_dir_path results/decode-iid-deletion-128-v2 --load_path results/train-iid-deletion-128-v2/model/model.weights.h5
```


## Training an NPD with optimized input distribution

Train and evaluate an NPD on the Ising channel:
```bash
python experiments/optimize_inputs.py --channel deletion --d 0.2 --N 128 --batch 10 --epochs 1000 --steps_per_epoch 5000 --embedding_config_path configs/attention_small_embedding.json --npd_config_path configs/npd_small_config.json --save_dir_path results/optimize-deletion-128-0.2 --verbose 2
```

Model and logs are saved under:

```
results/<save_dir>
```



## Notes

For testing on other channels, you can change the `--channel` argument and implement a new channel class in `src/channels.py`

---

## Citation

TODO: add after Arxiv


---

## License

This project is licensed under the Apache License 2.0. See the [LICENSE](./LICENSE) file for details.

---
## Contact

Ziv Aharoni
Postdoctoral Associate, Duke University
ziv.aharoni at duke.edu
