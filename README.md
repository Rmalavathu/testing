# Exploration of Open World Object Detection (OWOD)
Built off [`PROB: Probabilistic Objectness for Open World Object Detection`](https://github.com/orrzohar/PROB)

# Abstract

In the domain of Open World Object Detection(OWOD), the PROB model represents a significant leap forward by introducing a probabilistic
framework to tackle the intricate task of identifying unknown objects. This paper conducts athorough review of the PROB model, elucidating
its noteworthy contributions, the specific research gap it addresses, and the innovative solution it proposes. Furthermore, the review 
extends to encompass two related papers, delving into the OW-RCNN and ORE models, thereby offering a comprehensive understanding of the 
evolving landscape within OWOD. The implementation section outlines the motivation, plan, and experimental details focused on the PROB model. 
Notably, the exploration emphasizes the use of Euclidean distance as an alternative to Mahalanobis Distance. While the obtained results did 
not yield conclusive findings, they have laid a robust foundation for understanding OWOD and the intricacies of the PROB model.



# Motivation
In my implementation, I am contemplating the use of Euclidean distance as an alternative to Mahalanobis Distance. While Mahalanobis Distance is widely adopted for its ability to account for correlations between different variables in a dataset, I am interested in exploring the impact of Euclidean distance—a metric I employed in my initial foray into basic object detection from scratch.

Euclidean distance is a common and computationally straightforward measure in machine learning. By experimenting with it as an alternative to Mahalanobis Distance, I aim to assess its effectiveness in the context of training to enhance the objectiveness of query embeddings. This exploration will not only provide valuable insights into the relative performance of these distance measures but also contribute to the broader understanding of their applicability in the specific context of the PROB model.

Additionally, I aim to develop a mechanism for visualizing all objects considered by the model and determining whether they are categorized as unknown or identified as specific objects. This visualization will enhance the practical utility of the model, providing insights into its decision-making process and improving its applicability in real-world scenarios.

![prob](./docs/Method.png)

# Installation

### Requirements

We have trained and tested our models on `Ubuntu 16.04`, `CUDA 11.1/11.3`, `GCC 5.4.0`, `Python 3.10.4`

```bash
conda create --name prob python==3.10.4
conda activate prob
pip install -r requirements.txt
pip install torch==1.12.0+cu113 torchvision==0.13.0+cu113 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu113
```

# Setup and Dataset

Use the setup.sh script to get you all setup

```bash
bash ./setup.sh
```

For reproducing any of the aforementioned results, please download our [weights](https://drive.google.com/uc?id=1TbSbpeWxRp1SGcp660n-35sd8F8xVBSq) and place them in the 
'exps' directory.

**note: you may need to give permissions to the .sh files under the 'configs' and 'tools' directories by running `chmod +x *.sh` in each directory.

```
PROB/
└── exps/
    ├── MOWODB/
    |   └── PROB/ (t1.ph - t4.ph)
    └── SOWODB/
        └── PROB/ (t1.ph - t4.ph)
```

Use the PASCAL VOC 2007 and 2012 Dataset, and COCO 2017 Dataset.

# Training

#### Training on single node

To train PROB on a single node with 1 GPUS, run
```bash
bash ./run.sh
```
**note: you may need to give permissions to the .sh files under the 'configs' and 'tools' directories by running `chmod +x *.sh` in each directory.

By editing the run.sh file, you can decide to run each one of the configurations defined in ``\configs``:

1. EVAL_M_OWOD_BENCHMARK.sh - evaluation of tasks 1-4 on the MOWOD Benchmark.
2. EVAL_S_OWOD_BENCHMARK.sh - evaluation of tasks 1-4 on the SOWOD Benchmark. 
3. M_OWOD_BENCHMARK.sh - training for tasks 1-4 on the MOWOD Benchmark.
4. M_OWOD_BENCHMARK_RANDOM_IL.sh - training for tasks 1-4 on the MOWOD Benchmark with random exemplar selection.
5. S_OWOD_BENCHMARK.sh - training for tasks 1-4 on the SOWOD Benchmark.

# Evaluating Model

To run PROB on a single node with 1 GPUS, run
```bash
bash ./run_eval.sh
```

# CODE

All code from PROB github. I changed the run_eval.sh and run.sh scripts to work for 1 gpu setup. On top of that for the euclidean distance, added a function in prob_deformable_detr.py under the models folder. Also changed the engine.py folder to create function to visualize the results and the main_open_world.py to add it to current testing functions and utilize a parameter to just get visualization. 




**Note:**
Please check the [Deformable DETR](https://github.com/fundamentalvision/Deformable-DETR) repository for more training and evaluation details.
Please check the [PROB](https://github.com/orrzohar/PROB) repository for more training and evaluation details.




# Citing

If you use PROB, please consider citing:

```bibtex
@InProceedings{Zohar_2023_CVPR,
    author    = {Zohar, Orr and Wang, Kuan-Chieh and Yeung, Serena},
    title     = {PROB: Probabilistic Objectness for Open World Object Detection},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2023},
    pages     = {11444-11453}
}
```


