# Acknowledgements

## 致敬 / Credit

This project builds directly on **HM3D-M3**, and we are grateful to its authors for
releasing the benchmark, the 16K train episodes, the IIN goal images, and the
100K observation–assignment teacher samples that make this work possible:

> Hongyu Ding, Kanyu Xu, Mingjia Chen, Zixuan Chen, Jieqi Shi, Jing Huo,
> Yifan Zhang, Yang Gao.
> *HM3D-M3: A Benchmark for Multi-Agent, Multi-Modal, Multi-Object Goal Navigation.*
> arXiv:2610.20121, 2026.

We also want to credit the paper's learned hierarchical goal assigner **HiGA**: the
assigner in this repository (`m3assign/model.py`) is an independent reimplementation
of its upper level (per-agent/per-goal scoring + one-to-one Hungarian decoding),
trained on the released teacher samples. Any weakness here is ours, not the
original authors'.

Additional thanks to the people and projects this code stands on:

- **HM3D / Habitat-Matterport 3D** (Ramakrishnan et al.) and the Matterport team, for the scenes.
- **Habitat-Lab / Habitat-Sim** (Meta AI), for the simulator and the navmesh geodesics used in evaluation.
- **CLIP** (Radford et al., OpenAI), for the goal-similarity heat maps shipped in the teacher samples.
- **OVON** and the HM3D-OVON line of work, for the object-goal vocabulary and task formulation.

## Citation

```bibtex
@misc{ding2026hm3dm3benchmarkmultiagentmultimodal,
  title   = {HM3D-M3: A Benchmark for Multi-Agent, Multi-Modal, Multi-Object Goal Navigation},
  author  = {Hongyu Ding and Kanyu Xu and Mingjia Chen and Zixuan Chen and
             Jieqi Shi and Jing Huo and Yifan Zhang and Yang Gao},
  year    = {2026},
  eprint  = {2610.20121},
  archivePrefix = {arXiv},
  primaryClass  = {cs.RO},
  url     = {https://arxiv.org/abs/2610.20121}
}
```

## For the paper (draft)

> We thank Hongyu Ding, Kanyu Xu, Mingjia Chen, Zixuan Chen, Jieqi Shi, Jing Huo,
> Yifan Zhang and Yang Gao for building and releasing HM3D-M3, its episodes, IIN goal
> images and the 100K observation–assignment teacher samples, on which this work is
> built, and for clearly documenting the HiGA assigner we reimplement. We also thank
> the HM3D and Habitat teams for the scenes and simulator, and the CLIP authors for the
> vision–language model behind the goal heat maps.

## Data and license notes

- HM3D-M3 annotations, episodes and teacher samples are **CC BY-NC 4.0** and additionally
  subject to the **HM3D / Matterport academic EULA** (non-commercial research only).
- This repository does **not** redistribute HM3D scenes, episodes, IIN images or teacher
  shards; the path constants in `m3assign/data.py` must be pointed at your own download
  before running.
- Checkpoints and reports here are derived artifacts and inherit the same non-commercial terms.
