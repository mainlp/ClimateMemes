# **ClimateMemes**

**Code and dataset for the paper:** [What Media Frames Reveal About Stance: A Dataset and Study about Memes in Climate Change Discourse](https://arxiv.org/abs/2505.16592)

This repository provides the **CLIMATEMEMES dataset** and code for evaluating multimodal models on stance and media-frame detection in climate-change memes.


## **Table of Contents**

- [Dataset](#dataset)
    
- [Installation](#installation)
    
- [Code Overview](#code-overview)
    
- [Usage](#usage)

- [Citation](#citation)

- [License](#license)


## **Dataset**

  
The **CLIMATEMEMES dataset** includes 1,184 climate-change memes collected from 47 subreddits (2016–2024). Each meme is annotated with:

- **Stance:** Convinced, Skeptical, or Neither
    
- **Media Frame:** Real, Hoax, Cause, Impact, Allocation, Propriety, Adequacy, Prospect

See [paper](https://arxiv.org/abs/2505.16592) for definitions.


## **Installation**

This project uses **Poetry** for dependency management.

```
git clone https://github.com/mainlp/ClimateMemes.git
cd ClimateMemes
pip install poetry
poetry install
```


## **Code Overview**

The code/ folder includes scripts for running different multimodal models:

### **LLaVA models**

- llava.py — main script for LLaVA experiments
    
- llava_backbone.py — backbone architecture for LLaVA
    
- llava_embedding.py — embedding module for LLaVA

### **Molmo models**

- molmo.py — main script for Molmo experiments
    
- molmo_backbone.py — backbone architecture for Molmo
    
- molmo_embedding.py — embedding module for Molmo
    
These scripts allow evaluation of model performance under different input configurations, including image-only, text-only, and combined inputs.


Ensure that the dataset directory structure is preserved when running the scripts.

## **Citation**

If you use this repository, please cite the following paper:

```bibtex
@misc{zhou2025mediaframesrevealstance,
      title={What Media Frames Reveal About Stance: A Dataset and Study about Memes in Climate Change Discourse}, 
      author={Shijia Zhou and Siyao Peng and Simon Luebke and Jörg Haßler and Mario Haim and Saif M. Mohammad and Barbara Plank},
      year={2025},
      eprint={2505.16592},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2505.16592}, 

```
  
## **License**

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
