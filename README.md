# All-GCL: A large-scale dataset of functional mouse ganglion cell layer responses
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.20305942-blue)](https://doi.org/10.5281/zenodo.20305942)
[![Dataset](https://img.shields.io/badge/🤗%20Dataset-Hugging%20Face-yellow)](https://huggingface.co/datasets/eulerlab/all-gcl)
[![GCL Classifier](https://img.shields.io/badge/GitHub-GCL%20Classifier-black?logo=github)](https://github.com/eulerlab/gcl_classifier)
[![djimaging](https://img.shields.io/badge/GitHub-djimaging-black?logo=github)](https://github.com/eulerlab/djimaging/tree/all-gcl-v0.1.0)
[![QDSpy Stimuli](https://img.shields.io/badge/Stimuli-QDSpy-blue)](https://github.com/eulerlab/QDSpy-stimuli-Documentation)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![uv](https://img.shields.io/badge/package%20manager-uv-6A5ACD)](https://docs.astral.sh/uv/)

This repository accompanies the manuscript **"All-GCL: A large-scale dataset of functional mouse ganglion cell layer responses"** and provides code for reproducing the main figures, loading the dataset, and exploring the data through tutorials. 
All-GCL contains functional recordings from more than 80,000 mouse ganglion cell layer neurons, together with standardized metadata, pretrained classifiers, tutorials, and analysis code for reproducible neuroscience.

**Preprint:** [bioRxiv](https://www.biorxiv.org/content/10.64898/2025.12.04.691221v1)

## Contents
- Usage
- Data and Related Resources
- Authors and Acknowledgements
- Citation
- License

## Usage

1. **Download the data** from [Hugging Face](https://huggingface.co/datasets/eulerlab/all-gcl).

2. **Clone this repository:**

   ```bash
   git clone https://github.com/eulerlab/all-GCL-manuscript.git
   cd all-GCL-manuscript
   ```

3. **Install the package** using [uv](https://docs.astral.sh/uv/):

   ```bash
   uv sync
   ```

4. **Configure the dataset path** by updating `dataset_dir` in [`config.yaml`](config.yaml) to point to the folder containing the downloaded data.

5. **Tutorial**: Open the tutorial notebook [tutorial notebook](notebooks/tutorials/plot_traces_and_triggers.ipynb) or figures notebooks like [Fig2_dataset_overview](notebooks\Fig2_dataset_overview.ipynb) in jupyter for example via uv:
   ```bash
   uv run --with jupyter jupyter lab
   ```
   
## Data and Related Resources

- **Dataset (NWB format):**  
  Publicly available at Hugging Face:  
  https://huggingface.co/datasets/eulerlab/all-gcl  

- **Stimulus documentation:**  
  Detailed descriptions and implementation of visual stimuli (QDSpy):  
  https://github.com/eulerlab/QDSpy-stimuli-Documentation  

- **GCL classifier:**  
  Code and pretrained models for functional cell-type classification:  
  https://github.com/eulerlab/gcl_classifier

- **djimaging:**  
  DataJoint schema and tables used to generate this dataset originally (see [djimaging/README.md](djimaging/README.md) for details):  
  https://github.com/eulerlab/djimaging/releases/tag/all-gcl-v0.1.0


## Authors and Acknowledgements
Dominic Gonschorek<sup>#,1,2</sup>, Jonathan Oesterle<sup>#,1-3</sup>, Thomas Zenkel<sup>#,1,2</sup>, Federico D'Agostino<sup>#,2,4</sup>, Katrin Franke<sup>1,5-7</sup>, Ryan Arlinghaus<sup>1,2</sup>, Chenchen Cai<sup>1,2</sup>, Florentyna Deja<sup>1,2</sup>, Nadine Dyszkant<sup>1,2</sup>, Tom Schwerd-Kleine<sup>1,2</sup>, Klaudia Szatko<sup>1,2</sup>, Timm Schubert<sup>1,2</sup>, Philipp Berens<sup>1-4</sup>, Thomas Euler<sup>1,2,+</sup>

<sup>#</sup>These authors contributed equally

<sup>1</sup>Institute for Ophthalmic Research, University of Tübingen, Tübingen, Germany

<sup>2</sup>Werner Reichardt Centre for Integrative Neuroscience, University of Tübingen, Tübingen, Germany

<sup>3</sup>Hertie Institute for Artificial Intelligence in Brain Health, University of Tübingen, Tübingen, Germany

<sup>4</sup>Tübingen AI Center, University of Tübingen, Germany

<sup>5</sup>Department of Ophthalmology, Byers Eye Institute, Stanford University School of Medicine, Stanford, CA, USA

<sup>6</sup>Stanford Bio-X and Wu Tsai Neurosciences Institute, Stanford University, Stanford, CA, USA

<sup>7</sup>Wu Tsai Neurosciences Institute, Stanford University, Stanford, CA, USA

Correspondence:
thomas.euler@cin.uni-tuebingen.de

## Citation

If you use this dataset, please cite:

Gonschorek et al. (2025) "A large-scale dataset of functional mouse ganglion cell layer responses" bioRxiv  
https://www.biorxiv.org/content/10.64898/2025.12.04.691221v1

If you use data originating from one or more published studies included in this dataset, please also cite the corresponding original publications:

- Szatko, Klaudia P., et al. "Neural circuits in the mouse retina support color vision in the upper visual field." Nature communications 11.1 (2020): 3481. 

- Höfling, Larissa, et al. "A chromatic feature detector in the retina signals visual context changes." Elife 13 (2024): e86860.

- Gonschorek, Dominic, et al. "Nitric oxide modulates contrast suppression in a subset of mouse retinal ganglion cells." Elife 13 (2025): RP98742.

- Dyszkant, Nadine, et al. "Photoreceptor degeneration has heterogeneous effects on functional retinal ganglion cell types." The Journal of Physiology 603.21 (2025): 6599-6621.

## License

This repository is licensed under the MIT License.
The dataset is distributed under CC-BY-NC-ND 4.0 International license.
