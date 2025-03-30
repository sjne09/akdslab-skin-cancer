# Improved Diagnosis of Non-Melanoma Skin Cancer in Resource-Limited Settings

Spencer Ellis*, Steven Song*, Derek Reiman*, Xuan Hui, Renyu Zhang, Mohammad H. Shahriar, Maria Argos, John A. Baron, Mohammed Kamal, Christopher R. Shea, Robert L. Grossman, Aly A. Khan, and Habibul Ahsan (*Equal Contribution)

## Setup

In order to utilize most of the code in this repo, you must have been approved for access to the UNI, Prov-GigaPath, PRISM, and Virchow V1 models on Huggingface:
- UNI: https://huggingface.co/MahmoodLab/UNI
- Prov-GigaPath: https://huggingface.co/prov-gigapath/prov-gigapath
- PRISM: https://huggingface.co/paige-ai/Prism
- Virchow V1: https://huggingface.co/paige-ai/Virchow

### Setting Up Your Local Environment
1. Clone the repo:
    ```
    git clone https://github.com/sjne09/akdslab-skin-cancer.git
    ```

2. Set up a conda environment and install all dependencies:
    ```
    conda env create -f environment.yaml
    conda activate nmsc
    ```

3. Install [OpenSlide](https://openslide.org/)

4. Clone the [Prov-GigaPath repo](https://github.com/prov-gigapath/prov-gigapath) and install it as a package:
    ```
    git clone https://github.com/prov-gigapath/prov-gigapath
    cd prov-gigapath
    pip install -e .
    ```

5. Update `.env` with data and output directory paths

## Running

To rerun our experiments, see the scripts in `scripts.pipeline`.
