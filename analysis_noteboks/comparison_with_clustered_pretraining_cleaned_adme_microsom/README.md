Code to run comparision with MolBERT and Molformer.

This requires the additional installation of MolBERT.

1. Download the model files from [here](https://ndownloader.figshare.com/files/25611290)
    ```bash
    wget https://ndownloader.figshare.com/files/25611290
    ```
2. Next extract the files and move the checkpoint to this folder.
    ```bash
    mv 25611290 25611290.zip && unzip 25611290.zip
    mkdir checkpoints
    mv molbert_100epochs/checkpoints/last.ckpt ./checkpoints/molbert.ckpt
    mv molbert_100epochs/hparams.yaml hparams.yaml
    rm -r __MACOSX/ molbert_100epochs/ && rm 25611290.zip
    ```
3. Update the model checkpoint to newer pytorch lightning versions. This requires an NVIDIA GPU:
    ```bash
    python -m pytorch_lightning.utilities.upgrade_checkpoint ./molbert.ckpt
    ```
4. Install the MolBERT package
    ```
    pip install -r requirements.txt
    ```