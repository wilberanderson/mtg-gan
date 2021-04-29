### MTG cGAN
##### Running
1. Download most recent [unique artwork](https://scryfall.com/docs/api/bulk-data) file and point `crop-harvester.py` at it.
2. Run these in order:
	a. `crop-harvester.py`
	b. `dataset_builder.py`
	c. `cgan.py`
3. The cGAN options can be set via command line or by modifying the file.

Sources:
https://www.kaggle.com/arturlacerda/pytorch-conditional-gan
https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/cgan/cgan.py
