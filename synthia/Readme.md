# (1) Download the SYNTHIA-Dataset 
$ wget --no-check-certificate http://synthia-dataset.cvc.uab.cat/SYNTHIA_RAND_CITYSCAPES.rar

# (2) For Ubuntu, maybe the rar package is missiong
$sudo apt-get install rar

# (3) unrar the package
$ unrar x SYNTHIA_RAND_CITYSCAPES.rar

# (4) the following directory structure is necessary

synthia
    - Depth 
        - Depth 
    - GT
        - COLOR
        - LABELS
    - RGB
