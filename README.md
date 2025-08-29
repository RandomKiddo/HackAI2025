# HackAI 2025 | ISS AI Docking Locator and Distance Regressor

![GitHub License](https://img.shields.io/github/license/RandomKiddo/HackAI2025)
![GitHub repo size](https://img.shields.io/github/repo-size/RandomKiddo/HackAI2025)

:heavy_exclamation_mark: ***THIS REPO MAY HAVE BEEN CHANGED OR UPDATED SINCE THE ORIGINAL HACK AI VERSION*** :heavy_exclamation_mark: <br>
:heavy_exclamation_mark: ***SEE [THE HACK AI RELEASE](https://github.com/RandomKiddo/HackAI2025/releases/tag/hackai2025) FOR THAT VERSION*** :heavy_exclamation_mark: <br>
:warning: ***SIGNIFICANT CHANGES AND IMPROVEMENTS HAVE BEEN MADE TO THE MODEL, BUT A LOT OF IT HAS BEEN LEFT UNPUBLISHED, AS IT IS PART OF AN UNDER-DEVELOPMENT SYSTEM*** :warning:

___

### Team

<u>LeTeam:</u> Neil Ghugare, Pranav Moola, Nishanth Kunchala, and Jacob Balek

___

### Acknowledgements

The data was collected on Kaggle, taken from a an AICrowd challenge dataset, located [here](https://www.kaggle.com/datasets/msafi04/iss-docking-dataset/data).

We used public-use STL files of the ISS and Dragon shuttle, located [here](https://www.thingiverse.com/thing:3570393#google_vignette) and [here](https://www.thingiverse.com/thing:4207259), respectively.

___

### HackAI 2025 Model

The HackAI 2025 model was created as a part of the 25-hour hackathon. We created a MHN regressing three values (distance, x-coordinate of the docking port, and y-coordinate of the docking port) through three separate heads. The backbone of the model was the `MobileNetV3Small` model provided by Keras. 

<img src="imgs/loss5.png" alt="Loss and Val Loss of the MHN HackAI Model" width="800"/>

<img src="imgs/model.png" alt="Top Layer of the MHN (MobileNet not included)" width="800"/>

Alongside that, a 3D visualization of the model outputs was created using STL files of the ISS and SpaceX Dragon capsule (provided). 

<p align="center">
  <img src="imgs/test.gif" alt="Test Gif of HackAI Model"/>
</p>

___

[Back to Top](#hackai-2025--iss-ai-docking-locator-and-distance-regressor)

<sub>This page was last edited on 08.25.2025</sub>
