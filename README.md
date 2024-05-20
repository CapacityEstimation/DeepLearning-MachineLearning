All content in this repository is published under a CC-BY-NC-SA license. You are free to: 

- Share — copy and redistribute the material in any medium or format
- Adapt — remix, transform, and build upon the material

When you want to run the models with the designed dataset. First download the dataset via: https://1drv.ms/f/s!AnE8BfHe3IOlg13v2ltV0eP1-AgP?e=9o4zgL and follow the instructions provided to load and store the dataset. After preparing the dataset you can replace the main.py and model.py files in the capacity_estimation folder from the original research. To run the desired model you can select the model and certain parameters in the terminal, for example running the XGBoost model with 750 boosting rounds:  main.py --model XGBoost --num_epochs 750 --load_saved_dataset. The dataset and the code were built upon the works from and are owned by He et al. (2022).

He, H., Zhang, J., Wang, Y., Jiang, B., Huang, S., Wang, C., Zhang, Y., Xiong,
G., Han, X., Guo, D., He, G., & Ouyang, M. (2022). Evbattery: A
large-scale electric vehicle dataset for battery health and capacity
estimation [https://doi.org/10.48550/arXiv.2201.12358].
