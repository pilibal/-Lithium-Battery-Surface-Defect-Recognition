##Lightweight Cooperative Attention for Empowering 
YOLOv7-Tiny in Lithium Battery Surface Defect Recognition

---



## Training Steps

1. **Dataset Preparation**  
This project uses the VOC format for training. You need to prepare your own dataset before training.  
- Place annotation files in `VOCdevkit/VOC2007/Annotation/`  
- Place image files in `VOCdevkit/VOC2007/JPEGImages/`

2. **Dataset Processing**  
After organizing the dataset, use `voc_annotation.py` to generate `2007_train.txt` and `2007_val.txt` for training.  
Modify the parameters in `voc_annotation.py`. For the first training run, you only need to modify `classes_path`, which points to the txt file corresponding to the detection classes.  
When training on your own dataset, create a `cls_classes.txt` file containing your target classes, e.g.:
```python
cat
dog
...
```
Set `classes_path` in `voc_annotation.py` to point to `cls_classes.txt`, then run `voc_annotation.py`.

3. **Network Training**  
Training parameters are located in `train.py`. Please read the comments carefully after downloading the repository. **The most important parameter is `classes_path` in `train.py`.**  
**`classes_path` must point to the same txt file used in `voc_annotation.py`! This must be modified when training on your own dataset!**  
After modifying `classes_path`, run `train.py` to start training. Weights will be saved in the `logs` folder after multiple epochs.

4. **Prediction**  
Prediction requires two files: `yolo.py` and `predict.py`. In `yolo.py`, modify `model_path` and `classes_path`.  
**`model_path` points to the trained weights file in the `logs` folder.  
`classes_path` points to the txt file for detection classes.**  
After modification, run `predict.py` for detection. Enter the image path when prompted.

## Evaluation Steps

1. This project uses the VOC format for evaluation.
2. If you have run `voc_annotation.py` before training, the code will automatically split the dataset into training, validation, and test sets. To modify the test set ratio, change `trainval_percent` in `voc_annotation.py`. `trainval_percent` specifies the ratio of (training set + validation set) to test set (default 9:1). `train_percent` specifies the ratio of training set to validation set within (training set + validation set) (default 9:1).
3. After splitting the test set with `voc_annotation.py`, modify `classes_path` in `get_map.py` to point to the same txt file used during training. This must be modified when evaluating on your own dataset.
4. In `yolo.py`, modify `model_path` and `classes_path`. ** `model_path` points to the trained weights file in the `logs` folder. `classes_path` points to the txt file for detection classes.**
5. Run `get_map.py` to obtain evaluation results, which will be saved in the `map_out` folder.
6. Run predict.py to obtain visualization results.

