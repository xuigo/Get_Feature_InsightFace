## Get_Feature_InsightFace

Tensorflow implementation of extracting features from a well-trained InsightFace model, which can be used to compare face similarity and calculate ID loss.

### Installation requirements

We use GPU and CPU respectively for image generation, both of which can work normally. All the dependence have been packaged to **requirments.txt** and you can use it for installation.

```shell
pip install -r requirments.txt
```

### Run

First，Modify the hyperparameters in the **run.py** file, such as **MODEL, PATH1, PATH2**. Before this, we assume that you have downloaded the checkpoint files and placed it in a suitable location.

```python
python run.py   # both GPU and CPU work well
```

### Results

```powershell
>>>
>>> Face Similarity is:0.9652482867240906  Face angle is:0.08416257798671722
```

We provide two return results, namely the similarity of faces and the angle difference between faces, named as similarity, angle. Both can be used for face recognition and ID loss calculation, the difference is, **similarity**: the bigger the more similar; **angle**: the smaller the value, the smaller the angle.

