# shrec-sketches-helpers

Helper scripts for sketch-based 3D shape experiments.  
Useful for rendering and pre-processing SHREC14 and PART-SHREC14 datasets.

## 1. Datasets
 
SHREC14 [[website](https://sites.usm.edu/bli/sharp/sharp/contest/2014/SBR/data.html)]

## 2. Image annotations

Run [meta.ipynb](meta.ipynb) to create pandas dataframes for both datasets.
There will be 2 dataframes per dataset: one for the sketches, one for the CAD models.
Each entry in the dataframe consist of the filename path, the label and the split.  

To get *Part-SHREC14* from origin *SHREC14*, select only those classes with more than 50 samples by following the method in *[Semantic embed-ding for sketch-based 3d shape retrieval](https://www.baidu.com/link?url=mhoLeQCnYt6kVFUKBt4Sdl_RJ7Wxm5VQKsylRyikzNNRxp2pm0H5sG34B-6y29hC2Vj719d5Pi3dti89cF_K3K&wd=&eqid=a84ca2c7001353bd00000003612b5a0a)*.

## 3. Word vectors

Run [w2v.ipynb](w2v.ipynb) to get the word vector for all class names.
Word vectors are stored in a dictionary in a `.npz` file.

It requires the gensim library.

## 4. Blender 2D rendering

- Download Blender 2.79 [[link](https://download.blender.org/release/Blender2.79/)]
- Use the Blender script from the [MVCNN](https://github.com/jongchyisu/mvcnn_pytorch) project
[[link](http://people.cs.umass.edu/~jcsu/papers/shape_recog/render_shaded_black_bg.blend)]
- Render 2D projections with [render_shaded_black_bg.blend](https://github.com/zeaggler/ModelNet_Blender_OFF2Multiview)(448 × 448, dark background) or [phong.blend](https://github.com/zeaggler/ModelNet_Blender_OFF2Multiview/blob/master/phong.py)(224 × 224, white background)

It also requires the python blender package

## 5. Image pre-processing

### Resizing

Run [resize_sk.py](resize_sk.py). Sketches and 3D-view images will be resized to 224x224.
