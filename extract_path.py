import os
import pandas as pd

root_dir = os.path.join('.', 'data')
data_dir = os.path.join(root_dir, 'blender-off-multiview-tool', 'SHREC14')
edge_dir = os.path.join(data_dir, 'SHREC14LSSTB_TARGET_MODELS_EDGE')
copy_dir = os.path.join(data_dir, 'SHREC14LSSTB_TARGET_MODELS_TRAIN')
labels_dir = os.path.join('.', 'labels')

cls_list = os.listdir(edge_dir)

base = 'SHREC14-EDGE'
txt_dir = os.path.join('labels', base)
if not os.path.exists(txt_dir):
    os.makedirs(txt_dir)

# 把模型文件的路径存下来
# 保存映射图片位置的txt
print('----- extract data path begin -----')

cad_pd = pd.read_hdf(
    os.path.join(labels_dir, 'REORGANIZE-PART-SHREC14',
                 'reorganize_cad_orig.hdf5'), 'cad')
new_index = []
new_cat = []
new_id = []
new_split = []
# 按类名遍历
for c in cls_list:
    cls_dir = os.path.join(edge_dir, c)
    item_list = os.listdir(cls_dir)
    new_path = os.path.join(copy_dir, c)
    # 按类内的实例遍历
    for item in item_list:
        view_dir = os.path.join(cls_dir, item)
        view_list = os.listdir(view_dir)
        # 排序，保证各个视图顺序相邻
        view_list = sorted(view_list)
        # 获取实例的id
        item_id = item.split('_')[-1]
        # 获取训练-测试集的划分
        split = cad_pd.loc[(cad_pd.cat == c)
                           & (cad_pd.id == item_id)].iloc[0, 2]
        # 按实例的12视图遍历
        for view in view_list:
            im_path = os.path.join(root_dir, 'views', c, view)
            new_index.append(im_path)
            new_cat.append(c)
            new_id.append(item_id)
            new_split.append(split)
    print('class', c, 'done')
cad_pd = pd.DataFrame(data={
    'cat': new_cat,
    'id': new_id,
    'split': new_split
},
                      index=new_index)
print(cad_pd)
cad_pd.to_hdf(os.path.join(txt_dir, 'cad_edge.hdf5'), 'cad')

# 保存草图文件位置
sk_pd = pd.read_hdf(
    os.path.join(labels_dir, 'REORGANIZE-PART-SHREC14', 'sk_orig.hdf5'), 'sk')
new_index = []
for item in sk_pd.index.values:
    im_dir = os.path.dirname(item)
    fname = os.path.basename(item)
    split = im_dir.split(os.path.sep)[-1]
    clsname = im_dir.split(os.path.sep)[-2]
    new_index.append(os.path.join(root_dir, 'sketch', clsname, split, fname))
sk_pd.index = new_index
print(sk_pd)
sk_pd.to_hdf(os.path.join(txt_dir, 'sk_edge.hdf5'), 'sk')

print('----- extract data path done -----')