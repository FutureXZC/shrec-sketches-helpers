import os
import shutil

root_dir = os.path.join('.', 'data')
data_dir = os.path.join(root_dir, 'blender-off-multiview-tool', 'SHREC14')
edge_dir = os.path.join(data_dir, 'SHREC14LSSTB_TARGET_MODELS_EDGE')
copy_dir = os.path.join(data_dir, 'SHREC14LSSTB_TARGET_MODELS_TRAIN')

cls_list = os.listdir(edge_dir)

base = 'SHREC14-EDGE'
txt_dir = os.path.join('labels', base)
if not os.path.exists(txt_dir):
    os.makedirs(txt_dir)

# 把模型文件的路径存下来
# 保存映射图片位置的txt
print('----- extract data path begin -----')
with open(os.path.join(txt_dir, 'edge_data.txt'), 'w') as f1:
    with open(os.path.join(txt_dir, 'edge_data_reorg.txt'), 'w') as f2:
        for c in cls_list:
            cls_dir = os.path.join(edge_dir, c)
            item_list = os.listdir(cls_dir)
            new_path = os.path.join(copy_dir, c)
            for item in item_list:
                view_dir = os.path.join(cls_dir, item)
                view_list = os.listdir(view_dir)
                # 排序，保证各个视图顺序相邻
                view_list = sorted(view_list)
                for view in view_list:
                    im_path = os.path.join(view_dir, view)
                    # if not os.path.exists(new_path):
                    #     os.makedirs(new_path)
                    # shutil.copy(im_path, new_path)
                    f1.write('%s\n' % os.path.join(root_dir, c, view))
                    f2.write('%s\n' % os.path.join(root_dir + view))
            print('class', c, 'done')
print('----- extract data path done -----')