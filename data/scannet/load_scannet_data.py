# Modified from
# https://github.com/facebookresearch/votenet/blob/master/scannet/load_scannet_data.py
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Load Scannet scenes with vertices and ground truth labels for semantic and
instance segmentations."""
import argparse
import inspect
import json
import os

import numpy as np
import scannet_utils

currentdir = os.path.dirname(
    os.path.abspath(inspect.getfile(inspect.currentframe())))


def read_aggregation(filename):
    assert os.path.isfile(filename)
    object_id_to_segs = {}
    label_to_segs = {}
    with open(filename) as f:
        data = json.load(f)
        num_objects = len(data['segGroups'])
        for i in range(num_objects):
            object_id = data['segGroups'][i][
                'objectId'] + 1  # instance ids should be 1-indexed
            label = data['segGroups'][i]['label']
            segs = data['segGroups'][i]['segments']
            object_id_to_segs[object_id] = segs
            if label in label_to_segs:
                label_to_segs[label].extend(segs)
            else:
                label_to_segs[label] = segs
    return object_id_to_segs, label_to_segs


def read_segmentation(filename):
    assert os.path.isfile(filename)
    seg_to_verts = {}
    with open(filename) as f:
        data = json.load(f)
        num_verts = len(data['segIndices'])
        for i in range(num_verts):
            seg_id = data['segIndices'][i]
            if seg_id in seg_to_verts:
                seg_to_verts[seg_id].append(i)
            else:
                seg_to_verts[seg_id] = [i]
    return seg_to_verts, num_verts


def extract_bbox(mesh_vertices, object_id_to_segs, object_id_to_label_id,
                 instance_ids):
    num_instances = len(np.unique(list(object_id_to_segs.keys())))
    instance_bboxes = np.zeros((num_instances, 7))
    for obj_id in object_id_to_segs:
        label_id = object_id_to_label_id[obj_id]
        obj_pc = mesh_vertices[instance_ids == obj_id, 0:3]
        if len(obj_pc) == 0:
            continue
        xyz_min = np.min(obj_pc, axis=0)
        xyz_max = np.max(obj_pc, axis=0)
        bbox = np.concatenate([(xyz_min + xyz_max) / 2.0, xyz_max - xyz_min,
                               np.array([label_id])])
        # NOTE: this assumes obj_id is in 1,2,3,.,,,.NUM_INSTANCES
        instance_bboxes[obj_id - 1, :] = bbox
    return instance_bboxes


def export(mesh_file,
           agg_file,
           seg_file,
           meta_file,
           label_map_file,
           output_file=None,
           test_mode=False):
    """Export original files to vert, ins_label, sem_label and bbox file.

    Args:
        mesh_file (str): Path of the mesh_file.
        agg_file (str): Path of the agg_file.
        seg_file (str): Path of the seg_file.
        meta_file (str): Path of the meta_file.
        label_map_file (str): Path of the label_map_file.
        output_file (str): Path of the output folder.
            Default: None.
        test_mode (bool): Whether is generating test data without labels.
            Default: False.

    It returns a tuple, which contains the the following things:
        np.ndarray: Vertices of points data.
        np.ndarray: Indexes of label.
        np.ndarray: Indexes of instance.
        np.ndarray: Instance bboxes.
        dict: Map from object_id to label_id.
    """
    # 标签映射文件：./data/scannet/meta_data/scannetv2-labels.combined.tsv
    # 该标签映射文件中有多种标签标准，比如 'nyu40id'
    label_map = scannet_utils.read_label_mapping(
        label_map_file, label_from='raw_category', label_to='nyu40id')

    # 加载原始点云数据，特征包括6维：XYZRGB
    mesh_vertices = scannet_utils.read_mesh_vertices_rgb(mesh_file)

    # 加载场景坐标轴对齐矩阵：一个 4x4 的变换矩阵
    # 将传感器坐标系下的原始点转化到另一个坐标系下
    # 该坐标系与房屋的两边平行（也就是与坐标轴平行）
    # Load scene axis alignment matrix
    lines = open(meta_file).readlines()

    # test set data doesn't have align_matrix
    axis_align_matrix = np.eye(4)
    for line in lines:
        if 'axisAlignment' in line:
            axis_align_matrix = [
                float(x)
                for x in line.rstrip().strip('axisAlignment = ').split(' ')
            ]
            break
    axis_align_matrix = np.array(axis_align_matrix).reshape((4, 4))

    # 对网格顶点进行全局的对齐
    # perform global alignment of mesh vertices
    pts = np.ones((mesh_vertices.shape[0], 4))

    # 同种类坐标下的原始点云，每一行的数据是 [x, y, z, 1]
    pts[:, 0:3] = mesh_vertices[:, 0:3]

    # 将原始网格顶点转换为对齐后的顶点
    pts = np.dot(pts, axis_align_matrix.transpose())  # Nx4
    aligned_mesh_vertices = np.concatenate([pts[:, 0:3], mesh_vertices[:, 3:]],
                                           axis=1)

    # 加载语义与实例标签
    # Load semantic and instance labels
    if not test_mode:
        # 每个物体都有一个语义标签，并且包含几个分割部分
        object_id_to_segs, label_to_segs = read_aggregation(agg_file)
        # 很多点属于同一分割部分
        seg_to_verts, num_verts = read_segmentation(seg_file)
        label_ids = np.zeros(shape=(num_verts), dtype=np.uint32)
        object_id_to_label_id = {}
        for label, segs in label_to_segs.items():
            label_id = label_map[label]
            for seg in segs:
                verts = seg_to_verts[seg]
                # 每个点都有一个语义标签
                label_ids[verts] = label_id
        instance_ids = np.zeros(
            shape=(num_verts), dtype=np.uint32)  # 0: unannotated
        for object_id, segs in object_id_to_segs.items():
            for seg in segs:
                verts = seg_to_verts[seg]
                # object_id 从 1 开始计数，比如 1,2,3,.,,,.NUM_INSTANCES
                # 每个点都属于一个物体
                instance_ids[verts] = object_id
                if object_id not in object_id_to_label_id:
                    object_id_to_label_id[object_id] = label_ids[verts][0]

        # 包围框格式为 [x, y, z, dx, dy, dz, label_id]
        # [x, y, z] 是包围框的重力中心, [dx, dy, dz] 是与坐标轴平行的
        # [label_id] 是 'nyu40id' 标准下的语义标签
        # 注意：因为三维包围框是与坐标轴平行的，所以旋转角是 0
        unaligned_bboxes = extract_bbox(mesh_vertices, object_id_to_segs,
                                        object_id_to_label_id, instance_ids)
        aligned_bboxes = extract_bbox(aligned_mesh_vertices, object_id_to_segs,
                                      object_id_to_label_id, instance_ids)

    else:
        label_ids = None
        instance_ids = None
        unaligned_bboxes = None
        aligned_bboxes = None
        object_id_to_label_id = None

    if output_file is not None:
        np.save(f'{output_file}_vert.npy', mesh_vertices)
        if not test_mode:
            np.save(f'{output_file}_sem_label.npy', label_ids)
            np.save(f'{output_file}_ins_label.npy', instance_ids)
            np.save(f'{output_file}_unaligned_bbox.npy', unaligned_bboxes)
            np.save(f'{output_file}_aligned_bbox.npy', aligned_bboxes)
            np.save(f'{output_file}_axis_align_matrix.npy', axis_align_matrix)

    return mesh_vertices, label_ids, instance_ids, unaligned_bboxes, \
        aligned_bboxes, object_id_to_label_id, axis_align_matrix


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--scan_path',
        required=True,
        help='path to scannet scene (e.g., data/ScanNet/v2/scene0000_00')
    parser.add_argument('--output_file', required=True, help='output file')
    parser.add_argument(
        '--label_map_file',
        required=True,
        help='path to scannetv2-labels.combined.tsv')
    opt = parser.parse_args()

    scan_name = os.path.split(opt.scan_path)[-1]
    mesh_file = os.path.join(opt.scan_path, f'{scan_name}_vh_clean_2.ply')
    agg_file = os.path.join(opt.scan_path, f'{scan_name}.aggregation.json')
    seg_file = os.path.join(opt.scan_path, f'{scan_name}_vh_clean_2.0.010000.segs.json')
    meta_file = os.path.join(opt.scan_path, f'{scan_name}.txt')  # includes axisAlignment info for the train set scans.
    
    export(
        mesh_file, 
        agg_file, 
        seg_file, 
        meta_file, 
        opt.label_map_file,
        opt.output_file
    )


if __name__ == '__main__':
    main()
