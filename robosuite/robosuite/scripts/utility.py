import numpy as np
from matplotlib import pyplot as plt
from skimage import io, segmentation as seg, color
from skimage.future import graph

def get_camera_image(env, camera_pos, rot_mat, arm='right', mask=True, vis_on=False):

    if arm == 'right':
        camera_id = env.sim.model.camera_name2id("eye_on_right_wrist")
        camera_obs = env.sim.render(
            camera_name="eye_on_right_wrist",
            width=env.camera_width,
            height=env.camera_height,
            depth=env.camera_depth
        )
        #rot_mat = get_arm_rotation(app_angle)
        env.sim.data.cam_xpos[camera_id] = camera_pos
        env.sim.data.cam_xmat[camera_id] = rot_mat.flatten()

        camera_obs = env.sim.render(
            camera_name="eye_on_right_wrist",
            width=env.camera_width,
            height=env.camera_height,
            depth=env.camera_depth
        )
        rgb, ddd = camera_obs

        n_try = 0
        while np.min(ddd) > 0.99:
            camera_obs = env.sim.render(
                camera_name="eye_on_right_wrist",
                width=env.camera_width,
                height=env.camera_height,
                depth=env.camera_depth
            )
            rgb, ddd = camera_obs
            n_try += 1
            if n_try == 5:
                return None

    elif arm == 'left':
        camera_id = env.sim.model.camera_name2id("eye_on_left_wrist")
        camera_obs = env.sim.render(
            camera_name="eye_on_left_wrist",
            width=env.camera_width,
            height=env.camera_height,
            depth=env.camera_depth
        )
        #rot_mat = get_arm_rotation(app_angle)
        env.sim.data.cam_xpos[camera_id] = camera_pos
        env.sim.data.cam_xmat[camera_id] = rot_mat.flatten()

        camera_obs = env.sim.render(
            camera_name="eye_on_left_wrist",
            width=env.camera_width,
            height=env.camera_height,
            depth=env.camera_depth
        )
        rgb, ddd = camera_obs

        n_try = 0
        while np.min(ddd) > 0.99:
            camera_obs = env.sim.render(
                camera_name="eye_on_left_wrist",
                width=env.camera_width,
                height=env.camera_height,
                depth=env.camera_depth
            )
            rgb, ddd = camera_obs
            n_try += 1
            if n_try == 5:
                return None

    extent = env.mjpy_model.stat.extent
    near = env.mjpy_model.vis.map.znear * extent
    far = env.mjpy_model.vis.map.zfar * extent

    im_depth = near / (1 - ddd * (1 - near / far))
    im_depth = np.flip(im_depth, axis=0)
    im_rgb = np.flip(rgb, axis=0)
    if mask:
        im_depth, _, _ = segmentation_green_object(im_rgb, im_depth, vis_on=False)

    if vis_on:
        io.imshow(im_rgb)
        io.show()
        io.imshow(im_depth, cmap='gray')
        io.show()
    
    return im_rgb, im_depth

def _weight_mean_color(graph, src, dst, n):
    """Callback to handle merging nodes by recomputing mean color.

    The method expects that the mean color of `dst` is already computed.

    Parameters
    ----------
    graph : RAG
        The graph under consideration.
    src, dst : int
        The vertices in `graph` to be merged.
    n : int
        A neighbor of `src` or `dst` or both.

    Returns
    -------
    data : dict
        A dictionary with the `"weight"` attribute set as the absolute
        difference of the mean color between node `dst` and `n`.
    """

    diff = graph.nodes[dst]['mean color'] - graph.nodes[n]['mean color']
    diff = np.linalg.norm(diff)
    return {'weight': diff}

def merge_mean_color(graph, src, dst):
    """Callback called before merging two nodes of a mean color distance graph.

    This method computes the mean color of `dst`.

    Parameters
    ----------
    graph : RAG
        The graph under consideration.
    src, dst : int
        The vertices in `graph` to be merged.
    """
    graph.nodes[dst]['total color'] += graph.nodes[src]['total color']
    graph.nodes[dst]['pixel count'] += graph.nodes[src]['pixel count']
    graph.nodes[dst]['mean color'] = (graph.nodes[dst]['total color'] / graph.nodes[dst]['pixel count'])

def segmentation_green_object(rgb_image, depth_image, clip=True, vis_on=False):
    labels = seg.slic(rgb_image, compactness=30, n_segments=400)
    g = graph.rag_mean_color(rgb_image, labels)

    labels2 = graph.merge_hierarchical(labels, g, thresh=35, rag_copy=False, in_place_merge=True, merge_func=merge_mean_color, weight_func=_weight_mean_color)
    out = color.label2rgb(labels2, rgb_image, kind='avg')
    mask = np.zeros(depth_image.shape)
    arm_mask = np.ones(depth_image.shape)
    mask_depth_image = np.ones(depth_image.shape) * np.max(depth_image)
    max_object_depth = -1
    for i in range(depth_image.shape[0]):
        for j in range(depth_image.shape[1]):
            if np.argmax(out[i, j, :]) == 1 and np.max(out[i, j, :]) - np.min(out[i, j, :]) > 20:
                arm_mask[i, j] = 0
                mask[i, j] = 1
                mask_depth_image[i, j] = depth_image[i, j]
                if depth_image[i, j] > max_object_depth:
                    max_object_depth = depth_image[i, j]
            else:
                if np.argmax(out[i, j, :]) == 2 and np.max(out[i, j, :]) - np.min(out[i, j, :]) > 20:
                    arm_mask[i, j] = 0

    for i in range(depth_image.shape[0]):
        for j in range(depth_image.shape[1]):
            if depth_image[i, j] >= max_object_depth:
                arm_mask[i, j] = 0
    mask_depth_image = np.clip(mask_depth_image, 0.0, max_object_depth)

    if vis_on:
        io.imshow(arm_mask, cmap='gray')
        io.show()
        io.imshow(mask_depth_image, cmap='gray')
        io.show()
    return mask_depth_image, mask, arm_mask

def segmentation_object(rgb_image, depth_image, red, green, blue, clip=True, vis_on=False):
    #no red, no black, no gray
    # preprocess rgb_image
    '''min_clip_depth = 0.4
    max_clip_depth = 0.8
    for i in range(depth_image.shape[0]):
        for j in range(depth_image.shape[1]):
            if depth_image[i, j] > max_clip_depth or depth_image[i, j] < min_clip_depth:
                rgb_image[i, j, :] = [0, 0, 0]'''

    labels = seg.slic(rgb_image, compactness=30, n_segments=400)
    g = graph.rag_mean_color(rgb_image, labels)

    labels2 = graph.merge_hierarchical(labels, g, thresh=35, rag_copy=False, in_place_merge=True, merge_func=merge_mean_color, weight_func=_weight_mean_color)
    '''max_label = np.max(labels2)
    idx = 0
    while idx <= max_label:
        for i in range(labels2.shape[0]):
            for j in range(labels2.shape[1]):
                if labels2[i, j] == idx:
                    print(rgb_image[i, j, :])
                    idx += 1
                    continue'''

    out = color.label2rgb(labels2, rgb_image, kind='avg')
    mask = np.zeros(depth_image.shape)
    arm_mask = np.zeros(depth_image.shape)
    mask_depth_image = np.ones(depth_image.shape) * np.max(depth_image)
    max_object_depth = -1
    ''' for i in range(depth_image.shape[0]):
        for j in range(depth_image.shape[1]):
            if np.max(out[i, j, :]) - np.min(out[i, j, :]) < 20: #grayscale
                continue
            elif np.argmax(out[i, j, :]) == 0 and np.max(out[i, j, :]) > 40: #red
                continue
            else:#elif np.argmax(out[i, j, :]) == np.argmax([red, green, blue]):
                arm_mask[i, j] = 0
                mask[i, j] = 1
                mask_depth_image[i, j] = depth_image[i, j]
                if depth_image[i, j] > max_object_depth:
                    max_object_depth = depth_image[i, j]'''

    for i in range(depth_image.shape[0]):
        for j in range(depth_image.shape[1]):
            if np.max(out[i, j, :]) - np.min(out[i, j, :]) < 20: #grayscale
                arm_mask[i, j] = 1
                mask[i, j] = 0
            elif np.argmax(out[i, j, :]) == 0 and np.max(out[i, j, :]) - np.min(out[i, j, :]) >  40: #red
                arm_mask[i, j] = 1
                mask[i, j] = 0
            else:#elif np.argmax(out[i, j, :]) == np.argmax([red, green, blue]):
                arm_mask[i, j] = 0
                mask[i, j] = 1
                mask_depth_image[i, j] = depth_image[i, j]
                if depth_image[i, j] > max_object_depth:
                    max_object_depth = depth_image[i, j]

    for i in range(depth_image.shape[0]):
        for j in range(depth_image.shape[1]):
            if depth_image[i, j] >= max_object_depth:
                arm_mask[i, j] = 0
    mask_depth_image = np.clip(mask_depth_image, 0.0, max_object_depth)

    if vis_on:
        io.imshow(out)
        io.show()
        io.imshow(arm_mask, cmap='gray')
        io.show()
        io.imshow(mask, cmap='gray')
        io.show()
        #io.imshow(mask_depth_image, cmap='gray')
        #io.show()
    return mask_depth_image, mask, arm_mask