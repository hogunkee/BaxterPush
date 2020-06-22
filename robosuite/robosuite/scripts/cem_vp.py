import numpy as np
import matplotlib.pyplot as plt
from new_motion_planner import get_camera_pos, get_arm_rotation
from sklearn.mixture import GaussianMixture
from utility import segmentation_green_object, get_camera_image


'''def _gqcnn_batch(depth_im, vis_on=True, num_candidates=10, num_vps=4):
    camera_obs = env.env.sim.render(
        camera_name=env.env.camera_name,
        width=env.env.camera_width,
        height=env.env.camera_height,
        depth=env.env.camera_depth,
        #device_id=1,
    )    
    rgb, ddd = camera_obs

    n_try = 0
    while np.min(ddd) > 0.99:
        camera_obs = env.env.sim.render(
            camera_name=env.env.camera_name,
            width=env.env.camera_width,
            height=env.env.camera_height,
            depth=env.env.camera_depth,
            #device_id=1,
        )    
        rgb, ddd = camera_obs
        n_try += 1
        if n_try == 5:
            return None
    # plt.imshow(depth_im)
    # plt.show()
    
    return env.env.policy.evaluate_4Dgqcnn_batch(np.flip(rgb, axis=0), depth_im, vis_on=vis_on, num_candidates=num_candidates, num_vps=num_vps)'''

def viewpoint_quality_fn(env, arena_pos, thetas, phis, num_candidates=10, _vis_on=False): # ind_, queue_,
    quality = np.zeros(len(thetas))

    num_workers = 4 #mp.cpu_count()
    depth_hats = np.zeros([len(thetas), 256, 256])
    rgb_hats = np.zeros([len(thetas), 256, 256, 3], dtype=np.uint8)
    num_vp = len(thetas)
    for i in range(num_vp):
        camera_pos, rot_mat = get_camera_pos(arena_pos, np.array([0, phis[i], thetas[i]]))
        rgb_hats[i, ...], depth_hats[i, ...] = get_camera_image(env, camera_pos, rot_mat, arm='left', mask=True, vis_on=False)
        result, depths, d_im = env.policy.evaluate_gqcnn(rgb_hats[i, ...], depth_hats[i, ...], vis_on=False, num_candidates=10)
        quality[i] = np.mean([result_.q_value for result_ in result])
        #depth_hats[i, ...], _, _ = segmentation_green_object(rgb_hats[i, ...], depth_hats[i, ...], clip=True, vis_on=False)'''

    '''num_batch = 4
    for i in range(num_vp):
        [result, depths, d_im], rot_c_im, rot_d_im = env.env.policy.evaluate_4Dgqcnn(rgb_hats[i, ...], depth_hats[i, ...], vis_on=True, num_candidates=10)
        for ind in range(0, int(num_vp / num_batch)): # when batch evaluation
            trial_count = 0
            results = None
            d_im = None
            brk = 0
            while results is None and d_im is None:
                if trial_count!=0:
                    print('GQ-CNN failed: #%d times.'%trial_count)
                try:
                    rgb_batch = rgb_hats[ind * num_batch:(ind + 1) * num_batch, ...]
                    depth_batch = depth_hats[ind * num_batch:(ind + 1) * num_batch, ...]
                    results, depths, d_im = env.env.policy.evaluate_4Dgqcnn_batch(rgb_batch, depth_batch, vis_on=False, num_candidates=num_candidates, num_vps=num_vps)
                    #results, depths, d_im = _gqcnn_batch(depth_hats[ind * num_batch:(ind + 1) * num_batch, :, :], vis_on=_vis_on, num_candidates=num_candidates, num_vps=num_batch) 
                except:
                    print('No Valid Grasps.')
                    pass

                trial_count += 1
                if trial_count == 3:
                    brk = 1
                    break
            if brk == 1:
                quality[ind * num_batch:(ind+1)*num_batch] = 0
            else:
                #HERE!
                quality[ind * num_batch:(ind+1)*num_batch] = [result_.q_value for result_ in results] #results.q_value
                #quality[ind * num_batch:(ind+1)*num_batch] = [np.mean([result.q_value for result in result_]) for result_ in results] #results.q_value'''


    return quality

def select_vp(env, arena_pos, num_iters=3, num_comp=3, num_seeds=100, num_gmm_samples=52, gmm_reg_covar=0.01, elite_p=0.25, num_candidates=10):
    # initial uniformly random seeds
    thetas  = np.random.uniform(np.pi, 2 * np.pi, num_seeds)
    phis = np.random.uniform(0, np.pi * 0.1, num_seeds)
    #thetas  = np.random.uniform(0, 0, num_seeds)
    #phis = np.random.uniform(0, 0, num_seeds)

    for k in range(num_iters):
        # evaluate and sort
        num_vp = len(thetas)
        qs = viewpoint_quality_fn(env, arena_pos, thetas, phis, num_candidates=num_candidates, _vis_on=False)

        print('Average Viewpoint Quality: ', np.mean(qs))
        # cem_it_start = time.time()

        # extract elites
        q_values_and_indices = zip(qs, np.arange(num_vp))
        q_values_and_indices = sorted(q_values_and_indices,
                                      key=lambda x: x[0],
                                      reverse=True)
        num_elites = int(elite_p*num_vp)
        elite_q_values = [i[0] for i in q_values_and_indices[:num_elites]]
        elite_vp_indices = [i[1] for i in q_values_and_indices[:num_elites]]
        elite_thetas = [thetas[i] for i in elite_vp_indices]
        elite_phis = [phis[i] for i in elite_vp_indices]
        elite_vps = np.array([ [theta,phi] for theta, phi in zip(elite_thetas,elite_phis) ])
        # elite_grasp_arr = np.array([g.feature_vec for g in elite_grasps])

        # Normalize elite set.
        elite_vp_mean = np.mean(elite_vps, axis=0)
        elite_vp_std = np.std(elite_vps, axis=0)
        elite_vp_std[elite_vp_std == 0] = 1e-6
        elite_vps = (elite_vps - elite_vp_mean) / elite_vp_std

        # refit
        uniform_weights = (1.0 / num_comp) * np.ones(num_comp)
        gmm = GaussianMixture(n_components=num_comp,
                                 weights_init=uniform_weights,
                                 reg_covar=gmm_reg_covar)
        gmm.fit(elite_vps)

        # resample
        vp_vecs, _ = gmm.sample(n_samples=num_gmm_samples)
        vp_vecs = elite_vp_std * vp_vecs + elite_vp_mean
        thetas = np.array([vp[0] for vp in vp_vecs]) 
        phis = np.array([vp[1] for vp in vp_vecs]) 

        # print('CEM '+str(i+1)+'-th iteration time: ', time.time() - cem_it_start)

    # predict final viewpoints
    num_vp = len(thetas)
    # queue = mp.Queue()
    # num_vp_p_worker = int(num_vp/num_workers)
    # workers = [mp.Process(target=viewpoint_quality_fn, args=(np.arange(i*num_vp_p_worker,(i+1)*num_vp_p_worker), queue, src_pcd, 
    #     thetas[i*num_vp_p_worker:(i+1)*num_vp_p_worker], phis[i*num_vp_p_worker:(i+1)*num_vp_p_worker], num_candidates, False)) for i in range(num_workers)]
    # [j.start() for j in workers]
    # qs = np.zeros(num_vp)
    # for i in range(num_workers):
    #     (inds, quality) = queue.get()
    #     qs[inds] = quality
    qs = viewpoint_quality_fn(env, arena_pos, thetas, phis, num_candidates=num_candidates, _vis_on=False)
    print('Average Viewpoint Quality: ', np.mean(qs))
    # extract elites
    q_values_and_indices = zip(qs, np.arange(num_vp))
    q_values_and_indices = sorted(q_values_and_indices,
                                  key=lambda x: x[0],
                                  reverse=True)
    final_q_value = q_values_and_indices[0][0]
    final_vp_ind = q_values_and_indices[0][1]
    second_q_value = q_values_and_indices[1][0]
    second_vp_ind = q_values_and_indices[1][1]
    # qs = viewpoint_quality_fn(src_pcd=src_pcd, thetas=[thetas[final_vp_ind]], phis=[phis[final_vp_ind]], num_candidates=num_candidates, _vis_on=True)
    return thetas[final_vp_ind], phis[final_vp_ind], final_q_value