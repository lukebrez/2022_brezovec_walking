def create_clusters(brain, n_clusters):
    t0 = time.time()
    clustering_dir = "/oak/stanford/groups/trc/data/Brezovec/2P_Imaging/20201129_super_slices"
    super_to_cluster = brain.reshape(-1, 3384*9)
    connectivity = grid_to_graph(256,128)
    cluster_model = AgglomerativeClustering(n_clusters=n_clusters,
                                    memory=clustering_dir,
                                    linkage='ward',
                                    connectivity=connectivity)
    cluster_model.fit(super_to_cluster)
    print('Duration: {}'.format(time.time()-t0))
    return cluster_model

labels = []
for z in range(49):
    print(z)
    t0 = time.time()
    brain_file = "/oak/stanford/groups/trc/data/Brezovec/2P_Imaging/20201129_super_slices/superslice_{}.nii".format(z)
    brain = np.array(nib.load(brain_file).get_data(), copy=True)
    print(f'Duration: {time.time()-t0}')
    brain = np.delete(brain, fly_idx_delete, axis=-1) #### DELETING FLY_095 ####

    n_clusters = 2000
    cluster_model = create_clusters(brain, n_clusters)
    labels.append(cluster_model.labels_)
    
save_file = '/oak/stanford/groups/trc/data/Brezovec/2P_Imaging/20201129_super_slices/final_9_cluster_labels_2000'
np.save(save_file, np.asarray(labels))