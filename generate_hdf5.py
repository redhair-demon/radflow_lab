import h5py
import numpy as np
import tqdm
import csv


def generate_hdf5(
        csvfile = "powerconsumption.csv",
        shape1 = 1
):
    csvdata = [] # np.array([], np.dtype('float16'))
    with open(csvfile, 'r') as file:
        for row in csv.DictReader(file):
            csvdata.append(row['PowerConsumption_Zone1'])
    csvdata = np.array(csvdata, np.dtype('float32'))
    print(len(csvdata))
    data_path = 'wiki.hdf5'
    data_f = h5py.File(data_path, 'w')

    # with open('data/wiki/trafficid2graphid.pkl', 'rb') as f:
    #     old2new = pickle.load(f)

    # with open(f'data/wiki/node_ids/test_ids.pkl', 'rb') as f:
    #     test_ids = list(pickle.load(f))

    # Consolidate masks
    bool_dt = h5py.vlen_dtype(np.dtype('bool'))
    masks = data_f.create_dataset('masks', data=[[True for _ in csvdata]])#, np.array([True for _ in csvdata], np.dtype('bool')))

    int32_dt = h5py.vlen_dtype(np.dtype('int32'))
    edges = data_f.create_dataset('edges', (1, 1), int32_dt)

    float32_dt = h5py.vlen_dtype(np.dtype('float32'))
    probs = data_f.create_dataset('probs', (1, 1), float32_dt)

    views = data_f.create_dataset('views', data=[csvdata])
    

    # # If there's enough memory, load all masks into memory
    # mask_f = h5py.File('data/wiki/masks.hdf5', 'r', driver='core')
    # views = data_f['views'][...]

    # outdegrees = np.ones((shape1, 1827), dtype=np.int32)  # self-loops
    # for key in tqdm(mask_f):
    #     for i, neigh in enumerate(mask_f[key]):
    #         m = mask_f[key][neigh][...]
    #         n_edges = (~m).astype(np.int32)
    #         outdegrees[int(neigh)] += n_edges

    # assert outdegrees.data.c_contiguous
    # data_f.create_dataset('outdegrees', dtype=np.int32, data=outdegrees)

    # normalised_views = views / outdegrees

    # for key in tqdm(mask_f):
    #     mask_dict = {}
    #     for i, neigh in enumerate(mask_f[key]):
    #         mask_dict[int(neigh)] = mask_f[key][neigh][...]

    #     if not mask_dict:
    #         continue

    #     edges_list = []
    #     probs_list = []
    #     max_count = 0
    #     kept_neigh_set = set()
    #     for day in range(1827):
    #         day_neighs = [n for n in mask_dict if not mask_dict[n][day]]
    #         sorted_neighs = sorted(day_neighs, key=lambda n: normalised_views[n, day],
    #                                reverse=True)
    #         sorted_array = np.array(sorted_neighs, dtype=np.int32)
    #         max_count = max(max_count, len(sorted_neighs))
    #         edges_list.append(sorted_array)
    #         kept_neigh_set |= set(sorted_neighs)

    #         if not sorted_neighs:
    #             probs_list.append(np.array([], dtype=np.float16))
    #             continue

    #         counts = np.array([normalised_views[n, day]
    #                            for n in sorted_neighs])
    #         counts[counts == -1] = 0
    #         # counts[np.isin(sorted_neighs, test_ids)] = 0
    #         # counts = np.log1p(counts)
    #         total = counts.sum()
    #         if total < 1e-6:
    #             probs_list.append(np.array([], dtype=np.float16))
    #             continue

    #         prob = counts / total
    #         probs_list.append(np.array(prob.cumsum(), np.float16))

    #     # Pad arrays
    #     edges_array = np.full((1827, max_count), -1, dtype=np.int32)
    #     probs_array = np.ones((1827, max_count), dtype=np.float16)

    #     for day in range(1827):
    #         edges_array[day, :len(edges_list[day])] = edges_list[day]
    #         probs_array[day, :len(probs_list[day])] = probs_list[day]

    #     probs[int(key)] = np.ascontiguousarray(probs_array)
    #     edges[int(key)] = np.ascontiguousarray(edges_array)

    #     kept_neigh_ids = sorted(kept_neigh_set)
    #     mask = np.ones((len(kept_neigh_ids), 1827), dtype=np.bool_)
    #     for i, n in enumerate(kept_neigh_ids):
    #         mask[i] = mask_dict[n]
    #         key2pos[int(key)][n] = i

    #     masks[int(key)] = np.ascontiguousarray(mask.transpose())

    # # with open('data/wiki/key2pos.pkl', 'wb') as f:
    # #     pickle.dump(key2pos, f)

    # count = 0
    # for key in tqdm(mask_f):
    #     count += len(mask_f[key])
    # print('Total edges', count)

if __name__ == "__main__":
    generate_hdf5()