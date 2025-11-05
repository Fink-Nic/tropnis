# type:ignore
import numpy as np
import matplotlib.pyplot as plt

file = "rel_err_learning_data_small.npy"
markers = iter(['v', '<', '>', '^', '1', '2', '3', '4'])

with open(file, 'rb') as f:
    min_batch_size = np.load(f)[0]
    rel_errors = np.load(f)
    fig, ax = plt.subplots()
    for rel_error in rel_errors:
        print(f"DATA FOR {rel_error=:.3f}:")
        rsds = np.load(f)
        spls = np.load(f)
        batchsizes = np.load(f)
        losses = np.load(f)
        batchsizes = np.clip(batchsizes, min_batch_size, None)
        mask = batchsizes > min_batch_size
        inv_mask = batchsizes <= min_batch_size

        marker = next(markers)
        ax.scatter(spls[mask], rsds[mask], c='blue',
                   marker=marker, label=f'{rel_error:.3f}')
        ax.scatter(spls[inv_mask], rsds[inv_mask], c='red', marker=marker)
        fig.suptitle(f"Relative error training comparison")
        ax.set_xlabel("# training samples")
        ax.set_ylabel("RSD")
    fig.legend(loc='upper right')
    plt.savefig('file'+'.png')
    # plt.show()
