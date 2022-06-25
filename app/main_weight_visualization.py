
from datasource.mongo import MongoDataSource
import matplotlib.pyplot as plt
import time
import pickle
import numpy as np

db = MongoDataSource()

print("start get Weight")
start = int(time.time())

filters = db.getLastWeight()

print("end get Weight")
print("***get Weight time(sec) :", int(time.time()) - start)

# with open("app/data/model/game2048_dqn_multi.h5", "wb") as f:
#     pickle.dump(filters, f)


# with open("app/data/model/game2048_dqn_multi.h5", 'rb') as f:
#     filters = pickle.load(f) # 단 한줄씩 읽어옴
filters = np.array(filters)
print("load")
# print(filters)
ix = 1
# for filter in filters:
filter = filters[0]
print(filter.shape)
n_filters = filter.shape[3]
print(n_filters)
for i in range(n_filters):
    # get the filter
    f = filter[:, :, :, i]
    # plot each channel separately
        # specify subplot and turn of axis
    ax = plt.subplot(int(filter.shape[3]/16), 16, ix)
    ax.set_xticks([])
    ax.set_yticks([])
            # plot filter channel in grayscale
    print(f.shape)
    s = f.shape[0] * f.shape[1] * f.shape[2]
    print(f.reshape(4, int(s/4)))
    plt.imshow(f.reshape(4, int(s/4)), cmap='gray')
    ix += 1
# show the figure
plt.axis('off')
plt.tight_layout()
plt.subplots_adjust(left = 0, bottom = 0, right = 1, top = 1, hspace = 0, wspace = 0)
plt.show()

# poetry run python app/main_weight_visualization.py