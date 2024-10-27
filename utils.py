import matplotlib.pyplot as plt
import cv2

def visualization(ALLFILES, queryimg_index, dst, SHOW, precision, recall):

    queryimg = cv2.imread(ALLFILES[queryimg_index][0])
    queryimg_resized = cv2.resize(queryimg, (queryimg.shape[1] // 2, queryimg.shape[0] // 2))
    queryimg_rgb = cv2.cvtColor(queryimg_resized, cv2.COLOR_BGR2RGB)

    fig, axs = plt.subplots(2, SHOW, figsize=(30, 10),  gridspec_kw={'height_ratios': [1, 1]})

    for ax in axs[0]:
        ax.axis('off')
    axs[0, SHOW // 2].imshow(queryimg_rgb)
    axs[0, SHOW // 2].set_title("Query Image")

    for i in range(SHOW):
        img = cv2.imread(ALLFILES[dst[i][1]][0])
        img = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        
        axs[1, i].imshow(img_rgb)
        axs[1, i].axis('off')
        axs[1, i].set_title(f"Rank:{i+1}\nlabel:{ALLFILES[dst[i][1]][1]}\nP={precision[i]:.3f}\nR={recall[i]:.3f}")
        axs[1, i].set_xlabel(f"")#

    plt.tight_layout()  
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()