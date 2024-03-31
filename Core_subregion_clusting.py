import os
import numpy as np
import nibabel as nib
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import glob
import warnings
import  csv
warnings.filterwarnings("ignore")
######################################################################################################
# 寻找最佳的cluster数量
# 使用不同的n_clusters数量并计算聚类结果的质量来确定最佳的n_clusters数量。
# 一个常用的方法是使用Kmeans，即绘制不同n_clusters数量下的聚类结果的CH值，
# 并找到CH值开始趋于平稳的拐点作为最佳的n_clusters数量。 如果存在多个nii.gz格式的image和mask文件，
# 你需要遍历所有的文件，并对每个文件都进行聚类和CH计算，最后找到所有文件中CH值最大的n_clusters数量作为最佳的n_clusters数量
######################################################################################################
# 所有超参数设置
main_path = './dataset/'   # 数据所在大文件夹名称
best_n_clusters = 5
mode_plot = 1   # 判断是否计算聚类得分并画图保存
mode_submask = 0   # 判断是否保存聚类的mask
create_image_list_shuffle = 1   # 判断是否创建打乱的image list
data_list_txt_str = './data_list_txt_str/'  # 所有K-fold数据List和打乱的真题数据存放的位置
image_dir_list = 'shuffle_data_submask.txt'
num_class = 3
##############################################################################
def create_image_list(base_path, image_dir_list,num_class):
    print('生成数据清单txt')
    image_path=[]
    for i in range(num_class):
        image_path.append(base_path+'/'+str(i)+'/')
    sum=0
    img_path=[]
    #遍历上面的路径，依次把信息追加到img_path列表中
    for label,p in enumerate(image_path):
        # image_dir=glob.glob(p+"/"+"*.nii")#返回路径下的所有图片详细的路径
        image_dir = glob.glob(p + "/" + "*")  # 返回路径下的所有图片详细的路径
        sum+=len(image_dir)
        print(len(image_dir))
        for image in image_dir:
            img_path.append((image,str(label)))
    #print(img_path[0])
    print("%d 个图像信息已经加载到txt文本!!!"%(sum))
    np.random.seed(123)  # 固定随机种子
    np.random.shuffle(img_path)
    file=open(image_dir_list,"w",encoding="utf-8")
    for img  in img_path:
        file.write(img[0]+','+img[1]+'\n')
    file.close()   # 写入后的文件内容：图片路径+对应label
########################################################################
if not os.path.exists(data_list_txt_str):
    os.makedirs(data_list_txt_str)

if create_image_list_shuffle==1:
    create_image_list(main_path, data_list_txt_str+image_dir_list,num_class)
########################################################################
train_images_path = []  # 存储训练集的所有图片路径
train_images_label = []  # 存储训练集图片对应索引信息
file = open(data_list_txt_str+image_dir_list, 'r', encoding='utf-8',newline="")
reader = csv.reader(file)
imgs_ls = []
for line in reader:
    imgs_ls.append(line)
print('Total image num=',len(imgs_ls))
file.close()
for i, row in enumerate(imgs_ls):
    train_images_path.append(row[0])  # 存储验证集的所有图片路径
    train_images_label.append(int(row[1]))  # 存储验证集图片对应索引信息
print('train num=',np.array(train_images_path).shape)
num_train = len(train_images_path)
# print(train_images_path[0])
########################################################################
# # 输入文件夹路径和文件名
# 定义3D Kmeans聚类函数
def kmeans_clustering(img, mask, n_clusters):
    # 初始化一个与原始图像相同大小的空数组，用于存储聚类后的结果
    clustered_img = np.zeros(img.shape)

    # 遍历每一层，对每一层进行Kmeans聚类
    ch_score_all = 0
    for i in range(img.shape[-1]):
        # 提取当前层的数据
        layer = img[:, :, i]
        mask_layer = mask[:, :, i]

        # 检查当前层的mask是否有ROI
        if np.any(mask_layer):
            try:
            # 将2D图像转化为1D数组，并只保留mask内的数据
                X = layer[mask_layer > 0].reshape((-1, 1))
                # 归一化处理
                X = (X - X.min()) / (X.max() - X.min())
                # 进行Kmeans聚类
                kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(X)
                # 将聚类结果转化为2D图像
                labels = kmeans.labels_.reshape(mask_layer[mask_layer > 0].shape)  # mask分开标记
                # 将当前层的聚类结果保存到clustered_img中
                clustered_img[:, :, i][mask_layer > 0] = labels + 1
                ch_score = metrics.calinski_harabaz_score(X, kmeans.predict(X))   # 评价得分
                ch_score_all = ch_score_all + ch_score

            except:
                continue
    print('calinski_harabaz_score', ch_score_all)
    return clustered_img,ch_score_all

######################################################################################################
# 定义函数，用于计算每个n_clusters对应的MSE
def calculate_mse(image_path, mask_path, n_clusters):
    img = nib.load(image_path).get_fdata()
    mask = nib.load(mask_path).get_fdata()
    clustered_img, ch_score_all = kmeans_clustering(img, mask, n_clusters)
    # mse = mean_squared_error(clustered_img[mask > 0], mask[mask > 0])
    return ch_score_all
######################################################################################################
if mode_plot==1:
    n_clusters_list = list(range(2, 7))
    mse_dict = {}
    for n_clusters in n_clusters_list:
        print('n_clusters',n_clusters)
        mse_values = []
        for i in range(num_train):
            try:
                str_split = train_images_path[i].split('/')
                img_path = train_images_path[i] + '/' + str_split[4] + '.nii.gz'
                mask_path = train_images_path[i] + '/' + str_split[4] + '-label.nii.gz'
                print("image path: " + img_path)
                mse = calculate_mse(img_path, mask_path, n_clusters)
                mse_values.append(mse)
            except Exception as e:
                print(e)
            pass
        print('mse_values',mse_values)
        # mse_dict[n_clusters] = np.median(mse_values) # np.mean(mse_values)
        # 剔除异常值，3sigma排除原则
        mean = np.mean(mse_values, axis=0)
        std = np.std(mse_values, axis=0)
        preprocessed_data_array = [x for x in mse_values if (x > mean - std)]
        mse_dict[n_clusters] = np.mean([x for x in preprocessed_data_array if (x < mean + std)])
        print('mse_dictm',mse_dict)

    ######################################################################################################
    # 找到所有文件中MSE值最小的n_clusters，即为最佳的n_clusters
    best_n_clusters = max(mse_dict, key=mse_dict.get)
    print("Best n_clusters is:", best_n_clusters)

    # 绘制MSE随n_clusters数量变化的曲线图
    fig_size1 = 8
    fig_size2 = 7
    start_mark = 18  # mark size
    label_size = 16
    line_widths = 2
    left_size = 0.17
    bottom_size = 0.12
    line_marker = ['>', 'p', 'd', 'h', 'o', 's']
    line_color = ['royalblue', 'blueviolet', 'dimgray', 'red', 'darkorange', 'green']
    font1 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 16,
             }

    mse_list = list(mse_dict.values())
    plt.plot(n_clusters_list, mse_list,line_marker[3] + '-',color=line_color[0],linewidth=line_widths, markersize=start_mark)

    ax = plt.gca()
    # axes.set_ylim([0, 1])
    plt.gcf().subplots_adjust(left=left_size, bottom=bottom_size)  # 防止底部显示不全
    plt.tick_params(labelsize=label_size)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    plt.xticks(n_clusters_list)
    plt.xlabel('Number of Clusters',font1)
    plt.ylabel('Calinski Harabaz',font1)


    plt.savefig("fig_subregion_1sigma_2to6.tif", dpi=400, bbox_inches='tight')  # 保存图片
    # plt.title('MSE vs. n_clusters')
    # plt.pause(5)  #显示秒数
    plt.show()
######################################################################################################
if mode_submask==1:
    for i in range(num_train):
        str_split = train_images_path[i].split('/')
        img_path = train_images_path[i] + '/' + str_split[4] + '.nii.gz'
        mask_path = train_images_path[i] + '/' + str_split[4] + '-label.nii.gz'
        print("image path: " + img_path)

        try:
            mse = calculate_mse(img_path, mask_path, n_clusters)
            mse_values.append(mse)
            img = nib.load(img_path)
            # 读取对应的mask文件
            mask = nib.load(mask_path)
            # 如果mask文件的大小与image文件的大小不同，则调整mask文件的大小
            # if mask.shape != img.shape:
            #     mask = nib.processing.resample_to_output(mask, img.affine, img.shape, order=0)

            habitat,ch_score_all = kmeans_clustering(img.get_fdata(), mask.get_fdata(), best_n_clusters)
            habitat_img = nib.Nifti1Image(habitat, img.affine)
            output_folder_all = imgs_path + '/habitat'  # 保存总体mask
            if not os.path.exists(output_folder_all):
                os.makedirs(output_folder_all)
            nib.save(habitat_img, os.path.join(output_folder_all, os.path.basename(img_path)))
            for i in range(1, best_n_clusters+1):
                output_folder = imgs_path + '/habitat' + str(i)   # 分别保存mask
                if not os.path.exists(output_folder):
                    os.makedirs(output_folder)

                mask_i = np.zeros_like(habitat)
                mask_i[habitat == i] = 1
                mask_i_img = nib.Nifti1Image(mask_i, img.affine)
                # out_path = os.path.join(output_folder, os.path.basename(img_path))
                nib.save(mask_i_img, os.path.join(output_folder, os.path.basename(img_path)))

        except Exception as e:
            print(e)
        pass
##############################################################################################################



