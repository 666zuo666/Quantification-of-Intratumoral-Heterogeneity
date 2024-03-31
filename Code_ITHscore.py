import os
import numpy as np
import nibabel as nib
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import ITHscore
import glob
import  csv
import warnings
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
save_str =  './radiomics_ITHscore.csv'  # 把结果保存成CSV文件,  {T2,A0,A1}
create_image_list_shuffle = 1   # 判断是否创建打乱的image list
data_list_txt_str = './data_list_txt_str/'  # 所有K-fold数据List和打乱的真题数据存放的位置
image_dir_list = 'shuffle_data_ITH.txt'
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
feature_all = ['id','label','Lung']
for i in range(num_train):
    # for i in range(2):
    result_list = []
    feature_name = []
    feat_name_tmp = []

    str_split = train_images_path[i].split('/')
    img_path = train_images_path[i] + '/' + str_split[4] + '.nii.gz'
    mask_path = train_images_path[i] + '/' + str_split[4] + '-label.nii.gz'
    print("image path: " + img_path)
    print("mask path: " + mask_path)

    try:

        # 1 Load image and segmentation
        image = nib.load(img_path).get_fdata()
        seg = nib.load(mask_path).get_fdata() # 读取对应的mask文件
        image = image.transpose(2, 0, 1)  # 调换数据的1/3维度
        seg = seg.transpose(2, 0, 1)
        # print(image.shape, seg.shape)

        # 2 Get the slice with maximum tumor size
        img, mask = ITHscore.get_largest_slice(image, seg)

        # 3 Locate and extract tumor
        sub_img, sub_mask = ITHscore.locate_tumor(img, mask)

        # 4 Extract pixel-wise radiomic features
        # features = ITHscore.extract_radiomic_features(sub_img, sub_mask, parallel=False)  # 单线程
        features = ITHscore.extract_radiomic_features(sub_img, sub_mask, parallel=True, workers=4)  # 多线程

        # 5 Generate clustering label map
        # With radiomic features for each pixel, we performed pixel clustering to generate clustering label map.
        label_map = ITHscore.pixel_clustering(sub_mask, features, cluster=best_n_clusters)

        # 6 (optional) Visualize heterogeneity patterns on label map
        # option 2: multiple resolutions
        fig = ITHscore.visualize(img, sub_img, mask, sub_mask, features, cluster="all")
        # plt.show()

        # 7 Calculate ITHscore
        # we calculate ITHscore from clustering label map.
        ithscore = ITHscore.calITHscore(label_map)
        print('ITHscore = ',ithscore)


        result_list_add = [np.array(str_split[4]), np.array(str_split[3]),ithscore]
        # print('result_list_add',result_list_add)

        #  把结果保存成CSV文件
        feature_all = np.vstack((feature_all, result_list_add))  # 默认情况下，axis=0可以不写
        # feature_all.append(result_list_add)

        #  把结果保存成CSV文件
        print('feature shape=', np.array(feature_all).shape)
        np.savetxt(save_str, feature_all, delimiter=',', fmt='%s')
        print('Saved, run over!')

    # LASSO不成功,打印报错信息并跳过到下一个seed
    except Exception as e:
        print(e)
    pass