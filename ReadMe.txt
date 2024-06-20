测试结果文件下载地址：
百度网盘：
链接：https://pan.baidu.com/s/1RxdttZnZBhuPqQOCqUbbBw?pwd=hge6 
提取码：hge6
OneDrive：
https://1drv.ms/u/c/90bf30fdc8dd9ee7/EVxOhbh2Z5hOmhuejKJRM80BJ4HYRSDUSCh0YQkwWt-R-w?e=FXL6ro


以下依次介绍Baseline模型训练、测试、评分的流程：

一、训练
    1.准备数据。将你的数据集放在dataset文件夹下。images包含所有的训练集图像，mask包含所有的标签。由随机采样的方式划分训练集和测试集。
        Baseline的划分为训练集7000张，测试集2000张，将训练集和测试集的TXT放在70_20文件夹下。请仿照ICPR_Track2文件的放置来规划你的数据集。
    2.路径更新。下载测试结果文件，解压到result_WS路径下。
        model/parse_args_train.py 中dataset更新为你的数据集名称，root更新指向dataset文件夹的路径，
        split_method更新为存储你的划分TXT的文件夹名称，base_size、crop_size将图片尺寸调整为512*512。
    3.运行train_test_evaluation.py 开启训练，训练结果和权重文件保存在result_WS/ICPR_Track2中。训练完成后，会打印模型在测试集上取得的指标数值。

二、测试
    1.更新路径。model/parse_args_test.py 中st_model指向result_WS中的文件夹，model_dir指向训练得到的权重.pth.tar的路径，root指向你的数据集路径。
    2.测试模型精度。运行test_and_visulization.py 即可获得测试指标，Pd、Fa、mIoU.
    3.测试模型复杂度。model/complexity.py 中net更新为你的模型，更新好图片的尺寸，如Baseline的图片尺寸为（3，512，512）。
        运行complexity.py 打印的Total params即为模型的参数量，Total FLops即为模型的运算量。

三、评分
    1.使用你的评价指标更新Eval.py中的参数。以下参数需要更新为你的指标数值，Fa虚警率，Pd检测率，IoU平均交并比，Params参数量，Flops运算量。
        注意格式、单位与Baseline保持一致，如参数量单位为M，运算量单位为GFLOPs等。
    2.运行Eval.py 得到评分Score。
