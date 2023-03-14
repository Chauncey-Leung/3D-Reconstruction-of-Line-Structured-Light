import cv2
import numpy as np
import math

np.seterr(divide='ignore', invalid='ignore')


def cv_imshow(img, img_name='img'):
    cv2.imshow(img_name, img)
    cv2.waitKey(0)


def PreProcess(gray_origin):
    """
    对图片进行预处理，gray_origin为灰度图
    :return 预处理后的图片
    """
    retval, bw = cv2.threshold(gray_origin, 255 * 0.9, 255, cv2.THRESH_BINARY)
    # 大津法效果不行，黑色背景太多，除非考虑分块，不然不如手动给定阈值
    # retval, bw2 = cv2.threshold(gray_origin, 255*0.99, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    # 裁剪，底座部分需要裁去
    # 需要根据实际情况调整，不过后面的形态学操作对连通域进行处理，可以不用裁剪
    contours, hierarch = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for i in range(len(contours)):
        if cv2.contourArea(contours[i]) < 1:  # 如果轮廓面积小于1，填充背景色（去除孤立元素）
            cv2.drawContours(bw, [contours[i]], -1, 0, thickness=-1)
            continue

    # 形态学闭操作，填充小对象
    kernel = np.array([[0, 0, 1, 0, 0],
                       [0, 1, 1, 1, 0],
                       [1, 1, 1, 1, 1],
                       [0, 1, 1, 1, 0],
                       [0, 0, 1, 0, 0]], dtype=np.uint8)
    bw2 = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, hierarchy = cv2.findContours(bw2, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    # 找到最大区域并填充非最大区域
    area = []
    for j in range(len(contours)):
        area.append(cv2.contourArea(contours[j]))
    max_idx = np.argmax(area)
    max_area = cv2.contourArea(contours[max_idx])
    for k in range(len(contours)):
        if k != max_idx:
            cv2.fillPoly(bw2, [contours[k]], 0)
    return bw2


def Hessian2D(img, sigma=1):
    Y, X = np.meshgrid(range(-round(3 * sigma), round(3 * sigma) + 1), range(-round(3 * sigma), round(3 * sigma) + 1))
    DGaussx = 1 / (2 * math.pi * (sigma ** 4)) * (-X) * np.exp(-(X * X + Y * Y) / (2 * sigma * sigma))
    DGaussy = 1 / (2 * math.pi * (sigma ** 4)) * (-Y) * np.exp(-(X * X + Y * Y) / (2 * sigma * sigma))
    DGaussxx = 1 / (2 * math.pi * (sigma ** 4)) * (X * X / (sigma * sigma) - 1) * np.exp(
        -(X * X + Y * Y) / (2 * sigma * sigma))
    DGaussxy = 1 / (2 * math.pi * (sigma ** 6)) * (X * Y) * np.exp(-(X * X + Y * Y) / (2 * sigma * sigma))
    DGaussyy = DGaussxx.T

    Dx = cv2.filter2D((img / 255), -1, DGaussx)
    Dy = cv2.filter2D(img / 255, -1, DGaussy)
    Dxx = cv2.filter2D(img / 255, -1, DGaussxx)
    Dxy = cv2.filter2D(img / 255, -1, DGaussxy)
    Dyy = cv2.filter2D(img / 255, -1, DGaussyy)
    return Dx, Dy, Dxx, Dxy, Dyy


def eigenToMatrix(Dxx, Dxy, Dyy, shape):
    """
    Hessian2D(bw, sigma)不能进行矩阵操作，自定义该函数得到2D图像各个像素处的特征值和特征向量
    :return
    eigenvalue_l Matirx,各像素处较小的特征值
    eigenvalue_h Matirx,各像素处较大的特征值
    (eigenvectorx，eigenvectory) 较大的特征值对应的特征向量
    """
    eigenvalue_l = np.zeros((shape[0], shape[1]), np.float32)
    eigenvalue_h = np.zeros((shape[0], shape[1]), np.float32)
    eigenvectorx = np.zeros((shape[0], shape[1]), np.float32)
    eigenvectory = np.zeros((shape[0], shape[1]), np.float32)
    for i in range(shape[0]):
        for j in range(shape[1]):
            if bw[i, j] == 255:
                hessian = np.zeros((2, 2), np.float32)
                hessian[0, 0] = Dxx[i, j]
                hessian[0, 1] = Dxy[i, j]
                hessian[1, 0] = Dxy[i, j]
                hessian[1, 1] = Dyy[i, j]
                ret, eigenVal, eigenVec = cv2.eigen(hessian)
                # Hessian矩阵的特征值就是形容其在该点附近特征向量方向的凹凸性
                # 特征值越大，凸性越强。
                # 所以hessian特征值绝对值最大的对应的特征向量就是光强变化最快的方向
                # 我们可以向这个地方去寻找最强光强的像素点（也就是中心线上的点）

                if ret:  # 如果存在特征值
                    # print(eigenVal.shape,eigenVec.shape)
                    # 去绝对值较大的特征值对应的特征向量为nx ny，
                    if np.abs(eigenVal[0, 0]) >= np.abs(eigenVal[1, 0]):
                        eigenvalue_h[i, j] = eigenVal[0, 0]
                        eigenvalue_l[i, j] = eigenVal[1, 0]
                        eigenvectorx[i, j] = eigenVec[0, 0]
                        eigenvectory[i, j] = eigenVec[0, 1]
                    else:
                        eigenvalue_h[i, j] = eigenVal[1, 0]
                        eigenvalue_l[i, j] = eigenVal[0, 0]
                        eigenvectorx[i, j] = eigenVec[1, 0]
                        eigenvectory[i, j] = eigenVec[1, 1]
    return eigenvalue_l, eigenvalue_h, eigenvectorx, eigenvectory


def Steger(img):
    """
    :param
    img: 预处理后的二值图
    :return
    img_res 在原图的基础上标注光带（红色）
    img_line二值图像，只显示光带
    CenterPoint_t 光带显示的整数坐标
    CenterPoint_f 光带的亚像素坐标
    """
    row, col = img.shape
    CenterPoint_t = []
    img_line = np.zeros((row, col), np.uint8)

    # 泰勒展开只能在很短的距离内适用
    # 如果px和py都小于0.5，说明这个极值点就位于当前像素内，附近光强分布函数适用
    # 如果px和py都很大，则说明这个光强最强点有点距离（同时附近光强分布函数不适用，不可以用t计算最强点）
    # 要跳过当前点，继续扫描距离激光中心更近的点
    t = -(eigenvectorx * Dx + eigenvectory * Dy) / (
            eigenvectorx * eigenvectorx * Dxx + 2 * eigenvectorx * eigenvectory * Dxy + eigenvectory * eigenvectory * Dyy)
    px = t * eigenvectorx
    py = t * eigenvectory
    for i in range(row):
        for j in range(col):
            if bw[i, j] == 255 and np.abs(px[i, j]) <= 0.5 and np.abs(py[i, j]) <= 0.5:
                CenterPoint_t.append([i, j])
    # 这里得到的CenterPoint_t为提取的光带坐标中心点，是整数，可以用来显示看效果
    # 但是还没完，我们需要进一步处理，CenterPoint_t + （px， py）是亚像素精度，用来后续的三维重建

    # LinePointNum = len(CenterPoint_t)
    img_res = img_origin.copy()
    # CenterPoint_t -> CenterPoint_f int转float
    CenterPoint_f = [[float(item[0]), float(item[1])] for item in CenterPoint_t]

    index = 0
    for point in CenterPoint_t:
        img_res[point[0], point[1], :] = (0, 0, 255)
        img_line[point[0], point[1]] = 255
        # cv_imshow(image_res)
        # cv_imshow(img_line)
        CenterPoint_f[index][0] += px[point[0], point[1]]
        CenterPoint_f[index][1] += py[point[0], point[1]]
        index += 1

    return img_res, img_line, CenterPoint_f, CenterPoint_t


def PostProcess(img_res, img_line, CenterPoint_f, CenterPoint_t, img_origin):
    """
    后处理函数，写代码发现，前述方法会导致最终图像有孤立连通域产生
    因此写该函数去除那些孤立的连通域
    此法较为暴力，如果想要优化，可以看看能否更好地预处理
    """
    contours, hierarch = cv2.findContours(img_line, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for i in range(len(contours)):
        if cv2.contourArea(contours[i]) < 20:  # 如果轮廓面积小于1，填充背景色（去除孤立元素）
            # cv2.drawContours(img_line, [contours[i]], -1, 0, thickness=-1)
            cv2.fillPoly(img_line, [contours[i]], 0)
            continue
    index = 0
    del_index = 0
    for point in CenterPoint_t:
        if img_line[point[0], point[1]] == 0:
            img_res[point[0], point[1], :] = img_origin[point[0], point[1], :]
            del CenterPoint_f[del_index]
            del_index -= 1
        index += 1
        del_index += 1
    return img_res, img_line, CenterPoint_f

if __name__ == '__main__':

    import time
    import os
    import tqdm

# =============================================================================
#     # 光平面标定
#     image_path = "../Image/LightPlaneCalibration"
#     save_path = "../Image/LightPlaneCalibration"
#     save_path_imgres = "../Image/LightPlaneCalibration/img_res"
#     save_path_imgline = "../Image/LightPlaneCalibrationd/img_line"
#     save_path_CenterPointf = "../Image/LightPlaneCalibration/CenterPoint_f"
# =============================================================================

    #扫描图像中心线提取 
    image_path = "../Image/Scan/BellImage"
    save_path = "../Image/Scan"
    save_path_imgres = "../Image/Scan/img_res"
    save_path_imgline = "../Image/Scan/img_line"
    save_path_CenterPointf = "../Image/Scan/CenterPoint_f"





    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    if not os.path.isdir(save_path_imgres):
        os.makedirs(save_path_imgres)
    if not os.path.isdir(save_path_imgline):
        os.makedirs(save_path_imgline)
    if not os.path.isdir(save_path_CenterPointf):
        os.makedirs(save_path_CenterPointf)

    sum_time = 0
    for img in tqdm.tqdm(os.listdir(image_path)):
        if img[-4:] == '.jpg':
            img_origin = cv2.imread(os.path.join(image_path, img))  # img_origin.shape = (720,1280,3)
            start_time = time.time()

            gray_origin = cv2.cvtColor(img_origin, cv2.COLOR_BGR2GRAY)
            bw = PreProcess(gray_origin)
            row, col = bw.shape
            sigma = 30 / (3 ** 0.5)  # 此参数可调，也可以自适应（需要参考文献）
            Dx, Dy, Dxx, Dxy, Dyy = Hessian2D(bw, sigma)
            eigenvalue1, eigenvalue2, eigenvectorx, eigenvectory = eigenToMatrix(Dxx, Dxy, Dyy, (row, col))
            img_res, img_line, CenterPoint_f, CenterPoint_t = Steger(bw)
            img_res, img_line, CenterPoint_f = PostProcess(img_res, img_line, CenterPoint_f, CenterPoint_t, img_origin)

            end_time = time.time()
            sum_time += end_time - start_time
            cv2.imwrite(os.path.join(save_path_imgres, img), img_res)
            cv2.imwrite(os.path.join(save_path_imgline, img.split('.')[0] + "_line.png"), img_line)
            with open(os.path.join(save_path_CenterPointf, img.split('.')[0] + ".txt"), 'w') as f:
                for i in CenterPoint_f:
                    for j in i:
                        f.write(str(j))
                        f.write(' ')
                    f.write('\n')
                f.close()


    average_time = sum_time / len(os.listdir(image_path))
    print("Average one image time: ", average_time)

# filename = 'figure/8.jpg'
# img_origin = cv2.imread(filename)
# # img_origin.shape = (720,1280,3)
# gray_origin = cv2.cvtColor(img_origin, cv2.COLOR_BGR2GRAY)
# # gray_origin.shape = (720,1280)
#
# bw = PreProcess(gray_origin)
# row, col = bw.shape
# sigma = 15 / (3 ** 0.5)  # 此参数可调，也可以自适应（需要参考文献）
#
# Dx, Dy, Dxx, Dxy, Dyy = Hessian2D(bw, sigma)
# eigenvalue1, eigenvalue2, eigenvectorx, eigenvectory = eigenToMatrix(Dxx, Dxy, Dyy, (row, col))
#
# img_res, img_line, CenterPoint_f = Steger(bw)
#
# cv_imshow(img_line)
#
# cv_imshow(img_res)
# cv_imshow(img_line)
# print(CenterPoint_f)
