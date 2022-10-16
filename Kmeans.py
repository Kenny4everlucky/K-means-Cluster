from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np


np.random.seed(22)


class Kmeans():
    def __init__(self, k=2, data: list = []) -> None:
        """[初始化]
        Args:
            k (int, optional): [要聚类的类别数]. Defaults to 2.
            data (list, optional): [要聚类的数据]. Defaults to [].
        """
        assert k <= len(data), 'Class is biger than data length'
        self.k = k
        self.__set_color__(k)
        # ~ stage 1
        self.center_idx = []
        self.center = []
        self.data = data
        while True:
            if len(self.center_idx) == self.k:
                break
            index = np.random.randint(len(data))
            if index in self.center_idx:
                continue
            self.center_idx.append(index)

        for idx in range(self.k):
            self.center.append(self.data[self.center_idx[idx]])

        # ~ stage 2
        while True:
            cluster = [[] for _ in range(self.k)]
            for i in range(len(data)):
                cluster_index = self.distance(data[i])
                cluster[cluster_index].append(i)
            self.center, updated = self.update_center(cluster)
            # self.plt(cluster=cluster)
            # plt.show()
            if updated:
                break
        # ~ stage 3
        self.res_cluster = cluster
        self.plt(cluster=cluster)
        plt.show()

    def distance(self, elem):
        """返回elem距离哪个中心最近
        Args:
            elem ([list]): [单个元素]
        Returns:
            [int]: [index]
        """

        val = np.inf
        for i in range(len(self.center)):
            dis = 0
            for j in range(len(elem)):
                dis += (self.center[i][j]-elem[j])**2
            if dis < val:
                val = dis
                min_index = i

        return min_index

    def update_center(self, cluster):
        """更新中心点
        Args:
            cluster ([二维数组]): 聚类后的索引下标，[[1,2],[0,3]]表示data[1,2]是一类，data[0,3]是一类
        Returns:
            [center_new,list]: [新的中心点]
            [bool]: [中心点]
        TODO: 这里使用三层for循环，可以直接使用numpy.average,numpy.sum指定维度计算
        """

        center_new = [[] for _ in range(self.k)]
        for i in range(len(self.data[0])):
            for j in range(len(cluster)):
                val = 0
                for k in range(len(cluster[j])):
                    val += self.data[cluster[j][k]][i]
                center_new[j].append(val/len(cluster[j]))
        #~ 计算是否变更
        for i in range(len(center_new)):
            for j in range(len(center_new[i])):
                if center_new[i][j] != self.center[i][j]:
                    return center_new, False
        return center_new, True

    def plt(self, cluster):
        """绘制聚类图，每进行一次计算中心点绘制一次
        Args:
            cluster ([二维数组]): 聚类后的索引下标，[[1,2],[0,3]]表示data[1,2]是一类，data[0,3]是一类
        """
        assert self.k <= 255*255*255, f'Color {self.k} is too much to plot'

        if len(self.data[0]) > 3:
            return

        data = np.array(self.data)
        if len(self.data[0]) == 2:
            for i in range(self.k):
                plt.plot(data[cluster[i]][:, 0], data[cluster[i]][:, 1],
                         'o', color=self.color_list[i])
            for i in range(self.k):
                plt.plot(self.center[i][0], self.center[i][1], '^',
                         color=self.color_list[len(self.color_list)-i-1], ms=10)
        if len(self.data[0]) == 3:
            fig = plt.figure()
            ax = Axes3D(fig)
            for i in range(self.k):
                ax.scatter(data[cluster[i]][:, 0], data[cluster[i]][:, 1],
                           data[cluster[i]][:, 2], 'o', color=self.color_list[i])
            for i in range(self.k):
                ax.scatter(self.center[i][0], self.center[i][1], self.center[i][2],
                           '^', color=self.color_list[i], s=60)

    def __set_color__(self, k):
        """画图所需要的颜色
        Args:
            k ([int]): [类别数，颜色类别]
        """
        color_list = []
        if k <= 125:  # 最多包含125种颜色
            R = list(range(0, 255, int(255/5)))
            G = list(range(0, 255, int(255/5)))
            B = list(range(0, 255, int(255/5)))
            for r in R:
                for g in G:
                    for b in B:
                        color_list.append('#'+"%02x" % r+"%02x" % g+"%02x" % b)

        else:  # 最多包含255*255*255种颜色
            R = list(range(0, 255))
            G = list(range(0, 255))
            B = list(range(0, 255))
            for r in R:
                for g in G:
                    for b in B:
                        color_list.append('#'+"%02x" % r+"%02x" % g+"%02x" % b)
        self.color_list = color_list

    def get_res(self):
        """返回聚类的最终结果的索引下标
        例如：[[1,2],[0,3]]表示data[1,2]是一类，data[0,3]是一类
        Returns:
            [二维数组]: [聚类的最终结果的索引下标]
        """
        return self.res_cluster


data = np.random.randint(1, 80, (300, 2))  # 2D shape=(300,2)
# data = np.random.rand(80, 3)*5          # 3D shape=(80,3)
km = Kmeans(k=4, data=data)
res = km.get_res()
print(res)  # 分类的索引
