import random
import pandas as pd
class vehicle():
    def __init__(self,mode=None):
        self.abs_v = random.choice([10.0])
        if mode==None:
            mode = random.choice(['up','down','left','right'])
            self.x = random.uniform(-245.0,245.0)
            self.y = random.uniform(-428.0,428.0)  
            if mode=='up':
                self.y = 0 if self.y<0 else 428
                self.vy = 0
                self.vx = self.abs_v * random.choice([1,-1])
            elif mode=='down':
                self.y = 0 if self.y>0 else -428
                self.vy = 0
                self.vx = self.abs_v * random.choice([1,-1])
            elif mode=='left':
                self.x = -245 if self.x<0 else 0
                self.vx = 0
                self.vy = self.abs_v * random.choice([-1,1])
            elif mode=='right':
                self.x = 245 if self.x>0 else 0
                self.vx = 0
                self.vy = self.abs_v * random.choice([-1,1])
        else:
            print('vehicle mode error')
            exit()

        self.pre_x = self.x
        self.pre_y = self.y
    
    def move(self,time):
        self.x += self.vx * time
        self.y += self.vy * time

    def check(self):
        if self.x == self.pre_x and self.y==self.pre_y:
            return
        assert self.vx * self.vy==0 , "vx:{};   vy:{}".format(self.vx,self.vy)
        if self.vx==0:
            if   self.y>=428.001 and self.x==-245:   # 左上角
                self.y = 428.0
                self.vy = 0
                self.vx = self.abs_v
            elif self.y>=428.001 and self.x==245:    # 右上角
                self.y = 428.0
                self.vy = 0
                self.vx = self.abs_v * -1
            elif self.y>=428.001 and self.x==0:      # 上边中间
                self.y = 428.0
                self.vy = 0
                self.vx = self.abs_v * random.choice([1,-1])
            elif self.y<=-428.001 and self.x==-245:  # 左下角
                self.y = -428.0
                self.vy = 0
                self.vx = self.abs_v 
            elif self.y<=-428.001 and self.x==245:   # 右下角
                self.y = -428.0
                self.vy = 0
                self.vx = self.abs_v * -1
            elif self.y<=-428.001 and self.x==0:     # 下边中间
                self.y = -428.0
                self.vy = 0
                self.vx = self.abs_v * random.choice([1,-1])
            elif self.y*self.pre_y<0:               # 穿过了中线
                assert self.y*self.vy>0, 'vy:{};  y:{};  pre_y:{}'.format(self.vy,self.y,self.pre_y)
                if self.x==-245:                       # 左侧穿过 
                    if random.choice([0,1])==1:        # 一半概率保持行驶状态，一半概率改为向右行
                            self.y = 0
                            self.vy = 0
                            self.vx = self.abs_v      
                elif self.x== 245:                     # 右侧穿过 
                    if random.choice([0,1])==1:        # 一半概率保持行驶状态，一半概率改为向左行
                            self.y = 0
                            self.vy = 0
                            self.vx = self.abs_v * -1   
                else:                                  # 中间穿过 
                    assert self.x==0, 'vy:{};  vx:{};  x:{}'.format(self.vy,self.vx,self.x)
                    mode = random.choice([0,1,2])        # 1/3概率保持行驶状态，1/3概率改为向左行,1/3概率改为向右行
                    if mode == 1:                         
                        self.y = 0
                        self.vy = 0
                        self.vx = self.abs_v * -1
                    elif  mode == 2:   
                        self.y = 0
                        self.vy = 0
                        self.vx = self.abs_v                                          
            else:                                    # 没有碰到边界
                pass
        else:
            if   self.y==428.0 and self.x<=-245.001:   # 左上角
                self.x = -245.0
                self.vx = 0
                self.vy = self.abs_v * -1
            elif self.y==-428 and self.x<=-245.001:    # 左下角
                self.x = -245.0
                self.vx = 0
                self.vy = self.abs_v 
            elif self.y==0 and self.x<=-245.001:       # 左侧中间
                self.x = -245.0
                self.vx = 0
                self.vy = self.abs_v * random.choice([1,-1])
            elif self.y==428.0 and self.x>=245.001:    # 右上角
                self.x = 245.0
                self.vx = 0
                self.vy = self.abs_v * -1 
            elif self.y==-428.0 and self.x>=245.001:   # 右下角
                self.x = 245.0
                self.vx = 0
                self.vy = self.abs_v 
            elif self.y==0 and self.x>=245.001:        # 右侧中间
                self.x = 245.0
                self.vx = 0
                self.vy = self.abs_v * random.choice([1,-1])
            elif self.x*self.pre_x<0:               # 穿过了中线
                assert self.x*self.vx>0, 'vx:{};  x:{};  pre_x:{}'.format(self.vx,self.x,self.pre_x)
                if self.y==428:                        # 上面穿过 
                    if random.choice([0,1])==1:        # 一半概率保持行驶状态，一半概率改为向下行
                            self.x = 0
                            self.vx = 0
                            self.vy = self.abs_v * -1 
                elif self.y==-428.0:                   # 下面穿过 
                    if random.choice([0,1])==1:        # 一半概率保持行驶状态，一半概率改为向上行
                            self.x = 0
                            self.vx = 0
                            self.vy = self.abs_v 
                else:                                  # 中间穿过 
                    assert self.y==0, 'vy:{};  vx:{};  x:{}'.format(self.vy,self.vx,self.x)
                    mode = random.choice([0,1,2])        # 1/3概率保持行驶状态，1/3概率改为向上行,1/3概率改为向下行
                    if mode == 1:                         
                        self.x = 0
                        self.vx = 0
                        self.vy = self.abs_v * -1
                    elif  mode == 2:   
                        self.x = 0
                        self.vx = 0
                        self.vy = self.abs_v                                          
            else:                                    # 没有碰到边界
                pass

        self.pre_x = self.x
        self.pre_y = self.y

    def getPos(self):
        return [self.x, self.y]

class fix_vehicle():
    def __init__(self, x, y, vx, vy):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        
    def getPos(self):
        return [self.x, self.y]
    
    def move(self,time):
        self.x += self.vx * time
        self.y += self.vy * time
        
# 相互发送的版本
# class traffic_scene():
#     def __init__(self, v2v_num):
#         self.v2vList = [vehicle() for _ in range(v2v_num)]
#         self.get_neighbor()
#     def get_neighbor(self):
#         neigh = np.array([v.getPos() for v in self.v2vList])
#         self.neighbor = []
#         for i in self.v2vList:
#             temp = neigh - i.getPos()
#             temp = temp.T
#             temp = np.hypot(temp[0],temp[1])
#             idx = np.argsort(temp)[1]
#             self.neighbor.append(self.v2vList[idx])

#     def run(self,time):import random
# class fix_random_vehicle():
#     def __init__(self, direction,co):
#         if direction=='up':
#             self.x = random.uniform(-10,10)
#             self.y = random.uniform(100,300)
#             self.vx = 0.0
#             self.vy = 10.0 if self.x>0 else -10.0
#         elif direction=='down':
#             self.x = random.uniform(-10,10)
#             self.y = random.uniform(-300,-100)
#             self.vx = 0.0
#             self.vy = 10.0 if self.x>0 else -10.0
#         elif direction=='left':
#             self.x = random.uniform(-300,-100)
#             self.y = random.uniform(-10,10)
#             self.vx = 10.0 if self.y<0 else -10.0
#             self.vy = 0.0
#         elif direction=='right':
#             self.x = random.uniform(100,300)
#             self.y = random.uniform(-10,10)
#             self.vx = 10.0 if self.y<0 else -10.0
#             self.vy = 0.0
        
#     def getPos(self):
#         return [self.x, self.y]
    
#     def move(self,time):
#         self.x += self.vx * time
#         self.y += self.vy * time
         
class fix_random_vehicle():
    def __init__(self, direction, coefficient):
        if direction=='up':
            self.x = random.uniform(-10,10)
            self.y = random.uniform(80.0+coefficient*100.0,105.0+coefficient*100.0)
            self.vx = 0.0
            self.vy = 10.0 if self.x>0 else -10.0
        elif direction=='down':
            self.x = random.uniform(-10,10)
            self.y = random.uniform(-1*(105.0+coefficient*100.0),-1*(80.0+coefficient*100.0))
            self.vx = 0.0
            self.vy = 10.0 if self.x>0 else -10.0
        elif direction=='left':
            self.x = random.uniform(-1*(105.0+coefficient*100.0),-1*(80.0+coefficient*100.0))
            self.y = random.uniform(-10,10)
            self.vx = 10.0 if self.y<0 else -10.0
            self.vy = 0.0
        elif direction=='right':
            self.x = random.uniform(80.0+coefficient*100.0,105.0+coefficient*100.0)
            self.y = random.uniform(-10,10)
            self.vx = 10.0 if self.y<0 else -10.0
            self.vy = 0.0
        
    def getPos(self):
        return [self.x, self.y]
    
    def move(self,time):
        self.x += self.vx * time
        self.y += self.vy * time
         


# 固定位置版本
class traffic_scene():
    def __init__(self, v2v_num):
        optional_location = np.array([[-8.26950561e+00,3.35295069e+02  ,0.00000000e+00 ,-1.00000000e+00],
                                        [ 1.33638390e+02,3.41098253e+00, -1.00000000e+00 , 0.00000000e+00],
                                        [ 3.03786071e+00,-2.26309603e+02 , 0.00000000e+00  ,1.00000000e+00],
                                        [-1.74755890e+02,-2.42335835e+00  ,1.00000000e+00  ,0.00000000e+00],
                                        [ 1.85023640e-01,1.74487337e+02  ,0.00000000e+00 , 1.00000000e+00],
                                        [ 1.80873482e+02,-4.56125000e-02 , 1.00000000e+00 , 0.00000000e+00],
                                        [-1.37386366e+00,-3.48022617e+02 , 0.00000000e+00, -1.00000000e+00],
                                        [-1.51631111e+02,-7.77201789e+00 , 1.00000000e+00 , 0.00000000e+00],
                                        [ 9.77643200e-02,7.31856871e+00 , 0.00000000e+00  ,1.00000000e+00],
                                        [ 1.39675640e+02,4.23037450e+00 ,-1.00000000e+00 , 0.00000000e+00],
                                        [-2.74118376e+00,-2.06292778e+02 , 0.00000000e+00, -1.00000000e+00],
                                        [-1.83061517e+02,-7.07142843e+00 , 1.00000000e+00 , 0.00000000e+00],
                                        [ 6.23672200e-01,2.14281452e+02 , 0.00000000e+00 , 1.00000000e+00],
                                        [ 1.51198958e+02,-1.10468926e+00 ,1.00000000e+00 , 0.00000000e+00],
                                        [ 2.80177996e+00,-3.36645206e+02,0.00000000e+00,1.00000000e+00],
                                        [-4.19785698e+01,4.32568267e+00,-1.00000000e+00,0.00000000e+00]])
        self.v2vList = [fix_vehicle(optional_location[i][0],optional_location[i][1],optional_location[i][2],optional_location[i][3]) for i in range(v2v_num)]
        # df = pd.read_csv('./env/vehicle_data.csv')
        # optional_location = df.to_numpy()
        # self.v2vList = [fix_vehicle(optional_location[i][0],optional_location[i][1],optional_location[i][2],optional_location[i][3]) for i in range(v2v_num)]

    def run(self,time):
        pass

    def getPos(self):
        v2vSenderPosList = [v.getPos() for v in self.v2vList]
        v2vRecvPosList = []
        for v in self.v2vList:
            v.move(-2.5)
            v2vRecvPosList.append(v.getPos())
            v.move(2.5)
        return v2vSenderPosList, v2vRecvPosList

# 固定位置但是相互发送的版本
# class traffic_scene():
#     def __init__(self, v2v_num):
#         # optional_location = np.array([[-8.26950561e+00,3.35295069e+02  ,0.00000000e+00 ,-1.00000000e+00],
#         #                                 [ 1.33638390e+02,3.41098253e+00, -1.00000000e+00 , 0.00000000e+00],
#         #                                 [ 3.03786071e+00,-2.26309603e+02 , 0.00000000e+00  ,1.00000000e+00],
#         #                                 [-1.74755890e+02,-2.42335835e+00  ,1.00000000e+00  ,0.00000000e+00],
#         #                                 [ 1.85023640e-01,1.74487337e+02  ,0.00000000e+00 , 1.00000000e+00],
#         #                                 [ 1.80873482e+02,-4.56125000e-02 , 1.00000000e+00 , 0.00000000e+00],
#         #                                 [-1.37386366e+00,-3.48022617e+02 , 0.00000000e+00, -1.00000000e+00],
#         #                                 [-1.51631111e+02,-7.77201789e+00 , 1.00000000e+00 , 0.00000000e+00],
#         #                                 [ 9.77643200e-02,7.31856871e+00 , 0.00000000e+00  ,1.00000000e+00],
#         #                                 [ 1.39675640e+02,4.23037450e+00 ,-1.00000000e+00 , 0.00000000e+00],
#         #                                 [-2.74118376e+00,-2.06292778e+02 , 0.00000000e+00, -1.00000000e+00],
#         #                                 [-1.83061517e+02,-7.07142843e+00 , 1.00000000e+00 , 0.00000000e+00],
#         #                                 [ 6.23672200e-01,2.14281452e+02 , 0.00000000e+00 , 1.00000000e+00],
#         #                                 [ 1.51198958e+02,-1.10468926e+00 ,1.00000000e+00 , 0.00000000e+00],
#         #                                 [ 2.80177996e+00,-3.36645206e+02,0.00000000e+00,1.00000000e+00],
#         #                                 [-4.19785698e+01,4.32568267e+00,-1.00000000e+00,0.00000000e+00]])
#         # 从CSV文件读取数据
#         df = pd.read_csv('./env/vehicle_data.csv')

#         # 将DataFrame转换为numpy数组
#         optional_location = df.to_numpy()
#         self.v2vList = [fix_vehicle(optional_location[i][0],optional_location[i][1],optional_location[i][2],optional_location[i][3]) for i in range(v2v_num)]
#         self._get_neighbor()

#     def _get_neighbor(self):
#         neigh = np.array([v.getPos() for v in self.v2vList])
#         self.neighbor = []
#         for i in self.v2vList:
#             temp = neigh - i.getPos()
#             temp = temp.T 
#             temp = np.hypot(temp[0],temp[1])
#             idx = np.argsort(temp)[1]
#             self.neighbor.append(self.v2vList[idx])

#     def run(self,time):
#         pass

#     def getPos(self):
#         v2vSenderPosList = [v.getPos() for v in self.v2vList]
#         v2vRecvPosList = [v.getPos() for v in self.neighbor]
#         return v2vSenderPosList, v2vRecvPosList

import math 

import numpy as np

def time2db(x):
    return 10.0 * np.log10(x)

def db2time(x):
    return np.power(10.0,x/10.0)

class channel():
    def __init__(self, v2v_num, channel_num):
        self.BSpos = [0,0]
        self.traffic = traffic_scene(v2v_num)
        self.v2v_num = v2v_num
        self.chan_num = channel_num
        self.v2vShadowStd = 3
        self.v2iShadowStd = 8
        self.V2V_shadowing = np.random.normal(0,self.v2vShadowStd,(self.v2v_num,))
        self.V2i_shadowing = np.random.normal(0,self.v2iShadowStd,(self.v2v_num,))
        self.bsAntGain = 8
        self.bsNoiseFigure = 5
        self.vehAntGain = 3
        self.vehNoiseFigure = 9

    def V2I_Los_fading(self,pos,v2i_idx):
        # 计算路损
        BS_height = 25
        veh_height = 1.5
        dis = math.hypot(pos[0],pos[1])
        PL = 128.1 + 37.6 * np.log10(np.sqrt(np.power(dis,2) + (BS_height - veh_height) ** 2) / 1000)
        # 计算shadowing
        Decorrelation_distance = 50

        self.V2i_shadowing[v2i_idx] = np.exp(-1 * (1/20.0/Decorrelation_distance)) * self.V2i_shadowing[v2i_idx] \
            + np.sqrt(1 - np.exp(-2 * (1/20.0 / Decorrelation_distance))) * np.random.normal(0, self.v2iShadowStd)
        return PL+self.V2i_shadowing[v2i_idx]-self.bsAntGain-self.vehAntGain+self.bsNoiseFigure
        # return PL
    
    def V2V_Los_fading(self,pos1,pos2,v2v_idx):
        # 计算路损
        recv_height = 1.5
        send_height = 1.5
        fc = 2
        dx = pos1[0] - pos2[0]
        dy = pos1[1] - pos2[1]
        dis = math.hypot(dx,dy) + 0.0001
        d_bp = 4 * (recv_height - 1) * (send_height - 1) * fc * (10 ** 9) / (3 * 10 ** 8)

        def LOS(d):
            if d<=3:
                Los = 22.7 * np.log10(3) + 41 + 20 * np.log10(fc / 5)
            elif d<d_bp:
                Los = 22.7 * np.log10(d) + 41 + 20 * np.log10(fc / 5)
            else:
                Los = 40.0 * np.log10(d) + 9.45 - 17.3 * np.log10(send_height) - 17.3 * np.log10(send_height) + 2.7 * np.log10(fc / 5)
            return Los

        def NLOS(d_x,d_y):
            n_j = max(2.8 - 0.0024 * d_y, 1.84)
            return LOS(d_x) + 20 - 12.5 * n_j + 10 * n_j * np.log10(d_y) + 3 * np.log10(fc / 5)
        
        if min(dx,dy) < 10:
            PL = LOS(dis)
        else:
            PL = min(NLOS(dx,dy),NLOS(dy,dx))

        # 计算shadowing
        Decorrelation_distance = 10
        self.V2V_shadowing[v2v_idx] = np.exp(-1 * (1/10.0/Decorrelation_distance)) * self.V2V_shadowing[v2v_idx] \
            + np.sqrt(1 - np.exp(-2 * (1/10.0 / Decorrelation_distance))) * np.random.normal(0, self.v2vShadowStd)
        return PL+self.V2V_shadowing[v2v_idx]-self.vehAntGain*2+self.vehNoiseFigure
        # return PL 

    #     ###################生成距离矩阵##########################
    #     # 存放信道信息的矩阵的排布如下所示
    #     # ┌─                           ─┐
    #     # │       v2vS1 v2vS2 ... v2vSn │
    #     # │v2vR1                        │
    #     # │v2vR2                        │
    #     # │ ...                         │
    #     # │v2vRn                        │
    #     # │ BS1                         │
    #     # │ ...                         │
    #     # │ BSn                         │
    #     # └─                           ─┘

    def calculate_large_fading(self):
        v2vSList, v2vRList = self.traffic.getPos()
        SList = v2vSList 
        fading_array = [ [ self.V2V_Los_fading(SList[j],v2vRList[i],j) for j in range(len(SList))] for i in range(len(v2vRList))]
        fading_array.append([self.V2I_Los_fading(SList[j],j) for j in range(len(SList))])
        self.large_fading_array = np.array(fading_array,dtype=np.float32)

    def update_large_fading(self):
        self.traffic.run(0.1)
        self.calculate_large_fading()

    def update_fast_fading(self):
        self.CSI_array = np.expand_dims(self.large_fading_array,0).repeat(self.chan_num,axis=0)
        fast_fading_array = np.random.exponential(scale=1,size=(self.chan_num,self.v2v_num+1,self.v2v_num))
        fast_fading_array = time2db(fast_fading_array)
        self.CSI_array = self.CSI_array + fast_fading_array
        self.CSI_array = np.concatenate((self.CSI_array,np.repeat(self.CSI_array[:,-1:,:],self.v2v_num-1,axis=1)),axis=1)

    def get_CSI_array(self):
        if hasattr(self,'CSI_array'):
            return self.CSI_array
        else:
            self.calculate_large_fading()
            self.update_fast_fading()
            return self.CSI_array

import gymnasium as gym
# from ray.rllib.env.multi_agent_env import MultiAgentEnv
from pettingzoo.utils.env import ParallelEnv,AECEnv
import functools

v2v_bias_array = [37.528934,66.85606,75.36916,54.97333,37.749146,66.825325,75.19542,54.94541,]
v2v_scale_array = [5.7747164,5.7364583,5.8057914,5.383468, 5.426577,5.396247,5.6161017, 5.3047013, ]
v2i_bias_array = [87.034775,92.854126,99.405785,97.2252,87.47563, 97.60653,91.71312,94.899254,]
v2i_scale_array = [5.682411,5.714255,5.7594466,5.72364,5.6979065,5.616413,5.695519,5.6114902,]
In_bias_array = [-90.02143, -92.13434, -94.56302, -94.046326, -89.928665, -90.32962, -97.28236, -96.56299]
In_scale_array = [33.198353,33.658314, 35.192787,35.113323,33.27018,33.349865, 35.900333, 34.97152, ]

class v2vEnv(ParallelEnv):
    def __init__(self,config):
        self.metadata = config
        self.v2v_num = config['agent_num']
        self.channel_num = config['channel_num']
        self.chan = channel(self.v2v_num, self.channel_num) 
        self.max_power = db2time(config['max_power_dbm'])/1000.0    # v2i用户功率为30dbm，此处转化为瓦特的单位
        self.n0 = config['AWGN_n0']                # 单位 dbm
        self.band_width = config['bandwidth']
        self.N0 = db2time(self.n0)/1000.0      # 高斯白噪声功率为-114dbm，此处转化为瓦特的单位
        self.period = config['period']
        self.slot = config['slot']
        self.pre_v2v_payload = config['payload']
        self.pre_v2i_threshold = config['v2i_threshold_per_second_Hz'] * self.band_width / 1000.0 * self.slot
        self.power_weight = config['power_weight']
        self.v2v_weight = config['v2v_weight']
        self.v2i_weight = config['v2i_weight']
        self.global_weight = config['global_weight']
        self.aoi_requirement = config['aoi_requirement']
        self.action_level = config['action_level']
        
        # 100ms 发送 patload bytes数据           
        self.timeleft = self.period
        self.render_mode = False
        self.success = np.zeros((self.v2v_num,),dtype=np.float32)
        self.aoi = np.clip(np.random.normal(self.slot*20,self.slot*5,(self.v2v_num,)),self.slot,self.aoi_requirement) 
        self.aoi_count = np.zeros((self.v2v_num,),dtype=np.float32)
        # self.pow_count = np.zeros((self.v2v_num,),dtype=np.float32)
        
        
        self.payload = np.array([self.pre_v2v_payload] * self.v2v_num, dtype=np.float32)   

        self.agents = self.possible_agents = [i for i in range(self.v2v_num)]
        self.observation_spaces = {i : gym.spaces.Box(-1.0, 1.0, (self.channel_num*3+3,), dtype=np.float32) for i in self.agents} #考虑db单位的信噪比 所以最小值设置为0
        self.action_spaces = {i: gym.spaces.Discrete(self.channel_num*2*self.action_level) for i in self.agents}    
                      

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent) -> gym.spaces.Space:
        return self.observation_spaces[agent]

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent) -> gym.spaces.Space:
        return self.action_spaces[agent]
     
    def reset(self, seed=None, return_info=False, options=None):
        self.agents = self.possible_agents
        self.success = np.zeros((self.v2v_num,),dtype=np.float32)     
        # self.aoi = np.array([self.slot] * self.v2v_num, dtype=np.float32)               
        self.aoi = np.clip(np.random.normal(self.slot*20,self.slot*5,(self.v2v_num,)),self.slot,self.aoi_requirement)     # aoi是否要重置，我也不知道
        self.aoi_count = np.zeros((self.v2v_num,),dtype=np.float32)
        self.payload = np.array([self.pre_v2v_payload] * self.v2v_num, dtype=np.float32)   
        self.timeleft = self.period
        self.chan = channel(self.v2v_num,self.channel_num)

        self.chan.update_large_fading()

        CSI_array = self.chan.get_CSI_array() 
        [v2v_CSI,v2i_CSI] = np.split(CSI_array, 2, axis=1)
        v2v_CSI = np.triu(v2v_CSI)-np.triu(v2v_CSI,k=1)
        v2i_CSI = np.triu(v2i_CSI)-np.triu(v2i_CSI,k=1)
        v2v_CSI = (np.ones((self.v2v_num,),dtype=np.float32) @ v2v_CSI).transpose()
        v2i_CSI = (np.ones((self.v2v_num,),dtype=np.float32) @ v2i_CSI).transpose()
        for i in range(self.v2v_num):
            v2v_CSI[i] = (v2v_CSI[i] - v2v_bias_array[i])/3.0/v2v_scale_array[i]
            v2i_CSI[i] = (v2i_CSI[i] - v2i_bias_array[i])/3.0/v2i_scale_array[i]
        Inter = np.zeros((self.channel_num,self.v2v_num),dtype=np.float32)
        Inter = np.concatenate((Inter,
                            np.expand_dims(self.payload/self.pre_v2v_payload,0),                    
                            np.array([[self.timeleft/self.period]*self.v2v_num],dtype=np.float32),
                            # np.array([[i for i in range(self.v2v_num)]],dtype=np.float32),
                            np.expand_dims(self.aoi/20.0,0))) 
        all_state = Inter.transpose()
        obs = {i:np.concatenate((v2v_CSI[i],v2i_CSI[i],all_state[i])) for i in self.agents}
        return obs,{}

    def seed(self, seed=None):
        pass
    
    def render(self) -> None:
        return None
    
    def state(self) -> np.ndarray:
        return self.all_state
    
    def action_decode(self,action):
        if isinstance(action,np.ndarray):
            if len(action.shape) != 0 :
                action = action[0]
        assert 0<= action < self.channel_num*self.action_level*2, "action:{} outside the legal range".format(action)
        mode = action // (self.channel_num*self.action_level)
        rest = action % (self.channel_num*self.action_level)
        chan = rest // self.action_level
        power = rest % self.action_level
        power = self.max_power * power / (self.action_level-1)
        return mode, chan, power
    
    def get_feedback(self,channel_select,power_list,CSI_times_array):
        # 信道选择的处理
        """
        信道选择的矩阵形式说明：是一个三维矩阵，具有channel_num个元素
        每一个元素是 发送方数量 X 1 的列向量
        """
        channel_select_array = np.zeros((self.v2v_num,self.channel_num))

        for i in range(self.v2v_num):
            channel_select_array[i,int(channel_select[i])] = 1

        channel_select_array = channel_select_array.transpose()
        channel_select_array_3d = np.expand_dims(channel_select_array,2)        


        # 功率控制的处理
        """
        功率控制信息是2维的，实际是列向量，规模是 发送方数量 X 1
        """
        power_array = np.expand_dims(power_list,1)

        # 计算接收能量
        """
        recv_power_all是2维的 chan_num X 接收方数量
        recv_power 是 1维的 接收方数量
        """
        # 每个信道每个接收方收到的能量强度 chan_num X 接收方数量 X 1 该结果备用之后生成obs
        recv_power_all = CSI_times_array @ (power_array * channel_select_array_3d)
        # 加上高斯白噪声，需要注意因为仅选择一个信道可以在此处添加减少计算，否则需要后面再加
        # AGWN = self.N0 * np.ones_like(recv_power_all)
        # recv_power_all += AGWN 
        # 利用信道选择信息进行过滤，仅保留选择的信道，未选择信道能量置为 0   
        recv_power = recv_power_all * channel_select_array_3d
        # 将3d的矩阵降为成2维的
        recv_power_all = np.squeeze(recv_power_all,2)
        recv_power = np.squeeze(recv_power,2)
        recv_power = np.sum(recv_power,axis=0)

        # 计算发送能量
        """
        signal_power1 是2维的 chan_num X 接收方数量,用于之后状态的计算使用
        signal_power 是 1维的 接收方数量   用于后面的互信息计算
        """
        signal_power = np.triu(CSI_times_array)-np.triu(CSI_times_array,k=1)
        signal_power = (signal_power @ (power_array * channel_select_array_3d)) * channel_select_array_3d
        signal_power1 = np.squeeze(signal_power,2)
        signal_power = np.sum(signal_power1,axis=0)

        # 计算信噪比
        Interfere_power = recv_power - signal_power 
        # assert Interfere_power.all() >= self.N0 , "Interfere error less than AGWN is {}".format(Interfere_power)
        AGWN = self.N0 * np.ones_like(Interfere_power)
        Interfere_power += AGWN
        SINR_array = signal_power / Interfere_power

        # 计算互信息
        MI = self.band_width * np.log2(1+SINR_array)

        MI = MI/1000 * self.slot       # 单位是bit/slot 
        AGWN = self.N0 * np.ones_like(recv_power_all)
        # recv_power_all += AGWN
        return MI,(recv_power_all-signal_power1)+AGWN
    
    def step(self, action_dict):
        channel_select = []
        power_list = []
        mode_list = []        # mode=1 表示v2v工作模式  mode=0 表示v2i工作模式
        for agent in range(self.v2v_num):
            mode, chan_select, power = self.action_decode(action_dict[agent])   # 需要输出的功率转化为单位瓦特了
            channel_select.append(chan_select)
            mode_list.append(mode)
            power_list.append(power)

        power_list = np.array(power_list)
        CSI_array = self.chan.get_CSI_array() * (-1) 
        CSI_times_array = db2time(CSI_array)
        [v2v_CSI,v2i_CSI] = np.split(CSI_times_array, 2, axis=1)

        v2v_MI, Interference = self.get_feedback(channel_select,power_list,v2v_CSI)
        v2i_MI , Interference_R = self.get_feedback(channel_select,power_list,v2i_CSI)
        mode = np.array(mode_list)
        v2v_MI = v2v_MI * mode
        v2i_MI = v2i_MI * (1-mode)

        self.aoi =  self.aoi + self.slot
        self.aoi[v2i_MI>=self.pre_v2i_threshold] = self.slot
        self.payload -= v2v_MI
        self.payload[self.payload<0] = 0
        self.timeleft -= self.slot

        self.aoi_count += self.aoi
        
        reward = -1.0 * power_list * self.power_weight                               # 功率分量

        # reward -= (self.aoi - self.slot)                                          # aoi分量
        
        # reward -= self.payload/self.pre_v2v_payload * self.v2v_weight                # V2V分量
        
        # v2i_reward = np.ones_like(v2i_MI)
        # v2i_reward[v2i_MI<self.pre_v2i_threshold] = 0
        # reward += v2i_reward * self.v2i_weight                                       # V2i分量
        
        # Interference_R = time2db(Interference_R)
        # global_rew = np.sum(Interference_R) * -1 * self.global_weight 


        if self.timeleft==0:
            temp = np.zeros_like(self.payload)
            temp[self.payload>0] = 1
            self.success += temp
            aoi = self.aoi_count / self.period * self.slot 
            temp = np.zeros_like(aoi)
            temp[aoi>self.aoi_requirement] = 1
            terminated = {i:True for i in self.agents}

        else:
            terminated = {i:False for i in self.agents}
            temp = np.zeros_like(self.aoi)

        rew= {i:reward[i] for i in self.agents}
        truncated = {i:False for i in self.agents}
        
        info = {i:{'cost':np.array([power_list[i],self.success[i],temp[i],self.aoi[i]])} for i in range(self.v2v_num)}   
            
        

        CSI_array = self.chan.get_CSI_array() 
        [v2v_CSI,v2i_CSI] = np.split(CSI_array, 2, axis=1)
        v2v_CSI = np.triu(v2v_CSI)-np.triu(v2v_CSI,k=1)
        v2i_CSI = np.triu(v2i_CSI)-np.triu(v2i_CSI,k=1)
        v2v_CSI = (np.ones((self.v2v_num,),dtype=np.float32) @ v2v_CSI).transpose()
        v2i_CSI = (np.ones((self.v2v_num,),dtype=np.float32) @ v2i_CSI).transpose()
        Interference = time2db(Interference)
        for i in range(self.v2v_num):
            v2v_CSI[i] = (v2v_CSI[i] - v2v_bias_array[i])/3.0/v2v_scale_array[i]
            v2i_CSI[i] = (v2i_CSI[i] - v2i_bias_array[i])/3.0/v2i_scale_array[i]
            Interference[:,i] = (Interference[:,i] - In_bias_array[i])/3.0/In_scale_array[i]
        Interference = np.concatenate((Interference,
                                np.expand_dims(self.payload/self.pre_v2v_payload,0),                    
                                np.array([[self.timeleft/self.period]*self.v2v_num],dtype=np.float32),
                                np.expand_dims(self.aoi/20.0,0)))  
        all_state = Interference.transpose()
        obs = {i:np.concatenate((v2v_CSI[i],v2i_CSI[i],all_state[i])) for i in self.agents}

        self.chan.update_fast_fading()
        if terminated[0]:
            self.agents = []
        
        return obs, rew, terminated, truncated, info

import argparse
import random
import importlib
import supersuit as ss
import numpy as np
from distutils.util import strtobool
def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-envs", type=int, default=10,
        help="the number of parallel game environments")
    parser.add_argument("--num-minibatches", type=int, default=100,
        help="the number of mini-batches")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--total-timesteps", type=int, default=50000000,
        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=2.5e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gamma", type=float, default=1.0,
        help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
        help="the lambda for the general advantage estimation")
    parser.add_argument("--load-model", type=bool, default=False,
        help="begin with an exist model")
    parser.add_argument("--update-epochs", type=int, default=10,
        help="the K epochs to update the policy")
    parser.add_argument("--clip-coef", type=float, default=0.1,
        help="the surrogate clipping coefficient")
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles advantages normalization")
    parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    parser.add_argument("--ent-coef", type=float, default=0.01,
        help="coefficient of the entropy")
    parser.add_argument("--vf-coef", type=float, default=0.5,
        help="coefficient of the value function")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
        help="the maximum norm for the gradient clipping")
    parser.add_argument("--global_begin", type=float, default=0,)
    parser.add_argument("--target-kl", type=float, default=None,
        help="the target KL divergence threshold")
    args = parser.parse_args()
    # fmt: on
    return args

def Heuristic_decision_making(id, state, env_config):
    ''' 
    state是一个[1,15]的向量 将向量的前4项提取存储为V2V_CSI 将4:8项提取为V2I_CSI,
    将8:12项提取为Interference
    将12项提取为payload 将13项提取为lefttime 将14项提取为aoi
    '''
    V2V_CSI = state[:4]
    V2I_CSI = state[4:8]
    Interference = state[8:12]
    payload = state[12]
    lefttime = state[13]
    aoi = state[14]
    
    '''
    信道分配原则 除以信道数量取余 
    由于正好有4个区域的点与信道数量相同 可能导致同一区域信道相同
    因此根据除以4的商再加一次偏移
    '''
    channel_no = ((id % 4) + (id // 4) ) % 4 
    
    # 根据aoi的情况 确定工作任务 1：V2V任务  0：V2I任务    
    # 注意这里需要还原归一化，如果任务变化，此处也需要修改    
    # task_mode = 0 if aoi * 20 > (env_config['aoi_requirement']-(id % 4)) else 1
    task_mode = 0 if aoi * 20 > (env_config['aoi_requirement']) else 1
    
    # 确定需要传输的互信息量
    if task_mode == 0:
        R = env_config['v2i_threshold_per_second_Hz'] / env_config['period'] * env_config['slot'] / env_config['aoi_prob_threshold'] 
    else:
        R = payload * env_config['payload'] / (lefttime * env_config['period']) / env_config['outage_prob_threshold'] / env_config['bandwidth']
    
    # 确定需要的SNR
    SNR = math.pow(2.0,R) - 1
    
    # 计算信噪系数 这里注意需要还原归一化，如果环境设置变化，这里也要修改
    if task_mode == 0:
        # CoverI = 0 - ((V2I_CSI[channel_no] * 34.2)+97.0) - ((Interference[channel_no]*78)-106)
        CoverI = (0 - V2I_CSI[channel_no]  - Interference[channel_no]) - 70
    else:
        # CoverI = 0 - ((V2V_CSI[channel_no] * 19)+44.0) - ((Interference[channel_no]*78)-106 )
        CoverI = (0 - V2V_CSI[channel_no]  - Interference[channel_no]) 
    
    if SNR==0:
        powerlevel = 0
    else:
        # 计算需求的功率 +30是将dbW转化为dbm
        P = time2db(SNR) - CoverI + 30.0 
        # 由计算的功率计算动作等级
        P = P if P <= env_config['max_power_dbm'] else env_config['max_power_dbm']
        P = db2time(P)
        powerlevel = math.ceil(P * (env_config['action_level']-1) / db2time(env_config['max_power_dbm']))
    
    return powerlevel + channel_no * env_config['action_level'] + task_mode * env_config['action_level'] * env_config['channel_num']

def process_state_statistics(data1,data2,data3):
    V2V = np.stack(data1).reshape(1,-1)
    V2I = np.stack(data2).reshape(1,-1)
    Int = np.stack(data3).reshape(1,-1)
    return np.mean(V2V), np.std(V2V), np.mean(V2I), np.std(V2I), np.mean(Int), np.std(Int)

if __name__ == "__main__":
    import time

    start_time = time.time()  # 获取当前时间
    env_config = {
        'agent_num': 8,
        'channel_num':4,
        'payload': 80.0*1060,
        'v2i_threshold_per_second_Hz': 3.0,
        'slot':1.0,
        'period':100.0,
        'AWGN_n0':-114.0,
        'bandwidth':180*1e3,
        'max_power_dbm':30.0,
        'aoi_requirement':8.0,
        'aoi_prob_threshold':0.95,
        'outage_prob_threshold':0.95,
        'power_weight': 150.0,
        'v2v_weight': 1.0,
        'v2i_weight': 1.0,
        'global_weight':1.0,
        'action_level' :8,
        'period_num' : 1,
    }
    args = parse_args()
    args.num_steps = int(env_config['period'] / env_config['slot'] * env_config['period_num'])
    # aoi = []
    # succ = []
    # aoi_succ = []
    # power = []
    env = v2vEnv(env_config)
    env1 = ss.pettingzoo_env_to_vec_env_v1(env)
    envs = ss.concat_vec_envs_v1(env1, args.num_envs, num_cpus=1, base_class="gymnasium")
    envs.single_observation_space = envs.observation_space
    envs.single_action_space = envs.action_space
    envs.is_vector_env = True
    
    print(envs.single_action_space.n)
    envs.close()