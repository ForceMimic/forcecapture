import numpy as np

L515_2_BASE = np.array([[1,0,0,0],
                        [0,-np.sin(70 / 180 * np.pi),np.cos(70 / 180 * np.pi),0],
                        [0,-np.cos(70 / 180 * np.pi),-np.sin(70 / 180 * np.pi),0.59],
                        [0,0,0,1]])
L515_2_T265l = np.array([[1,0,0,8.43/1000], 
                        [0,-np.cos(70 / 180 * np.pi),-np.sin(70 / 180 * np.pi),30.12/1000], 
                        [0,np.sin(70 / 180 * np.pi),-np.cos(70 / 180 * np.pi),-95.66/1000], 
                        [0,0,0,1]])
T265l_2_L515 = np.linalg.inv(L515_2_T265l)
T265l_2_BASE = L515_2_BASE @ T265l_2_L515
T265r_2_T265l = np.array([[1,0,0,0], 
                        [0,1,0,-32/1000], 
                        [0,0,1,0], 
                        [0,0,0,1]])
T265r_2_L515 = T265l_2_L515 @ T265r_2_T265l
L515_2_T265r = np.linalg.inv(T265r_2_L515)
T265r_2_BASE = L515_2_BASE @ T265r_2_L515
T265l_2_GRIPPER = np.array([[-1,0,0,0],
                            [0,np.cos(45 / 180 * np.pi),-np.sin(45 / 180 * np.pi),38.53/1000],
                            [0,-np.sin(45 / 180 * np.pi),-np.cos(45 / 180 * np.pi),-19.47/1000],
                            [0,0,0,1]])
T265r_2_PYFT = np.array([[-1,0,0,0],
                        [0,np.cos(45 / 180 * np.pi),-np.sin(45 / 180 * np.pi),81.81/1000],
                        [0,-np.sin(45 / 180 * np.pi),-np.cos(45 / 180 * np.pi),-19.5/1000],
                        [0,0,0,1]])

ANGLE_2_WIDTH = 17 * 3.14 / 1000.0 * 2 / 360.0

OBJECT_SPACE = ((-0.15, 0.1), (0.135, 0.385), (0.035, 0.335))
GRIPPER_SPACE = ((-0.095-0.01, 0.095+0.01), (-0.0165-0.005, 0.0165+0.005), (0.-0.005, 0.12+0.005))
PYFT_SPACE = ((-0.035-0.005, 0.035+0.005), (-0.035-0.005, 0.035+0.005), (0.-0.005, 0.14+0.005))
BASE_SPACE = ((-0.4, 0.4), (0.035, 0.585), (0.035, 0.835))
