#!/home/sivan/miniconda3/envs/mujo_ros/bin/python3

import mujoco
 
model = mujoco.MjModel.from_xml_path("model/K1/pgc50final/meshes/pgc_50_35.urdf")
 
mujoco.mj_saveLastXML("model/K1/pgc50final/meshes/pgc_50_35.xml",model)

