from typing import List, Mapping, Optional, Tuple, Union
import math
import numpy as np
import random

def normalized_to_pixel_coordinates(
    normalized_x: float, normalized_y: float, image_width: int,
    image_height: int) -> Union[None, Tuple[int, int]]:
  """Converts normalized value pair to pixel coordinates."""

  # Checks if the float value is between 0 and 1.
  def is_valid_normalized_value(value: float) -> bool:
    return (value > 0 or math.isclose(0, value)) and (value < 1 or
                                                      math.isclose(1, value))

  if not (is_valid_normalized_value(normalized_x) and
          is_valid_normalized_value(normalized_y)):
    # TODO: Draw coordinates even if it's outside of the image bounds.
    return None
  x_px = min(math.floor(normalized_x * image_width), image_width - 1)
  y_px = min(math.floor(normalized_y * image_height), image_height - 1)
  return x_px, y_px

def calculate_segment_length(pos_x, from_joint: int, to_joint: int):
   offset_joint = pos_x[to_joint:to_joint+1, :] - pos_x[from_joint:from_joint+1, :]
   return np.linalg.norm(offset_joint)

def calculate_normalized_offset(pos_x, from_joint: int, to_joint: int, normal_rate = 1):
    """ The function takes in the index of two joints and returns the offset that the to_joint need
    move so that the lenght of the two joints is 1
    
    :param pos_x: the position of the joints
    :param from_joint: the index of the joint that the offset is calculated from, defaults to 0
    :param to_joint: the joint that we want to move, defaults to 1
    :return: The normalized offset of the joint.
    """
    
    #offset_joint = pos_x[:, to_joint:to_joint+1, :] - pos_x[:, from_joint:from_joint+1, :]
    offset_joint = pos_x[to_joint:to_joint+1, :] - pos_x[from_joint:from_joint+1, :]
    normalized_joint_offset = offset_joint / (np.linalg.norm(offset_joint, axis=-1, keepdims=True)+1e-8) * normal_rate - offset_joint
    return normalized_joint_offset

def min_max_normalize(data):
    data_array = np.array(data)
    normalized_data = data_array / np.min(data_array)
    return np.round(normalized_data,5)

def shoulder_normalize(data, shoulder_dis):
   np_data = np.array(data)
   normalized_data = np_data / (shoulder_dis+1e-8)
   return normalized_data

def normalize_all_joints(pos_x_orig: np.ndarray) -> np.ndarray:
    # normalize all the joints lengths to 1
    # first, move all the points along the direction of the first point to the second point with a fixed length 1
    # second, move thumb (1, 2) (2, 3) (3, 4) (4, 5) to a fixed length 1
    new_pos_x = pos_x_orig.copy()
    # pos_x = [b, 21, 3]
    
    normalized_thumb_offset_0 = calculate_normalized_offset(new_pos_x, from_joint=0, to_joint=1,normal_rate= 1.0)
    normalized_thumb_offset_1 = calculate_normalized_offset(new_pos_x, from_joint=1, to_joint=2,normal_rate= 0.25)
    normalized_thumb_offset_2 = calculate_normalized_offset(new_pos_x, from_joint=2, to_joint=3,normal_rate= 0.25)
    normalized_thumb_offset_3 = calculate_normalized_offset(new_pos_x, from_joint=3, to_joint=4,normal_rate= 0.25)
    
    normalized_index_offset_0 = calculate_normalized_offset(new_pos_x, from_joint=0, to_joint=5,normal_rate=1.25)
    normalized_index_offset_1 = calculate_normalized_offset(new_pos_x, from_joint=5, to_joint=6,normal_rate=0.3125)
    normalized_index_offset_2 = calculate_normalized_offset(new_pos_x, from_joint=6, to_joint=7,normal_rate=0.3125)
    normalized_index_offset_3 = calculate_normalized_offset(new_pos_x, from_joint=7, to_joint=8,normal_rate=0.3125)
    
    normalized_middle_offset_0 = calculate_normalized_offset(new_pos_x, from_joint=0, to_joint=9,normal_rate=1.25)
    normalized_middle_offset_1 = calculate_normalized_offset(new_pos_x, from_joint=9, to_joint=10,normal_rate=0.3125)
    normalized_middle_offset_2 = calculate_normalized_offset(new_pos_x, from_joint=10, to_joint=11,normal_rate=0.3125)
    normalized_middle_offset_3 = calculate_normalized_offset(new_pos_x, from_joint=11, to_joint=12,normal_rate=0.3125)
    
    normalized_ring_offset_0 = calculate_normalized_offset(new_pos_x, from_joint=0, to_joint=13,normal_rate=1.25)
    normalized_ring_offset_1 = calculate_normalized_offset(new_pos_x, from_joint=13, to_joint=14,normal_rate=0.3125)
    normalized_ring_offset_2 = calculate_normalized_offset(new_pos_x, from_joint=14, to_joint=15,normal_rate=0.3125)
    normalized_ring_offset_3 = calculate_normalized_offset(new_pos_x, from_joint=15, to_joint=16,normal_rate=0.3125)


    normalized_pinky_offset_0 = calculate_normalized_offset(new_pos_x, from_joint=0, to_joint=17,normal_rate= 1.0)
    normalized_pinky_offset_1 = calculate_normalized_offset(new_pos_x, from_joint=17, to_joint=18,normal_rate=0.25)
    normalized_pinky_offset_2 = calculate_normalized_offset(new_pos_x, from_joint=18, to_joint=19,normal_rate=0.25)
    normalized_pinky_offset_3 = calculate_normalized_offset(new_pos_x, from_joint=19, to_joint=20,normal_rate=0.25)

    
    # move the thumb from the tip to the base
    new_pos_x[4:5, :] += normalized_thumb_offset_3 + normalized_thumb_offset_2 + normalized_thumb_offset_1 + normalized_thumb_offset_0
    new_pos_x[3:4, :] += normalized_thumb_offset_2 + normalized_thumb_offset_1 + normalized_thumb_offset_0
    new_pos_x[2:3, :] += normalized_thumb_offset_1 + normalized_thumb_offset_0
    new_pos_x[1:2, :] += normalized_thumb_offset_0
    
    # move the index from the tip to the base
    new_pos_x[8:9, :] += normalized_index_offset_3 + normalized_index_offset_2 + normalized_index_offset_1 + normalized_index_offset_0
    new_pos_x[7:8, :] += normalized_index_offset_2 + normalized_index_offset_1 + normalized_index_offset_0
    new_pos_x[6:7, :] += normalized_index_offset_1 + normalized_index_offset_0
    new_pos_x[5:6, :] += normalized_index_offset_0
    
    # move the middle from the tip to the base
    new_pos_x[12:13, :] += normalized_middle_offset_3 + normalized_middle_offset_2 + normalized_middle_offset_1 + normalized_middle_offset_0
    new_pos_x[11:12, :] += normalized_middle_offset_2 + normalized_middle_offset_1 + normalized_middle_offset_0
    new_pos_x[10:11, :] += normalized_middle_offset_1 + normalized_middle_offset_0
    new_pos_x[9:10, :] += normalized_middle_offset_0   
    # move the ring from the tip to the base
    new_pos_x[16:17, :] += normalized_ring_offset_3 + normalized_ring_offset_2 + normalized_ring_offset_1 + normalized_ring_offset_0
    new_pos_x[15:16, :] += normalized_ring_offset_2 + normalized_ring_offset_1 + normalized_ring_offset_0
    new_pos_x[14:15, :] += normalized_ring_offset_1 + normalized_ring_offset_0
    new_pos_x[13:14, :] += normalized_ring_offset_0
    
    # move the pinky from the tip to the base
    new_pos_x[20:21, :] += normalized_pinky_offset_3 + normalized_pinky_offset_2 + normalized_pinky_offset_1 + normalized_pinky_offset_0
    new_pos_x[19:20, :] += normalized_pinky_offset_2 + normalized_pinky_offset_1 + normalized_pinky_offset_0
    new_pos_x[18:19, :] += normalized_pinky_offset_1 + normalized_pinky_offset_0
    new_pos_x[17:18, :] += normalized_pinky_offset_0

    return new_pos_x
# input shape (6,3)

def normalize_body_joints(joints:List):
    """
      norm 11~ 16 joints from left wrist to right wrist in mediapipe body landmark
      choose the middle point between left shoulder and right shoulder as reference point
      norm the distance between joints to 1 
      0 is left_shoulder
      1 is right_shoulder
      2 is left_elbow
      3 is right_elbow
      4 is left_wrist
      5 is right_wrist
      6 is middle point whereas mid of 0 and 1 
    """
    middle_point = []
    for n1,n2 in zip(joints[0],joints[1]):
      middle_point.append((n1+n2)/2)
    joints.append(middle_point)
    new_joints = np.array(joints)
    left_shoulder_dis = calculate_segment_length(new_joints,from_joint=6,to_joint=0)
    right_shoulder_dis = calculate_segment_length(new_joints,from_joint=6,to_joint=1)
    left_elbow_dis = calculate_segment_length(new_joints,from_joint=0,to_joint=2)
    right_elbow_dis = calculate_segment_length(new_joints,from_joint=1,to_joint=3)
    left_wrist_dis = calculate_segment_length(new_joints,from_joint=2,to_joint=4)
    right_wrist_dis = calculate_segment_length(new_joints,from_joint=3,to_joint=5)
    norm_list = shoulder_normalize([left_shoulder_dis,right_shoulder_dis,left_elbow_dis,right_elbow_dis,left_wrist_dis,right_wrist_dis],left_shoulder_dis/2)

    norm_left_shoulder = calculate_normalized_offset(new_joints,from_joint=6,to_joint=0,normal_rate=norm_list[0])
    norm_right_shoulder = calculate_normalized_offset(new_joints,from_joint=6,to_joint=1,normal_rate=norm_list[1])
    norm_left_elbow = calculate_normalized_offset(new_joints,from_joint=0,to_joint=2,normal_rate=norm_list[2])
    norm_right_elbow = calculate_normalized_offset(new_joints,from_joint=1,to_joint=3,normal_rate=norm_list[3])
    norm_left_wrist = calculate_normalized_offset(new_joints,from_joint=2,to_joint=4,normal_rate=norm_list[4])
    norm_right_wrist = calculate_normalized_offset(new_joints,from_joint=3,to_joint=5,normal_rate=norm_list[5])

    

    new_joints[0:1,:] += norm_left_shoulder
    new_joints[1:2,:] += norm_right_shoulder
    new_joints[2:3,:] += norm_left_shoulder + norm_left_elbow
    new_joints[3:4,:] += norm_right_shoulder + norm_right_elbow
    new_joints[4:5,:] += norm_left_shoulder + norm_left_elbow + norm_left_wrist
    new_joints[5:6,:] += norm_right_shoulder + norm_right_elbow + norm_right_wrist

    return new_joints[:-1]

def shift_body2orgin(joints:List):
    joints = np.array(joints)
    middle_point = (joints[0]+joints[1])/2
    new_joints = joints - middle_point
    return new_joints