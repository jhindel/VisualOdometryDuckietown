import rosbag

bag = rosbag.Bag('alex_1big_8.bag')

for topic, msg, t in bag.read_messages(topics=['/alex/camera_node/camera_info']): #['distortion_model', 'D', 'K']
    print(f" K: {msg.K}, \n D: {msg.D}, \n distorsion model: {msg.distortion_model}")
    print(msg)
    break
bag.close()
