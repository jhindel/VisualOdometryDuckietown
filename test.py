import numpy as np

# simple transformation of going 1 in y, then turning 90 degrees and going 1 in x
transform1 = np.asarray([[0, -1, 0], [1, 0, 1], [0, 0, 1]])
transform2 = np.asarray([[1, 0, 1], [0, 1, 0], [0, 0, 1]])

transform3 = np.asarray([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
transform4 = np.asarray([[1, 0, 0], [0, 1, 1], [0, 0, 1]])

dot = np.dot(transform1, transform2)

dot2 = np.dot(transform3, transform4)

print(transform1)
print(transform2)
print(dot)
print(dot2)


# simple transformation of going 1 in y, then turning 90 degrees and going 1 in x
transform1 = np.asarray([[0, -1, 0], [1, 0, 1], [0, 0, 1]])
transform2 = np.asarray([[1, 0, 1], [0, 1, 0], [0, 0, 1]])

new_pos = np.dot(transform1,(np.asarray([0, 0, 1])))
print(new_pos)
new_pos_2 = np.dot(transform2, new_pos)
print(new_pos_2)