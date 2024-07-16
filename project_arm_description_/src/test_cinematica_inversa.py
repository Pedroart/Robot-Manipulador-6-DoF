#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import JointState
from markers import *
from lab4functions import *

if __name__ == '__main__':

    rospy.init_node("testInvKine")
    pub = rospy.Publisher('joint_states', JointState, queue_size=10)

    bmarker      = BallMarker(color['RED'])
    bmarker_des  = BallMarker(color['GREEN'])

    # Joint names
    jnames = ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6', 'joint_7']

    # Desired position
    xd = np.array([0.80, -0.3, 0.20]) # posicion del eflector final
    # Initial configuration
    q0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ,0.0]) # posicion de la configuracion deseada
    # Inverse kinematics
    q = ikine_ur5(xd, q0)

    # Resulting position (end effector with respect to the base link)
    T = fkine_7dof(q)
    print('Obtained value:\n', np.round(T,3))

    # Red marker shows the achieved position
    bmarker.xyz(T[0:3,3])
    # Green marker shows the desired position
    bmarker_des.xyz(xd)

    # Objeto (mensaje) de tipo JointState
    jstate = JointState()
    # Asignar valores al mensaje
    jstate.header.stamp = rospy.Time.now()
    jstate.name = jnames
    # Add the head joint value (with value 0) to the joints
    jstate.position = q

    # Loop rate (in Hz)
    rate = rospy.Rate(10)
    
    # Continuous execution loop
    while not rospy.is_shutdown():
        # Current time (needed for ROS)
        jstate.header.stamp = rospy.Time.now()
        # Publish the message
        pub.publish(jstate)
        bmarker.publish()
        bmarker_des.publish()
        # Wait for the next iteration
        rate.sleep()
