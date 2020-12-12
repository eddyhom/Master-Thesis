# Master Thesis
This is a master thesis for the robotic program at MDH.

For compatibility reasons - Ubuntu 16 was used, along with Gazebo 7.16, ROS kinetik and python 2.7.
                            TensorFlow2 uses python 3.
                            
- The simulation files can be found under Gazebo/thesis_ws (where thesis_ws is the whole environment).
- The CNN files can be found under Python/CNN and Q-Learning files under Python/QLearning.

Main files: 
- Python/CNN/server.py - The server side classifies the faces with the trained model (Choose model from Python/CNN/Models and train CNN with trainCNN.py/smallCNN.py)
- Python/QLearning/QLearning_2Persons.py - Training of Q-Learning algorithm with positive and negative person.
- Gazebo/thesis_ws/src/add_jeff/launch/start_test_world.launch - contains Gazebo world information.
- Gazebo/thesis_ws/src/add_jeff/src/listener2.py - Controls the environment with ROS and communicates with the server.

Running simulation:
1- Launch server side using python3. $ python3 ../../server.py (takes some time wait until it starts)
2- Launch Gazebo world from Ubuntu 16 terminal. $ roslaunch add_jeff start_test_world.launch
3- Initialize camera from another ubuntu terminal. $ rosrun image_view image_view image:=/my_vehicle/camera/image_raw
4- Finally initialize the client side in ubuntu. $ rosrun add_jeff listener2.py

Flow of simulation.
1-Initialize Server since it might take several seconds to load the libraries and the models.
2-Launch Gazebo world.
3-Initialize camera mounted on robot to be able to classify persons' emotions, program starts immediately collecting images and saving them in given directory.
  --Define directory to save images in launched world. -Gazebo/thesis_ws/src/add_jeff/worlds/*.world under:
        <link name='camera_link'>
        ...
          <sensor name='camera1' type='camera'>
            <camera name='head'>
              <save enabled='1'>
                <path>/home/peter/thesis_ws/src/add_jeff/src/camera_save_tutorial</path> <---- DEFINE HERE
              </save>
              ...
            </camera>
          ...
          </sensor>
        </link>
4- Run listener2.py. It connects all the programs.
   I- It creates a ROS Publisher to control the velocity of the robot, which finally controls the navigation.
   II - It creates a Socket to connect to the Server side.
   III - It creates 2 Subscribers one to know the persons positions and the other one to know the robots odometry.
   IV - Check if two faces have been found, if not find them (Adapted for only two static people, then locks their location).
        -Faces are collected from the directory where they were saved in Step 3. Later sent to server. 
        -Server replaces image from positive person with one from - Python/CNN/DB/Test/happy and Master-Thesis/Python/CNN/DB/Test/fear_sadness for a negative one.
        -Later this image is fed into the model and the emotion(s) found are returned in the following way:
              emotions = {'1': "Positive", '2': "Negative", '0': "NoFace", '12': "Positive-Negative", '21': "Negative-Positive"}
         Positive-Negative means positive person is to the left of negative, and Negative-Positive is the opposite.
         -When faces have been found the QLearning algorithm takes over.
         -Finally when robot is sufficiently close to the target it stops and displays an image with the path taken.
                                                              


                            


