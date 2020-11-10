# Master Thesis
This is a master thesis for the robotic program at MDH.

For compatibility reasons - Ubuntu 16 was used, along with Gazebo 7.16, ROS kinetik and python 2.7.
                            TensorFlow2 uses python 3.
                            
- The simulation files can be found under Gazebo/thesis_ws (where thesis_ws is the whole environment).

- The CNN files can be found under Python/CNN and Q-Learning files under Python/QLearning.

Main files: 

- Python/CNN/server.py - The server side classifies the faces with the trained model (Choose model from Python/CNN/Models and train CNN with trainCNN.py/smallCNN.py)

- Python/QLearning/QLearning_2Persons.py - Training of Q-Learning algorithm with positive and negative person.

- Gazebo/thesis_ws/src/add_jeff/launch/start_test_world.launch - launches world with command "roslaunch add_jeff start_test_world.launch"

--  to initialize the camera use command "rosrun image_view image_view image:=/my_vehicle/camera/image_raw"

- Gazebo/thesis_ws/src/add_jeff/src/ --  to initialize client side use command "rosrun add_jeff listener2.py"
                                                              


                            


