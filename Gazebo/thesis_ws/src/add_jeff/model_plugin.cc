#include <functional>
#include <gazebo/gazebo.hh>
#include <gazebo/physics/Joint.hh>
#include <gazebo/common/common.hh>
#include <gazebo/physics/physics.hh>
#include <gazebo/physics/Model.hh>
#include <gazebo/physics/PhysicsTypes.hh>
#include <gazebo/transport/transport.hh>
#include <gazebo/msgs/msgs.hh>
#include <thread>
#include "ros/ros.h"
#include "ros/callback_queue.h"
#include "ros/subscribe_options.h"
#include "std_msgs/Float32.h"


namespace gazebo
{
	class WheelController : public ModelPlugin
	{

		public: void Load(physics::ModelPtr _parent, sdf::ElementPtr _sdf)
		{
			//Store the pointer to the model
			this->model = _parent;

			// Listen to the update event. This event is broadcast every simulation iteration
			this->updateConnection = event::Events::ConnectWorldUpdateBegin(std::bind(&WheelController::OnUpdate, this));

			this->old_secs = ros::Time::now().toSec();

			std::cerr << "\nThe WheelController is attached to model[" << _parent->GetName() << "]\n";


			if(_sdf->HasElement("wheel_kp"))
				this->wheel_kp = _sdf->Get<double>("wheel_kp");
			if(_sdf->HasElement("wheel_ki"))
				this->wheel_ki = _sdf->Get<double>("wheel_ki");
			if(_sdf->HasElement("wheel_kd"))
				this->wheel_kd = _sdf->Get<double>("wheel_kd");



			// Create a topic Name
			std::string left_wheel_speed = "/" + this->model->GetName() + "/vel_cmd/left_wheel_speed";
			std::string right_wheel_speed = "/" + this->model->GetName() + "/vel_cmd/right_wheel_speed";

			
			// Create ROS node,
			this->rosNode.reset(new ros::NodeHandle("gazebo_client"));

			const auto &jointController = this->model->GetJointController();
			jointController->Reset();

			
			jointController->AddJoint(model->GetJoint("right_wheel_motor")); // Right Wheel
			jointController->AddJoint(model->GetJoint("left_wheel_motor")); // Left Wheel


			this->right_wheel_name = model->GetJoint("right_wheel_motor")->GetScopedName();// Right Wheel
			this->left_wheel_name = model->GetJoint("left_wheel_motor")->GetScopedName();// Left Wheel

			jointController->SetVelocityPID(this->right_wheel_name, common::PID(this->wheel_kp, this->wheel_ki, this->wheel_kd));
			jointController->SetVelocityPID(this->left_wheel_name, common::PID(this->wheel_kp, this->wheel_ki, this->wheel_kd));

			jointController->SetVelocityTarget(this->right_wheel_name, 0.0);
			jointController->SetVelocityTarget(this->left_wheel_name, 0.0);
			
			
			
			// Create a named topic and subscribe to it.
			ros::SubscribeOptions so = ros::SubscribeOptions::create<std_msgs::Float32>(left_wheel_speed, 
				1, boost::bind(&WheelController::OnRosMsg_left_wheel_speed, this, _1), ros::VoidPtr(), &this->rosQueue);
			this->rosSub = this->rosNode->subscribe(so);

			this->rosQueueThread = std::thread(std::bind(&WheelController::QueueThread, this));


			// Create a named topic2 and subscribe to it.
			ros::SubscribeOptions so2 = ros::SubscribeOptions::create<std_msgs::Float32>(right_wheel_speed, 
				1, boost::bind(&WheelController::OnRosMsg_right_wheel_speed, this, _1), ros::VoidPtr(), &this->rosQueue2);
			this->rosSub2 = this->rosNode->subscribe(so2);

			this->rosQueueThread2 = std::thread(std::bind(&WheelController::QueueThread2, this));


		}

		public: void OnUpdate() // Send Vector3d
		{
			double new_secs = ros::Time::now().toSec();
			double delta = new_secs - this->old_secs;

			double max_delta = 0.0;

			if(this->freq_update != 0.0){
				max_delta = 1.0 / this->freq_update;
			}

			if(delta > max_delta && delta != 0.0){
				this->old_secs = new_secs;
			}

			// Debug
			ROS_DEBUG("Update Wheel Speed PID...");
			const auto &jointController = this->model->GetJointController();
			jointController->SetVelocityTarget(this->right_wheel_name, this->right_wheel_speed_magn);
			jointController->SetVelocityTarget(this->left_wheel_name, this->left_wheel_speed_magn);
	
		}


		public: void SetLeftWheelSpeed(const double &_freq)
		{
			this->left_wheel_speed_magn = _freq;
		}

		public: void SetRightWheelSpeed(const double &_magn)
		{
			this->right_wheel_speed_magn = _magn;
		}

		// Sets speed on Righ wheel with given data
		public: void OnRosMsg_left_wheel_speed(const std_msgs::Float32ConstPtr &_msg)
		{
			this->SetLeftWheelSpeed(_msg->data);
		}

		// Ros helper function to process messages
		private: void QueueThread()
		{
			static const double timeout = 0.01;
			while (this->rosNode->ok())
			{
				this->rosQueue.callAvailable(ros::WallDuration(timeout));
			}
		}


		// Sets speed on Righ wheel with given data
		public: void OnRosMsg_right_wheel_speed(const std_msgs::Float32ConstPtr &_msg)
		{
			this->SetRightWheelSpeed(_msg->data);
		}

		
		// Ros helper function to process messages2
		private: void QueueThread2()
		{
			static const double timeout = 0.01;
			while (this->rosNode->ok())
			{
				this->rosQueue2.callAvailable(ros::WallDuration(timeout));
			}
		}


		// Pointer to the update event connection
		private: event::ConnectionPtr updateConnection;

		// Time memory
		double old_secs;
		double freq_update = 10.0;

		// Magnitude of the oscillations
		double left_wheel_speed_magn = 0.0;
		double right_wheel_speed_magn = 0.0;

		// Pointer to the model
		private: physics::ModelPtr model;

		/// A node use for ROS transport
		private: std::unique_ptr<ros::NodeHandle> rosNode;

		/// A ROS subscriber
		private: ros::Subscriber rosSub;

		/// A ROS callbackqueue that helps process messages
		private: ros::CallbackQueue rosQueue;

		/// A thread the keeps running the rosQueue
		private: std::thread rosQueueThread;


		/// A ROS subscriber 2
		private: ros::Subscriber rosSub2;

		/// A ROS callbackqueue that helps process messages 2
		private: ros::CallbackQueue rosQueue2;

		/// A thread the keeps running the rosQueue2
		private: std::thread rosQueueThread2;

		std::string right_wheel_name;
		std::string left_wheel_name;

		double wheel_kp = 0.1;
		double wheel_ki = 0.0;
		double wheel_kd = 0.0;
		
		
	};

	// Register this plugin with the simulator
	GZ_REGISTER_MODEL_PLUGIN(WheelController)
}
		
