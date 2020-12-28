#include <ros/ros.h>
#include <control_msgs/FollowJointTrajectoryAction.h>
#include <actionlib/client/simple_action_client.h>
#include <iostream>
#include <string>
#include <std_msgs/Float64MultiArray.h>
#include <jointspace/OptStates.h>
#include "sensor_msgs/JointState.h"
#include <Eigen/Eigen>
#include <kdl/jntarray.hpp>
#include <array>
#include <kdl/chaindynparam.hpp>
#include <kdl_parser/kdl_parser.hpp>
#include <kdl/tree.hpp>
#include <kdl/chain.hpp>
#include <kdl/jntspaceinertiamatrix.hpp>
#include <std_msgs/Float64.h>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/time_synchronizer.h>
#include <sensor_msgs/Image.h>
#include <dynamic_reconfigure/server.h>
#include <ros/console.h>
#include <kdl/rotational_interpolation_sa.hpp>
#include <kdl/frames.hpp>
#include <jointspace/OptStatesWt.h>
#include <chrono>

#include <ros_action_server/MyMsgAction.h>



using namespace std;
using namespace std::chrono;

#define N_JOINT 6

typedef actionlib::SimpleActionClient <ros_action_server::MyMsgAction> TrajClient;
//typedef actionlib::SimpleActionClient <control_msgs::FollowJointTrajectoryAction> TrajClient;

//rosmsg show ros_action_server/MyMsgAction.msg

class Ur3Arm
{
private:
  TrajClient* traj_client_;
  std::string tauTopicNames[N_JOINT] = {
    "tau_1",
    "tau_2",
    "tau_3",
    "tau_4",
    "tau_5",
    "tau_6",
  };
  std::string dataTopicNames[12] = {
    "pose_error_1",
    "pose_error_2",
    "pose_error_3",
    "pose_error_4",
    "pose_error_5",
    "pose_error_6",
    "velo_error_1",
    "velo_error_2",
    "velo_error_3",
    "velo_error_4",
    "velo_error_5",
    "velo_error_6",
  };

  message_filters::Subscriber<sensor_msgs::JointState> sub1;
  message_filters::Subscriber<jointspace::OptStatesWt> sub2;

  typedef message_filters::sync_policies::ExactTime<sensor_msgs::JointState, jointspace::OptStatesWt> MySyncPolicy;
  typedef message_filters::Synchronizer<MySyncPolicy> Sync;
  boost::shared_ptr<Sync> sync;

public:
  ros::NodeHandle n;
  //ros::Subscriber opt_states_sub; ros::Subscriber joint_states_sub;
  std::vector<ros::Publisher> pose_multiple_pub;
  std::vector<ros::Publisher> data_multiple_pub;
  ros::Publisher pose_pub;
  ros::Publisher data_pub;
  KDL::JntArray jointPosCurrent, jointVelCurrent, jointEffort;
  Eigen::MatrixXf q_cur, qdot_cur;
  Eigen::MatrixXf q_des, qdot_des, qddot_des;
  KDL::Tree mytree; KDL::Chain mychain;
  std::array<int, N_JOINT> map_joint_states;
  std::array<float, N_JOINT> k_p;
  std::array<float, N_JOINT> k_d;
  std::array<float, N_JOINT> desired_torque;

  Ur3Arm()
  {
    jointPosCurrent.resize(N_JOINT), jointVelCurrent.resize(N_JOINT), jointEffort.resize(N_JOINT);
    q_cur.resize(N_JOINT,1); qdot_cur.resize(N_JOINT,1);
    q_des.resize(N_JOINT, 1), qdot_des.resize(N_JOINT,1), qddot_des.resize(N_JOINT,1);
    map_joint_states={2, 1, 0, 3, 4, 5};
    k_p={10, 12, 20.5,  25,  12,  12};          // specify p and d gains
    k_d={15, 13, 14,  14,  10,  17};
    desired_torque={50.0, -50, 40, 30, 20, 10.0};

    // BEGIN
    float mass = 5 ; double Ixx, Iyy, Izz; double l= 0.08, r = l/2.0; // from URDF

    Ixx = (1.0/12.0) * mass * ( 3*r*r + l*l);
    Iyy = Ixx;
    Izz =  (1.0/2.0) * mass * ( r*r );
    double I[6]={Ixx, Iyy, Izz, 0, 0, 0};   // Ixy = Ixz= Iyz = 0;
    double offset[6] = {0, 0, 0, 0, 0, 0} ; std::string tool_name = "new_tool";
    KDL::Vector r_cog(r, r, l/2.0); //! for a cylinder
    KDL::Joint fixed_joint = KDL::Joint(KDL::Joint::None);
    KDL::Frame tip_frame = KDL::Frame(KDL::Rotation::RPY(offset[0],offset[1],offset[2]),KDL::Vector(offset[3],offset[4],offset[5]));

    // rotational inertia in the cog
    KDL::RotationalInertia Inertia_cog = KDL::RotationalInertia(I[0], I[1], I[2], I[3], I[4], I[5]);
    KDL::RigidBodyInertia Inertia = KDL::RigidBodyInertia(mass, r_cog, Inertia_cog);
    KDL::Segment segment = KDL::Segment(tool_name, fixed_joint, tip_frame, Inertia);

    //parse kdl tree from Urdf
    if(!kdl_parser::treeFromFile("/home/mujib/test_ws/src/universal_robot/ur_description/urdf/ur5_joint_test.urdf", mytree)){
      ROS_ERROR("[ST] Failed to construct kdl tree for elfin ! ");
    }

    if (!mytree.addSegment(segment, "wrist_3_link")) {  //! adding segment to the tree
      ROS_ERROR("[ST] Could not add segment to kdl tree");
    }

    if (!mytree.getChain("base_link", tool_name, mychain)){
      ROS_ERROR("[ST] Failed to construct kdl chain for elfin ! ");
    }
    // END

    unsigned int nj, ns; // resize variables using # of joints & segments
    nj =  mytree.getNrOfJoints(); ns = mychain.getNrOfSegments();
    if (ns == 0 || nj == 0){
      ROS_ERROR("[ST] Number of segments/joints are zero ! ");
    }
    if(jointPosCurrent.rows()!=nj || jointVelCurrent.rows()!=nj || jointEffort.rows() !=nj )
    {
      ROS_ERROR("[JS] ERROR in size of joint variables ! ");
    }

    sub1.subscribe(n, "/urbot/joint_states", 100);   // TODO: urbot/joint_states
    sub2.subscribe(n, "/opt_states", 100);
    sync.reset(new Sync(MySyncPolicy(100), sub1, sub2));
    sync->registerCallback(boost::bind(&Ur3Arm::callback, this, _1, _2));

    for (short int j = 0; j < N_JOINT; j++)
    {
      pose_pub = n.advertise<std_msgs::Float64>(tauTopicNames[j], 1);
      pose_multiple_pub.push_back(pose_pub);
    }

    for (short int j = 0; j < 12; j++)
    {
      data_pub = n.advertise<std_msgs::Float64>(dataTopicNames[j], 1);
      data_multiple_pub.push_back(data_pub);
    }

    traj_client_ = new TrajClient("trajectory_action", true); // spin a thread by default

    while(!traj_client_->waitForServer(ros::Duration(5.0)))
     {
       ROS_INFO("[ST] Waiting for the ros_action server");
     }

    if(traj_client_->isServerConnected()){
      ROS_INFO("[ST] ros_action server is Connected");}
    else {
      ROS_INFO("[ST] ros_action server is Not Connected !");
    }

    cout << "[ST] Constructed !" << endl;
  }

  // -> Destructor of class Ur3Arm
  ~Ur3Arm()
  {
    delete traj_client_;
  }

  void callback(const sensor_msgs::JointStateConstPtr &msg1, const jointspace::OptStatesWtConstPtr &msg2)
  {
    KDL::JntArray C(N_JOINT), gravity(N_JOINT);
    Eigen::MatrixXf qdotdot_k(N_JOINT,1), c(N_JOINT,1), g(N_JOINT,1), M_(N_JOINT, N_JOINT), tau(N_JOINT,1);
    KDL::JntSpaceInertiaMatrix M(N_JOINT);
    KDL::ChainDynParam dyn_param(mychain, KDL::Vector(0, 0, -9.80665));
    std::vector<std_msgs::Float64> msg_(N_JOINT);
    std::vector<std_msgs::Float64> msg_error(12);
    Eigen::MatrixXf position_error(N_JOINT,1);  Eigen::MatrixXf velocity_error(N_JOINT,1);

    for(short int k = 0; k < msg2->goal.size(); k++){
      for (short int l=0; l< N_JOINT; l++){

        q_des(l,0) =  msg2->goal[k].q.data[l];          // opt_states
        qdot_des(l,0) = msg2->goal[k].qdot.data[l];
        qddot_des(l,0) = msg2->goal[k].qddot.data[l];

        jointPosCurrent(l) = msg1->position[map_joint_states[l]]; // joint_state
        jointVelCurrent(l) = msg1->velocity[map_joint_states[l]];
        jointEffort(l) = msg1->effort[map_joint_states[l]];
        q_cur(l,0) = msg1->position[map_joint_states[l]];
        qdot_cur(l,0) = msg1->velocity[map_joint_states[l]];

        qdotdot_k(l,0)=k_p[l]*(q_cur(l,0)-q_des(l,0))+k_d[l]*(qdot_cur(l,0)-qdot_des(l,0)) + qddot_des(l,0);
        //qdotdot_k(l,0)=0;
      }

      dyn_param.JntToMass(jointPosCurrent, M);
      dyn_param.JntToGravity(jointPosCurrent, gravity);
      dyn_param.JntToCoriolis(jointPosCurrent, jointVelCurrent, C);

      for(short int i = 0; i < N_JOINT; i++){
        c(i,0) = C(i)*qdot_cur(i,0);
        g(i,0) = gravity(i);
        for(int j = 0; j < N_JOINT; j++){
          M_(i,j) = M(i,j);}
      }

      position_error = q_des;//-q_cur;
      velocity_error = qdot_des;//-qdot_cur;

      tau = M_ * qdotdot_k + c + g;
      //cout << "Applid tau: " << tau.transpose() << ", Current tau:" << jointEffort.data.transpose() << endl;

      /*cout << "q_cur: " << q_cur.transpose() << endl;
      cout << "q_des: " << q_des.transpose() << endl;
      cout << "qdotdot_k: " << qdotdot_k.transpose() << endl;
      cout << "Position Error: " << position_error.transpose() << endl;*/

      for (short int j = 0; j < N_JOINT; j++)
      {
        msg_[j].data = tau(j,0);
        pose_multiple_pub[j].publish(msg_[j]);  // publish torques to plot
      }

      for (short int j = 0; j < N_JOINT; j++)
      {
        msg_error[j].data = position_error(j,0);
        msg_error[j+N_JOINT].data = velocity_error(j,0);
        data_multiple_pub[j].publish(msg_error[j]);                 // publish position error to plot
        data_multiple_pub[j+N_JOINT].publish(msg_error[j+N_JOINT]); // publish velocity error to plot
      }

      ros_action_server::MyMsgGoal action;

      //action.tau.layout.dim.resize(N_JOINT); // resize goal
      action.tau.data.resize(N_JOINT);

      for(size_t i = 0; i < N_JOINT; i++)
      {
        action.tau.data[i] = tau(i,0); // publish torques to controller
      }

      action.header.frame_id = "base_link";
      action.header.stamp = ros::Time::now() + ros::Duration(0.01); // TODO: 0.001
      traj_client_->sendGoalAndWait(action, ros::Duration(0,0), ros::Duration(0,0));


      /*control_msgs::FollowJointTrajectoryGoal goal;
      goal.trajectory.joint_names.push_back("shoulder_pan_joint");
      goal.trajectory.joint_names.push_back("shoulder_lift_joint");
      goal.trajectory.joint_names.push_back("elbow_joint");
      goal.trajectory.joint_names.push_back("wrist_1_joint");
      goal.trajectory.joint_names.push_back("wrist_2_joint");
      goal.trajectory.joint_names.push_back("wrist_3_joint");
      goal.trajectory.points.resize(1); // important

      goal.trajectory.points[0].effort.resize(N_JOINT);
      goal.trajectory.points[0].positions.resize(N_JOINT);
      goal.trajectory.points[0].velocities.resize(N_JOINT);
      goal.trajectory.points[0].accelerations.resize(N_JOINT);

      for(size_t i = 0; i < N_JOINT; i++)
      {
        goal.trajectory.points[0].effort[i] = tau(i,0); // publish torques to controller
        goal.trajectory.points[0].positions[i] = q_des(i,0);
        goal.trajectory.points[0].velocities[i] = 0;//qdot_des(i,0);
        goal.trajectory.points[0].accelerations[i] = 0;//qddot_des(i,0); //qdotdot_k(i,0)
      }

      goal.trajectory.points[0].time_from_start = ros::Duration(1.0); // TODO: time from start ?
      //Function done, return the goal
      goal.trajectory.header.frame_id = "base_Link";
      goal.trajectory.header.stamp = ros::Time::now()+ ros::Duration(0.001); //0.001
      traj_client_->sendGoalAndWait(goal, ros::Duration(0,0), ros::Duration(0,0));*/

    } //!for loop
  }   //!callback

  actionlib::SimpleClientGoalState getState() //! Returns the current state of the action
   {
     return traj_client_->getState();
   }
};


int main (int argc, char** argv)
{
  // Init the ROS node
  ros::init(argc, argv, "simple_trajectory");
  cout << "[AC] Hello World !" << endl;
  Ur3Arm arm;  ros::Rate loop_rate(50);

  while(ros::ok())
  {
    cout << "[AC] STATE: " << arm.getState().toString() << endl;
    ros::spinOnce();
    loop_rate.sleep();
  }

  return 0;
}
