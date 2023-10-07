.. _benchmark:

Benchmarks
============

We conducted experiments in both the `Bullet-Safety-Gym <https://github.com/liuzuxin/Bullet-Safety-Gym>`_ and the `SafetyGymnasium <https://github.com/OmniSafeAI/safety-gymnasium>`_ environments.
All the experiments use the default configuration parameters as shown in the ``examples/mlp`` training scripts. The result curves are averaged over 3 seeds.

Bullet-Safety-Gym
-----------------

You can use the drop-down menu to check the reward and cost curves.

.. raw:: html

   <label for="select-bullet-env">Select the environment:</label>
   <select id="select-bullet-env">
     <option value="SafetyCarCircle-v0">SafetyCarCircle-v0</option>
     <option value="SafetyCarRun-v0">SafetyCarRun-v0</option>
     <option value="SafetyDroneCircle-v0">SafetyDroneCircle-v0</option>
     <option value="SafetyDroneRun-v0">SafetyDroneRun-v0</option>
     <option value="SafetyBallCircle-v0">SafetyBallCircle-v0</option>
     <option value="SafetyBallRun-v0">SafetyBallRun-v0</option>
     <option value="SafetyAntRun-v0">SafetyAntRun-v0</option>
     <!-- Add more options as needed -->
   </select>

   <style>
     #bullet-image-container {
       display: flex;
       justify-content: space-between;
     }
     #bullet-reward-image, #bullet-cost-image {
       max-width: 49%;
       height: auto;
     }
   </style>

   <div id="bullet-image-container">
     <img id="bullet-reward-image" src="../_static/images/bullet/SafetyCarCircle-v0-reward.png" alt="Reward image">
     <img id="bullet-cost-image" src="../_static/images/bullet/SafetyCarCircle-v0-cost.png" alt="Cost image">
   </div>

   <script src="../_static/js/benchmark.js"></script>


Safety-Gymnasium-Velocity-Tasks
-------------------------------

.. raw:: html

   <label for="select-velocity-env">Select the environment:</label>
   <select id="select-velocity-env">
     <option value="SafetyHalfCheetahVelocityGymnasium-v1">SafetyHalfCheetahVelocityGymnasium-v1</option>
     <option value="SafetyHopperVelocityGymnasium-v1">SafetyHopperVelocityGymnasium-v1</option>
     <option value="SafetySwimmerVelocityGymnasium-v1">SafetySwimmerVelocityGymnasium-v1</option>
     <option value="SafetyWalker2dVelocityGymnasium-v1">SafetyWalker2dVelocityGymnasium-v1</option>
     <option value="SafetyAntVelocityGymnasium-v1">SafetyAntVelocityGymnasium-v1</option>
     <!-- Add more options as needed -->
   </select>

   <style>
     #velocity-image-container {
       display: flex;
       justify-content: space-between;
     }
     #velocity-reward-image, #velocity-cost-image {
       max-width: 49%;
       height: auto;
     }
   </style>

   <div id="velocity-image-container">
     <img id="velocity-reward-image" src="../_static/images/safety-gymnasium-velocity/SafetyHalfCheetahVelocityGymnasium-v1-reward.png" alt="Reward image">
     <img id="velocity-cost-image" src="../_static/images/safety-gymnasium-velocity/SafetyHalfCheetahVelocityGymnasium-v1-cost.png" alt="Cost image">
   </div>

   <script src="../_static/js/benchmark.js"></script>


Safety-Gymnasium-Navigation-Tasks
---------------------------------

.. raw:: html

   <label for="select-navigation-env">Select the environment:</label>
   <select id="select-navigation-env">
     <option value="SafetyPointButton1Gymnasium-v0">SafetyPointButton1Gymnasium-v0</option>
     <option value="SafetyPointButton2Gymnasium-v0">SafetyPointButton2Gymnasium-v0</option>
     <option value="SafetyPointGoal1Gymnasium-v0">SafetyPointGoal1Gymnasium-v0</option>
     <option value="SafetyPointGoal2Gymnasium-v0">SafetyPointGoal2Gymnasium-v0</option>
     <option value="SafetyPointPush1Gymnasium-v0">SafetyPointPush1Gymnasium-v0</option>
     <option value="SafetyPointPush2Gymnasium-v0">SafetyPointPush2Gymnasium-v0</option>
     <!-- Add more options as needed -->
   </select>

   <style>
     #navigation-image-container {
       display: flex;
       justify-content: space-between;
     }
     #navigation-reward-image, #navigation-cost-image {
       max-width: 49%;
       height: auto;
     }
   </style>

   <div id="navigation-image-container">
     <img id="navigation-reward-image" src="../_static/images/safety-gymnasium-navigation/SafetyPointButton1Gymnasium-v0-reward.png" alt="Reward image">
     <img id="navigation-cost-image" src="../_static/images/safety-gymnasium-navigation/SafetyPointButton1Gymnasium-v0-cost.png" alt="Cost image">
   </div>

   <script src="../_static/js/benchmark.js"></script>
