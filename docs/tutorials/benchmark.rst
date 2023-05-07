.. _benchmark:

Benchmarks
============

We conducted experiments in both the `Bullet-Safety-Gym <https://github.com/liuzuxin/Bullet-Safety-Gym>`_ and the `SafetyGymnasium <https://github.com/OmniSafeAI/safety-gymnasium>`_ environments.
All the experiments use the default configuration parameters as shown in the ``examples/mlp`` training scripts. The result curves are averaged over 3 seeds.

Bullet-Safety-Gym
-----------------

You can use the drop-down menu to check the reward and cost curves.

.. raw:: html

   <label for="select-env">Select the environment:</label>
   <select id="select-env">
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
     #image-container {
       display: flex;
       justify-content: space-between;
     }
     #reward-image, #cost-image {
       max-width: 49%;
       height: auto;
     }
   </style>

   <div id="image-container">
     <img id="reward-image" src="../_static/images/bullet/SafetyCarCircle-v0-reward.png" alt="Reward image">
     <img id="cost-image" src="../_static/images/bullet/SafetyCarCircle-v0-cost.png" alt="Cost image">
   </div>

   <script src="../_static/js/benchmark.js"></script>


Safety-Gymnasium-Velocity-Tasks
-------------------------------

To be updated...

