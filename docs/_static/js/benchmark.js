document.addEventListener("DOMContentLoaded", function () {
    var bulletSelectEnv = document.getElementById("select-bullet-env");
    var bulletRewardImage = document.getElementById("bullet-reward-image");
    var bulletCostImage = document.getElementById("bullet-cost-image");

    var velocitySelectEnv = document.getElementById("select-velocity-env");
    var velocityRewardImage = document.getElementById("velocity-reward-image");
    var velocityCostImage = document.getElementById("velocity-cost-image");

    var navigationSelectEnv = document.getElementById("select-navigation-env");
    var navigationRewardImage = document.getElementById("navigation-reward-image");
    var navigationCostImage = document.getElementById("navigation-cost-image");

    bulletSelectEnv.addEventListener("change", function () {
        var env = bulletSelectEnv.value;
        var newRewardSrc = "../_static/images/bullet/" + env + "-reward.png";
        var newCostSrc = "../_static/images/bullet/" + env + "-cost.png";
        bulletRewardImage.src = newRewardSrc;
        bulletCostImage.src = newCostSrc;
    });

    velocitySelectEnv.addEventListener("change", function () {
        var env = velocitySelectEnv.value;
        var newRewardSrc = "../_static/images/safety-gymnasium-velocity/" + env + "-reward.png";
        var newCostSrc = "../_static/images/safety-gymnasium-velocity/" + env + "-cost.png";
        velocityRewardImage.src = newRewardSrc;
        velocityCostImage.src = newCostSrc;
    });

    navigationSelectEnv.addEventListener("change", function () {
        var env = navigationSelectEnv.value;
        var newRewardSrc = "../_static/images/safety-gymnasium-navigation/" + env + "-reward.png";
        var newCostSrc = "../_static/images/safety-gymnasium-navigation/" + env + "-cost.png";
        navigationRewardImage.src = newRewardSrc;
        navigationCostImage.src = newCostSrc;
    });
});
