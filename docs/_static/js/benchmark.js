document.addEventListener("DOMContentLoaded", function () {
    var selectEnv = document.getElementById("select-env");
    var rewardImage = document.getElementById("reward-image");
    var costImage = document.getElementById("cost-image");

    selectEnv.addEventListener("change", function () {
        var env = selectEnv.value;
        var newRewardSrc = "";

        if (env.includes("Velocity")) {
            newRewardSrc = "../_static/images/safety-gymnasium-velocity/" + env + "-reward.png";
        } else if (env.includes("Gymnasium")) {
            newRewardSrc = "../_static/images/safety-gymnasium-navigation/" + env + "-reward.png";
        } else {
            newRewardSrc = "../_static/images/bullet/" + env + "-reward.png";
        }

        var newCostSrc = newRewardSrc.replace("-reward.png", "-cost.png");

        rewardImage.src = newRewardSrc;
        costImage.src = newCostSrc;
    });
});
