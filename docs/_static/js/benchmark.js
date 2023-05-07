document.addEventListener("DOMContentLoaded", function () {
    var selectEnv = document.getElementById("select-env");
    var rewardImage = document.getElementById("reward-image");
    var costImage = document.getElementById("cost-image");

    selectEnv.addEventListener("change", function () {
        var env = selectEnv.value;
        var newRewardSrc = "../_static/images/bullet/" + env + "-reward.png";
        var newCostSrc = "../_static/images/bullet/" + env + "-cost.png";
        // console.log("New reward src:", newRewardSrc);
        // console.log("New cost src:", newCostSrc);
        rewardImage.src = newRewardSrc;
        costImage.src = newCostSrc;
    });
});
