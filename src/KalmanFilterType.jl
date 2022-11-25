abstract type KalmanFilterMethod end

struct KalmanFilter{kfmethod <: KalmanFilterMethod}
    method::kfmethod
end