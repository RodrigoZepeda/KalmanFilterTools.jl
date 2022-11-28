using KalmanFilterTools, Plots

#First version of Kalman
A = [[0.20, 0.2] [0.5, 0.8]]
B = [[0.01, 0.5, 0.7] [0.8, 0.1, 0.2]]'
C = [[0.71, 0.3] [0.1, 0.3] [0.08, 0.5] [0.2, 0.21]]'
D = [[0.4, 0.1, 0.7] [0.3, 0.25, 0.01] [0.0, 0.1, 0.0] [0.2, 0.37, 0.01]]'
x,y = SimulateKalman(1000, A, B, C, D)

plot(1:size(y,2), y', 
    plot_titlelocation=:left,
    plot_titlefontvalign=:bottom,
    plot_titlevspan=0.2,
    foreground_color_grid = "gray75",
    gridstyle=:dot,
    gridlinewidth=1.0,
    gridalpha=0.5,
    grid=:xy,
    plot_title = "\n\nKalman filter simulations", 
    background_color="#FAF6EC",
    xlabel = "Time\n", 
    ylabel = "\ny(t)",
    legend = false,
    formatter=:plain,
    tick_direction=:out,
    xguidefonthalign = :right, 
    yguidefontvalign = :top,
    palette = :Dark2_5)

    using KalmanFilterTools, Plots

#First version of Kalman
x,y = SimulateKalman(100, m = 4, q = 5, n = 7)

plot(1:size(y,2), y', 
    plot_titlelocation=:left,
    plot_titlefontvalign=:bottom,
    plot_titlevspan=0.2,
    foreground_color_grid = "gray75",
    gridstyle=:dot,
    gridlinewidth=1.0,
    gridalpha=0.5,
    grid=:xy,
    plot_title = "\n\nKalman filter simulations", 
    background_color="#FAF6EC",
    xlabel = "Time\n", 
    ylabel = "\ny(t)",
    legend = false,
    formatter=:plain,
    tick_direction=:out,
    xguidefonthalign = :right, 
    yguidefontvalign = :top,
    palette = :Dark2_5)