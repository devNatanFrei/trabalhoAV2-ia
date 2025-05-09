adaline = Adaline(learning_rate=0.01, n_ephocs=1000)
# adaline.fit(x, y)

# mlp_under = MLP(n_hidden=3, learning_rate=0.01, n_ephocs=5000, random_state=42)
# mlp_under.fit(x, y)

# mlp_over = MLP(n_hidden=50, learning_rate=0.01, n_ephocs=5000, random_state=42)
# mlp_over.fit(x, y)

# # plot_learning_curve(
# #     [adaline, mlp_under, mlp_over],  
# #     ['Adaline', 'MLP Underfitting', 'MLP Overfitting'], 
# #     title='Learning Curve'
# # )

# print("Monte Carlo Validation Results")
# print()

# results_adaline = monte_carlo_validation(Adaline, x, y, R=250, learning_rate=0.01, n_ephocs=1000)
# results_mlp_under = monte_carlo_validation(MLP, x, y, R=250, n_hidden=3, learning_rate=0.01, n_ephocs=5000, random_state=42)
# results_mlp_over = monte_carlo_validation(MLP, x, y, R=250, n_hidden=50, learning_rate=0.01, n_ephocs=5000, random_state=42)

# print("Adaline Results:", results_adaline)
# print("MLP Underfitting Results:", results_mlp_under)
# print("MLP Overfitting Results:", results_mlp_over)