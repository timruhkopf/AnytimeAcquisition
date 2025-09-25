from src.callbacks.abstract_callback import AbstractCallback


class PlotTrajectoriesCallback(AbstractCallback):
    def __init__(self, plot_every_n_epochs=10000):
        super().__init__()
        self.plot_every_n_epochs = plot_every_n_epochs
        self.fig = None
        self.plotted=False

    def on_epoch_end(self):
        if (self.epoch + 1) % self.plot_every_n_epochs == 0:
            n_traces = 10

            # self.env.resample()
            initial_condition = self.trainer.env.sample_initial_condition(B=n_traces)
            X = self.trainer.model.generate(
                self.trainer.env, B=n_traces, T=25,
                initial_condition=initial_condition
            )
            self.trainer.env.plot3d_matplotlib(traces=X)

            # self.fig = self.trainer.env.plot3d(fig=self.fig, traces=X, static=True)
            # if not self.plotted:
            #     self.fig.show()
            #     self.plotted=True
            #
            # print()
            # self.fig.data = []
            # self.fig.layout = {}

            #
            if False and hasattr(self.trainer.model, 'explore'):
                n_traces = 1000
                T = 100

                # self.env.resample()
                initial_condition = self.trainer.env.sample_initial_condition(B=n_traces)
                exploration_X = self.trainer.model.explore(

                    self.epoch, self.trainer.env, B=n_traces, T=T

                    # initial_condition=initial_condition
                )

                # self.trainer.env.plot3d_matplotlib(traces=exploration_X)

                import torch
                inc = torch.cummin(exploration_X[:, :, -1], dim=1)

                inc.values

                import matplotlib.pyplot as plt
                import seaborn as sns
                import pandas as pd


                # (Residuals plot on y vs time) --------------------
                # will be computed on the entire batch & alternatives ==1
                t = torch.linspace(1, T, T)

                residuals_flat = inc.values.view(-1, T)  # (B, T)
                # Convert to long dataframe format for seaborn plotting
                data = []
                for t in range(T):
                    for val in residuals_flat[:, t].cpu().detach().numpy():
                        data.append((t + 1, val))  # t+1 for 1-based time indexing

                df = pd.DataFrame(data, columns=['Time', 'Residual'])

                # Plot distribution of residuals over time with violin plot
                plt.figure(figsize=(14, 6))
                sns.violinplot(x='Time', y='Residual', data=df, inner="quartile", scale='width')
                plt.title('Distribution of Residuals over Time')
                plt.xlabel('Time step (t)')
                plt.ylabel('Residual value')
                plt.xticks(rotation=45)
                plt.grid(True, linestyle='--', alpha=0.5)
                plt.show()


            
            