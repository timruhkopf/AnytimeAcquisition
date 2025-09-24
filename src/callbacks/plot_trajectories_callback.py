from src.callbacks.abstract_callback import AbstractCallback


class PlotTrajectoriesCallback(AbstractCallback):
    def __init__(self, plot_every_n_epochs=20):
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
            # TODO check how i can do the plotting on my local, despite plotly

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
            # if hasattr(self.trainer.model, 'explore'):
            #     exploration_X = self.trainer.model.explore(
            #         self.trainer.env, B=n_traces, T=25,
            #         initial_condition=initial_condition
            #     )
            #
            #     self.trainer.env.plot3d(traces=exploration_X)
            
            