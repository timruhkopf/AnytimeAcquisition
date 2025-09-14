from src.callbacks.abstract_callback import AbstractCallback


class PlotTrajectoriesCallback(AbstractCallback):
    def __init__(self, plot_every_n_epochs=1000):
        super().__init__()
        self.plot_every_n_epochs = plot_every_n_epochs

    def on_epoch_end(self):
        if (self.epoch + 1) % self.plot_every_n_epochs == 0:
            n_traces = 20

            initial_condition = self.trainer.env.sample_initial_condition(B=n_traces)
            X = self.trainer.model.generate(
                self.trainer.env, B=n_traces, T=25,
                initial_condition=initial_condition
            )

            self.trainer.env.plot3d(traces=X)
            print()
            #
            # if hasattr(self.trainer.model, 'explore'):
            #     exploration_X = self.trainer.model.explore(
            #         self.trainer.env, B=n_traces, T=25,
            #         initial_condition=initial_condition
            #     )
            #
            #     self.trainer.env.plot3d(traces=exploration_X)