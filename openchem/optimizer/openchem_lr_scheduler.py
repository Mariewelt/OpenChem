class OpenChemLRScheduler(object):

    def __init__(self, params, optimizer):
        self.params = params[1]
        self._scheduler = params[0](optimizer, **self.params)

    @property
    def scheduler(self):
        return self._scheduler

    def step(self):
        """Performs a single scheduler step."""
        return self.scheduler.step()
