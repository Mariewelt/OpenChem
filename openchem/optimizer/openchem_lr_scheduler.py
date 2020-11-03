class OpenChemLRScheduler(object):
    def __init__(self, params, optimizer):
        self.params = params[1]
        self._by_iter = self.params.pop("by_iter", False)
        self._scheduler = params[0](optimizer, **self.params)

    @property
    def scheduler(self):
        return self._scheduler

    @property
    def by_iteration(self):
        """
        If true, make step every iteration, else every epoch (default)
        """
        return self._by_iter

    def step(self):
        """Performs a single scheduler step."""
        return self.scheduler.step()
