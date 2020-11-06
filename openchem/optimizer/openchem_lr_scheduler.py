class OpenChemLRScheduler(object):
    def __init__(self, params, optimizer):
        if params[0] is not None:
            self.params = params[1]
            self._by_iter = self.params.pop("by_iter", False)
            self._scheduler = params[0](optimizer, **self.params)
        else:
            self._scheduler = None
            self._by_iter = False

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
        if self._scheduler is not None:
            return self.scheduler.step()
