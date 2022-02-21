class BasicLinker(object):

    def __init__(self):
        # type: () -> None
        """
        Generate natural language based on structured data

        Parameters
        ----------
        """

        # self._log = logger.getChild(self.__class__.__name__)
        # self._log.info("Booted")

    def link_entities(self, capsule):
        raise NotImplementedError()

    def link_predicates(self, capsule):
        raise NotImplementedError()

    def link(self, capsule):
        raise NotImplementedError()
