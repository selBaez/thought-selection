from cltl.brain.infrastructure.rdf_builder import RdfBuilder

from src.entity_linking.api import BasicLinker


class LabelBasedLinker(BasicLinker):

    def __init__(self):
        """
        OIE Analyzer Object

        Parameters
        ----------
        """

        super(BasicLinker, self).__init__()
        self._rdf_builder = RdfBuilder()

    def link(self, capsule):
        capsule = self.link_entities(capsule)
        capsule = self.link_predicates(capsule)

        return capsule

    def link_entities(self, capsule):
        capsule['subject']['uri'] = str(self._rdf_builder.create_resource_uri('LW', capsule['subject']['label'].lower()))
        capsule['object']['uri'] = str(self._rdf_builder.create_resource_uri('LW', capsule['object']['label'].lower()))

        return capsule

    def link_predicates(self, capsule):
        capsule['predicate']['uri'] = str(self._rdf_builder.create_resource_uri('N2MU', capsule['predicate']['label'].lower()))

        return capsule
