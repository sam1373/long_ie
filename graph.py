class Graph(object):
    def __init__(self, entities, triggers, relations, roles, evidence=None,
                 coref_matrix=None, cluster_labels=None, cluster_labels_ev=None,
                 mentions=None):
        """
        :param entities (list): A list of entities represented as a tuple of
        (start_offset, end_offset, label_idx). end_offset = the index of the end
        token + 1.
        :param triggers (list): A list of triggers represented as a tuple of
        (start_offset, end_offset, label_idx). end_offset = the index of the end
        token + 1.
        :param relations (list): A list of relations represented as a tuple of
        (entity_idx_1, entity_idx_2, label_idx). As we do not
        consider the direction of relations (list), it is better to have
        entity_idx_1 < entity_idx_2.
        :param roles: A list of roles represented as a tuple of (trigger_idx_1,
        entity_idx_2, label_idx).
        :param vocabs (dict): Label type vocabularies.
        """
        self.entities = entities
        self.triggers = triggers
        self.relations = relations
        self.roles = roles
        self.coref_matrix = coref_matrix
        self.cluster_labels = cluster_labels
        self.cluster_labels_ev = cluster_labels_ev
        self.mentions = [] if mentions is None else mentions

        self.entity_num = len(entities)
        self.trigger_num = len(triggers)
        self.relation_num = len(relations)
        self.role_num = len(roles)

        self.sub_relations = []

    def __eq__(self, other):
        if isinstance(other, Graph):
            equal = (self.entities == other.entities and
                     self.triggers == other.triggers and
                     self.relations == other.relations and
                     self.roles == other.roles and
                     self.mentions == other.mentions and
                     self.sub_relations == other.sub_relations)
            return equal
        return False

    def to_dict(self):
        """Convert a graph to a dict object
        :return (dict): A dictionary representing the graph, where label indices
        have been replaced with label strings.
        """
        entities = self.entities
        triggers = self.triggers
        relations = self.relations
        roles = self.roles
        return {
            'entities': entities,
            'triggers': triggers,
            'relations': relations,
            'roles': roles,
            'coref_matrix': self.coref_matrix
        }

    def __str__(self):
        return str(self.to_dict())

    def copy(self):
        """Make a copy of the graph
        :return (Graph): a copy of the current graph.
        """
        graph = Graph(
            entities=self.entities.copy(),
            triggers=self.triggers.copy(),
            relations=self.relations.copy(),
            roles=self.roles.copy(),
            mentions=self.mentions.copy(),
        )
        graph.graph_local_score = self.graph_local_score
        # graph.sub_relations = self.sub_relations.copy()
        return graph

    def clean(self, relation_directional=False, symmetric_relations=None):
        # self.entities.sort(key=lambda x: (x[0], x[1]))
        # self.triggers.sort(key=lambda x: (x[0], x[1]))
        self.roles.sort(key=lambda x: (x[0], x[1]))

        # self.relations.sort(key=lambda x: (x[0], x[1]))
        relations = [(i, j, k) for i, j, k in self.relations]
        relations.sort(key=lambda x: (x[0], x[1]))

        # clean relations
        if relation_directional and symmetric_relations:
            relations_tmp = []
            for i, j, k in relations:
                if k not in symmetric_relations:
                    relations_tmp.append((i, j, k))
                else:
                    if j < i:
                        i, j = j, i
                    relations_tmp.append((i, j, k))
            relations = relations_tmp

        self.relations = relations

        return self

    @staticmethod
    def empty_graph():
        """Create a graph without any node and edge.
        :param vocabs (dict): Vocabulary object.
        """
        return Graph([], [], [], [])

    def to_label_idxs(self, max_entity_num, max_trigger_num,
                      relation_directional=False,
                      symmetric_relation_idxs=None):
        """Generate label index tensors (which are actually list objects not
        Pytorch tensors) to gather calculated scores.
        :param max_entity_num: Max entity number of the batch.
        :param max_trigger_num: Max trigger number of the batch.
        :return: Index and mask tensors.
        """
        entity_idxs = [i[-1] for i in self.entities] + [0] * (
                    max_entity_num - self.entity_num)
        entity_mask = [1] * self.entity_num + [0] * (
                    max_entity_num - self.entity_num)

        trigger_idxs = [i[-1] for i in self.triggers] + [0] * (
                    max_trigger_num - self.trigger_num)
        trigger_mask = [1] * self.trigger_num + [0] * (
                    max_trigger_num - self.trigger_num)

        relation_idxs = [0] * max_entity_num * max_entity_num
        relation_mask = [
            1 if i < self.entity_num and j < self.entity_num and i != j else 0
            for i in range(max_entity_num) for j in range(max_entity_num)]

        for i, j, relation in self.relations:
            # TODO: check relation label idxs and mask
            relation_idxs[i * max_entity_num + j] = relation
            if not relation_directional:
                relation_idxs[j * max_entity_num + i] = relation
            if relation_directional and symmetric_relation_idxs and relation in symmetric_relation_idxs:
                relation_idxs[j * max_entity_num + i] = relation

        role_idxs = [0] * max_trigger_num * max_entity_num
        for i, j, role in self.roles:
            role_idxs[i * max_entity_num + j] = role
        role_mask = [1 if i < self.trigger_num and j < self.entity_num else 0
                     for i in range(max_trigger_num) for j in
                     range(max_entity_num)]

        return (
            entity_idxs, entity_mask, trigger_idxs, trigger_mask,
            relation_idxs, relation_mask, role_idxs, role_mask,
        )

